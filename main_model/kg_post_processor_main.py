"""Core graph retrieval module used by the proposed model.

Pipeline implemented here:
1. Expand a KG from entities found in vector-retrieved seed chunks.
2. Score each newly discovered edge for direct query relevance with an LLM.
3. Run edge-weighted Personalized PageRank with a query-aware seed restart.
4. Stop expansion only when PPR is concentrated and the correctness guards pass.
5. Build and rerank the final chunk/KG-fact candidates under explicit budgets.

This paper-facing copy intentionally excludes baseline and ablation processors.
"""

import json
import re
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, TextNode


def _query_text(query_like: Any) -> str:
    if query_like is None:
        return ""
    if isinstance(query_like, str):
        return query_like
    value = getattr(query_like, "query_str", None)
    return value if isinstance(value, str) else str(query_like)


class LLMSemanticEdgeScorerv2:
    """Score the direct relevance of individual KG edges to the question."""

    def __init__(self, llm: Any, batch_size: int = 8):
        self.llm = llm
        self.batch_size = batch_size
        self.cache: Dict[str, float] = {}

    @staticmethod
    def _cache_key(query: str, edge: Tuple[str, str, str]) -> str:
        head, relation, tail = edge
        return f"v2::{query}__[{head}] - ({relation}) - [{tail}]"

    @staticmethod
    def _parse_scores(raw: str, expected: int) -> List[float]:
        raw = str(raw).strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [float(value) for value in parsed[:expected]]
        except Exception:
            pass
        numbers = re.findall(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)", raw)
        return [float(value) for value in numbers[:expected]]

    def compute_scores_batch(
        self,
        query: str,
        edges: List[Tuple[str, str, str]],
    ) -> List[float]:
        if not edges:
            return []

        scores = [0.35] * len(edges)
        missing_indices: List[int] = []
        missing_edges: List[Tuple[str, str, str]] = []
        for index, edge in enumerate(edges):
            key = self._cache_key(query, edge)
            if key in self.cache:
                scores[index] = self.cache[key]
            else:
                missing_indices.append(index)
                missing_edges.append(edge)

        for start in range(0, len(missing_edges), self.batch_size):
            batch_indices = missing_indices[start:start + self.batch_size]
            batch_edges = missing_edges[start:start + self.batch_size]
            prompt = (
                "You score knowledge-graph edges by direct single-edge relevance to a question.\n"
                f"Question: {query}\n\n"
                "For each edge, give a relevance score from 0.0 to 1.0.\n"
                "Score high if the single edge itself is semantically related to the question wording, mentioned entities, or asked relation.\n"
                "Score low if the single edge is not directly related to the question, even if it might become useful in a longer reasoning path.\n"
                "Do not infer hidden multi-hop paths. Judge each edge independently.\n"
                "Return only a JSON array of numbers in the same order. No explanation.\n\n"
                "Edges:\n"
            )
            for edge_index, (head, relation, tail) in enumerate(batch_edges, 1):
                prompt += f"{edge_index}. {head} -- {relation} -- {tail}\n"

            try:
                response = self.llm.complete(prompt).text.strip()
                parsed = self._parse_scores(response, len(batch_edges))
            except Exception as exc:
                print(f"[LLMSemanticEdgeScorerv2] scoring failed: {exc}")
                parsed = []

            for local_index, original_index in enumerate(batch_indices):
                value = parsed[local_index] if local_index < len(parsed) else 0.35
                value = max(0.01, min(1.0, float(value)))
                scores[original_index] = value
                self.cache[self._cache_key(query, batch_edges[local_index])] = value

        return scores


class AdaptiveLLMSemanticPPRGraphFilterv2(BaseNodePostprocessor):
    """Adaptive LLM-weighted Personalized PageRank KG retriever."""

    doc2kg: Dict[str, Any]
    ents: Set[str]
    chunks_index: Dict[str, Any]
    llm_scorer: Any
    reranker: Any

    # top_k is a compatibility field for the vector seed count. It does not
    # truncate final context selection.
    top_k: int = 10
    max_hops: int = 3
    damping_factor: float = 0.85
    restart_top_n: int = 5
    ppr_keep_n: int = 20
    relevance_keep_n: int = 10
    final_ppr_mass: float = 0.90
    min_final_nodes: int = 10
    max_final_nodes: int = 50
    max_context_tokens: int = 3000
    convergence_min_hop: int = 2
    enable_early_stop: bool = True
    latest_reasoning_path: str = ""
    latest_debug: Dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def _node_entity(node: NodeWithScore) -> str:
        return str(node.node.id_).split("##")[0]

    @staticmethod
    def _as_score_list(scores: Any) -> List[float]:
        if isinstance(scores, list):
            return [float(value) for value in scores]
        if hasattr(scores, "tolist"):
            values = scores.tolist()
            if isinstance(values, list):
                return [float(value) for value in values]
        return [float(scores)]

    def _score_entities(
        self,
        query: str,
        entities: List[str],
        top_n: int,
    ) -> List[Tuple[str, float]]:
        ordered = list(dict.fromkeys(str(entity) for entity in entities if entity))
        if not ordered:
            return []
        try:
            raw_scores = self.reranker.compute_score(
                [[query, entity] for entity in ordered]
            )
            values = self._as_score_list(raw_scores)
            scored = [
                (entity, float(values[index]))
                for index, entity in enumerate(ordered[:len(values)])
            ]
            scored.extend((entity, 0.0) for entity in ordered[len(scored):])
        except Exception:
            scored = [(entity, 0.0) for entity in ordered]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_n]

    @staticmethod
    def _normalize_scores(scored: List[Tuple[str, float]]) -> Dict[str, float]:
        if not scored:
            return {}
        values = [float(score) for _, score in scored]
        low, high = min(values), max(values)
        if abs(high - low) < 1e-8:
            return {entity: 1.0 / len(scored) for entity, _ in scored}
        raw = {
            entity: (float(score) - low) / (high - low) + 1e-4
            for entity, score in scored
        }
        total = sum(raw.values()) or 1.0
        return {entity: value / total for entity, value in raw.items()}

    def _build_personalization(
        self,
        query: str,
        graph: nx.Graph,
        seed_entities: Set[str],
    ) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:
        candidates = [entity for entity in seed_entities if entity in graph]
        if not candidates:
            candidates = list(graph.nodes())
        restart_scored = self._score_entities(query, candidates, self.restart_top_n)
        weights = self._normalize_scores(restart_scored)
        if not weights and candidates:
            weights = {candidates[0]: 1.0}
            restart_scored = [(candidates[0], 1.0)]
        personalization = {node: 0.0 for node in graph.nodes()}
        for entity, weight in weights.items():
            if entity in personalization:
                personalization[entity] = weight
        return personalization, restart_scored

    def _query_relevance_map(
        self,
        query: str,
        entities: List[str],
    ) -> Dict[str, float]:
        return dict(self._score_entities(query, entities, len(entities)))

    @staticmethod
    def _k90(pr_scores: Dict[str, float]) -> Tuple[int, int]:
        sorted_scores = sorted(pr_scores.values(), reverse=True)
        total = sum(sorted_scores)
        if total <= 0:
            return 0, len(sorted_scores)
        cumulative = 0.0
        for index, score in enumerate(sorted_scores, 1):
            cumulative += score
            if cumulative >= 0.9 * total:
                return index, len(sorted_scores)
        return len(sorted_scores), len(sorted_scores)

    def _should_stop(
        self,
        hop: int,
        pr_scores: Dict[str, float],
        restart_scored: List[Tuple[str, float]],
        query_relevance: Dict[str, float],
        previous_top: List[str],
    ) -> Tuple[bool, str]:
        k90, total_nodes = self._k90(pr_scores)
        if hop < self.convergence_min_hop or total_nodes <= 5:
            return False, f"continue_min_hop_not_met(k90={k90}/{total_nodes})"
        if k90 > 0.2 * total_nodes:
            return False, f"continue_not_concentrated(k90={k90}/{total_nodes})"

        top_ppr = [
            entity
            for entity, _ in sorted(
                pr_scores.items(), key=lambda item: item[1], reverse=True
            )[:self.ppr_keep_n]
        ]
        restart_entities = {
            entity for entity, _ in restart_scored[:self.restart_top_n]
        }
        relevant_entities = {
            entity
            for entity, _ in sorted(
                query_relevance.items(), key=lambda item: item[1], reverse=True
            )[:self.restart_top_n]
        }
        restart_covered = bool(restart_entities.intersection(top_ppr))
        relevant_covered = bool(relevant_entities.intersection(top_ppr))
        stable = bool(
            previous_top
            and len(set(previous_top[:10]).intersection(top_ppr[:10]))
            >= max(3, len(previous_top[:10]) // 2)
        )
        if restart_covered and (relevant_covered or stable):
            return True, f"stop_concentrated_and_supported(k90={k90}/{total_nodes})"
        return False, f"continue_correctness_guard(k90={k90}/{total_nodes})"

    def _collect_edges_for_entities(
        self,
        entities: Set[str],
    ) -> Tuple[List[Tuple[str, str, str]], Set[str]]:
        edges: List[Tuple[str, str, str]] = []
        next_entities: Set[str] = set()
        for entity in entities:
            for triples in self.doc2kg.get(entity, {}).values():
                for head, relation, tail in triples:
                    head, relation, tail = map(lambda value: str(value).strip(), (head, relation, tail))
                    if not head or not tail:
                        continue
                    edges.append((head, relation, tail))
                    next_entities.update((head, tail))
        return edges, next_entities

    def _add_scored_edges(
        self,
        graph: nx.Graph,
        query: str,
        edges: List[Tuple[str, str, str]],
    ) -> None:
        unseen = [
            edge for edge in edges
            if not (graph.has_edge(edge[0], edge[2]) and "weight" in graph[edge[0]][edge[2]])
        ]
        if not unseen:
            return
        scores = self.llm_scorer.compute_scores_batch(query, unseen)
        for (head, relation, tail), score in zip(unseen, scores):
            weight = max(float(score), 0.01)
            graph.add_edge(
                head,
                tail,
                relation=relation,
                weight=weight,
                distance=1.0 / weight,
            )

    def _expand_context_candidates(
        self,
        seed_nodes: List[NodeWithScore],
        entities: List[str],
        graph: nx.Graph,
    ) -> List[NodeWithScore]:
        candidates = list(seed_nodes)
        existing_ids = {node.node.id_ for node in candidates}
        for entity in entities:
            for sequence, text in self.chunks_index.get(entity, {}).items():
                node_id = f"{entity}##{sequence}"
                if node_id not in existing_ids:
                    candidates.append(
                        NodeWithScore(node=TextNode(text=text, id_=node_id), score=0.0)
                    )
                    existing_ids.add(node_id)

        for fact_index, (head, tail, data) in enumerate(graph.edges(data=True)):
            if head not in entities and tail not in entities:
                continue
            node_id = f"kgfact##{fact_index}"
            if node_id in existing_ids:
                continue
            relation = data.get("relation", "related to")
            text = f"KG fact: {head} -- {relation} -- {tail}"
            candidates.append(
                NodeWithScore(node=TextNode(text=text, id_=node_id), score=0.0)
            )
            existing_ids.add(node_id)
        return candidates

    @staticmethod
    def _token_len(text: str) -> int:
        return max(1, len(str(text or "").split())) if text else 0

    def _select_entities_by_ppr_mass(
        self,
        pr_scores: Dict[str, float],
    ) -> List[str]:
        ranked = sorted(pr_scores.items(), key=lambda item: item[1], reverse=True)
        total = sum(score for _, score in ranked) or 1.0
        cumulative = 0.0
        selected: List[str] = []
        for entity, score in ranked:
            selected.append(entity)
            cumulative += float(score)
            if (
                len(selected) >= self.min_final_nodes
                and cumulative >= self.final_ppr_mass * total
            ):
                break
            if len(selected) >= self.max_final_nodes:
                break
        return selected

    def _select_final_nodes(
        self,
        query: str,
        candidates: List[NodeWithScore],
    ) -> List[NodeWithScore]:
        if not candidates:
            return candidates
        try:
            values = self._as_score_list(
                self.reranker.compute_score(
                    [[query, node.node.text] for node in candidates]
                )
            )
            for index, node in enumerate(candidates[:len(values)]):
                node.score = float(values[index])
        except Exception:
            pass
        candidates.sort(key=lambda node: node.score, reverse=True)

        selected: List[NodeWithScore] = []
        used_tokens = 0
        for node in candidates:
            node_tokens = self._token_len(getattr(node.node, "text", ""))
            over_budget = (
                self.max_context_tokens > 0
                and used_tokens + node_tokens > self.max_context_tokens
            )
            if selected and over_budget and len(selected) >= self.min_final_nodes:
                break
            selected.append(node)
            used_tokens += node_tokens
            if len(selected) >= self.max_final_nodes:
                break
        return selected or candidates[:1]

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Any,
    ) -> List[NodeWithScore]:
        query = _query_text(query_bundle)
        self.latest_reasoning_path = ""
        self.latest_debug = {}

        seed_entities = {
            self._node_entity(node)
            for node in nodes
            if self._node_entity(node) in self.doc2kg
        }
        current_entities = set(seed_entities)
        visited_entities: Set[str] = set()
        graph = nx.Graph()
        pr_scores: Dict[str, float] = {}
        restart_scored: List[Tuple[str, float]] = []
        previous_top: List[str] = []
        stop_reason = "max_hops"

        for hop in range(1, self.max_hops + 1):
            expand_entities = current_entities - visited_entities
            if not expand_entities:
                break
            visited_entities.update(expand_entities)
            edges, next_entities = self._collect_edges_for_entities(expand_entities)
            self._add_scored_edges(graph, query, edges)
            if graph.number_of_nodes() == 0:
                break

            personalization, restart_scored = self._build_personalization(
                query, graph, seed_entities
            )
            try:
                pr_scores = nx.pagerank(
                    graph,
                    alpha=self.damping_factor,
                    personalization=personalization,
                    weight="weight",
                    max_iter=100,
                )
            except Exception as exc:
                print(f"[AdaptiveLLMSemanticPPRGraphFilterv2] PageRank failed at hop {hop}: {exc}")
                break

            query_relevance = self._query_relevance_map(query, list(graph.nodes()))
            top_now = [
                entity
                for entity, _ in sorted(
                    pr_scores.items(), key=lambda item: item[1], reverse=True
                )[:self.ppr_keep_n]
            ]
            if self.enable_early_stop:
                should_stop, stop_reason = self._should_stop(
                    hop,
                    pr_scores,
                    restart_scored,
                    query_relevance,
                    previous_top,
                )
            else:
                should_stop = False
                stop_reason = f"fixed_hop_progress({hop}/{self.max_hops})"

            self.latest_debug = {
                "version": "v2",
                "hop": hop,
                "stop_reason": stop_reason,
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "restart_entities": restart_scored,
                "top_ppr_entities": [
                    (entity, pr_scores[entity]) for entity in top_now[:10]
                ],
            }
            if should_stop:
                break
            previous_top = top_now
            current_entities = next_entities - visited_entities

        if not pr_scores:
            return nodes

        ppr_entities = self._select_entities_by_ppr_mass(pr_scores)
        query_relevance = self._query_relevance_map(query, list(graph.nodes()))
        relevant_entities = [
            entity
            for entity, _ in sorted(
                query_relevance.items(), key=lambda item: item[1], reverse=True
            )[:self.relevance_keep_n]
        ]
        restart_entities = [
            entity for entity, _ in restart_scored[:self.restart_top_n]
        ]
        final_entities = list(
            dict.fromkeys(restart_entities + ppr_entities + relevant_entities)
        )

        anchor = restart_entities[0] if restart_entities else ppr_entities[0]
        paths: List[str] = []
        if anchor in graph:
            for target in final_entities[:8]:
                if target == anchor or target not in graph:
                    continue
                try:
                    path = nx.shortest_path(
                        graph, source=anchor, target=target, weight="distance"
                    )
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                parts = []
                for index in range(len(path) - 1):
                    relation = graph[path[index]][path[index + 1]].get(
                        "relation", "related to"
                    )
                    parts.append(
                        f"[{path[index]}]-({relation})-[{path[index + 1]}]"
                    )
                if parts:
                    paths.append(" -> ".join(parts))
        self.latest_reasoning_path = "\n".join(dict.fromkeys(paths))

        candidates = self._expand_context_candidates(nodes, final_entities, graph)
        selected = self._select_final_nodes(query, candidates)
        self.latest_debug.update(
            {
                "final_ppr_mass": self.final_ppr_mass,
                "ppr_mass_entities": len(ppr_entities),
                "final_entities": len(final_entities),
                "candidate_context_nodes": len(candidates),
                "selected_context_nodes": len(selected),
                "max_final_nodes": self.max_final_nodes,
                "max_context_tokens": self.max_context_tokens,
            }
        )
        return selected
