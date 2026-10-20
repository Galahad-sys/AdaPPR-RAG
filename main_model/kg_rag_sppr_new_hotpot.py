# -*- coding: utf-8 -*-
"""
kg_rag_sppr_new.py
融合顶级学术创新的 KG-RAG 流程：
1. 大模型 (LLM) 判别器驱动的动态图边权
2. 90/20 洛伦兹曲线收敛性自适应增量扩展
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm
import traceback

from FlagEmbedding import FlagReranker
from llama_index.llms.ollama import Ollama
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    PromptTemplate,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
import tiktoken

# 🚨 导入刚刚新增的两个自适应与大模型模块
from kg_post_processor_main import (
    LLMSemanticEdgeScorerv2,
    AdaptiveLLMSemanticPPRGraphFilterv2,
)
from kg_response_synthesizer import get_response_synthesizer


# --------- 此处保持和之前文件一样的 Embedding 截断包装 ---------
class SafeOllamaEmbedding(OllamaEmbedding):
    def __init__(self, *args, max_chars: int = 800, min_chars: int = 80, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_chars = max_chars
        self._min_chars = min_chars

    def _truncate_text(self, text: str, max_chars: int) -> str:
        if text is None: return ""
        text = str(text).strip()
        if len(text) <= max_chars: return text
        cut = text[:max_chars]
        for sep in ["。", "；", ";", "\n", "，", ",", " "]:
            pos = cut.rfind(sep)
            if pos >= int(max_chars * 0.6): return cut[:pos].strip()
        return cut.strip()

    def get_general_text_embedding(self, text: str):
        cur = self._max_chars
        while True:
            try:
                return super().get_general_text_embedding(self._truncate_text(text, cur))
            except Exception as e:
                msg = str(e).lower()
                if ("exceeds the context length" in msg) or ("status code: 400" in msg):
                    nxt = max(self._min_chars, cur // 2)
                    if nxt == cur: raise
                    cur = nxt
                    continue
                raise

    def _get_text_embeddings(self, texts):
        return [self.get_general_text_embedding(t) for t in texts]


# --------- 辅助工具函数 ---------
def sanitize_filename(name: str, max_len: int = 100) -> str:
    if name is None: return ""
    clean_name = re.sub(r'[\\/:*?"<>|\n\r\t]', "_", str(name))
    clean_name = re.sub(r'[_ ]+', "_", clean_name).strip('_')
    return clean_name[:max_len]


def load_knowledge_points(data_path: str) -> Tuple[List[str], Dict[str, List[str]]]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ent2paras: Dict[str, List[str]] = {}

    def add(ent: str, paras: List[str]):
        if not ent: return
        if ent not in ent2paras: ent2paras[ent] = []
        for p in paras or []:
            if p: ent2paras[ent].append(str(p).strip())

    seen_hotpot_titles = set()
    if isinstance(data, list):
        for item in data:
            if "context" in item:
                for ctx in item.get("context", []) or []:
                    if not isinstance(ctx, list) or len(ctx) != 2:
                        continue
                    title, sentences = ctx
                    title = str(title) if title else ""
                    if not title or title in seen_hotpot_titles:
                        continue
                    seen_hotpot_titles.add(title)
                    add(title, sentences or [])
            else:
                title = item.get("title")
                segments = item.get("segments", [])
                raw = item.get("raw", "")
                paras = segments if segments else ([raw] if raw else [])
                add(str(title) if title else "", paras)
    return list(ent2paras.keys()), ent2paras


def load_doc2kg(kg_dir: str, ents: List[str]) -> Dict[str, Dict[str, List[List[str]]]]:
    doc2kg = {}
    kg_dir_p = Path(kg_dir)
    for ent in ents:
        safe_ent = sanitize_filename(ent)
        fp = kg_dir_p / f"{safe_ent}.json"
        if fp.exists():
            with open(fp, "r", encoding="utf-8") as f:
                subkg = json.load(f)
                cleaned = {str(k): [[str(x).strip() for x in tri] for tri in v if len(tri) == 3]
                           for k, v in subkg.items() if isinstance(v, list)}
                if cleaned: doc2kg[ent] = cleaned
    return doc2kg


def build_chunks(ent2paras: Dict[str, List[str]]) -> Tuple[List[TextNode], Dict[str, Dict[str, str]], set]:
    doc_chunks, chunks_index, ents_set = [], {}, set(ent2paras.keys())
    for ent, paras in ent2paras.items():
        if not ent: continue
        chunks_index[ent] = {}
        for i, para in enumerate(paras):
            if not para: continue
            text, node_id = f"{ent}: {para}", f"{ent}##{i}"
            doc_chunks.append(TextNode(text=text, id_=node_id))
            chunks_index[ent][str(i)] = text
    return doc_chunks, chunks_index, ents_set


def load_qa_pairs(q_path: str, a_path: str) -> Tuple[List[str], List[Any]]:
    with open(q_path, "r", encoding="utf-8") as f:
        qs = [line.strip() for line in f if line.strip()]
    with open(a_path, "r", encoding="utf-8") as f:
        ans = json.load(f)
    return qs, ans


def normalize_result_text(s: str) -> str:
    if s is None: return ""
    s = re.sub(r"^```.*?\n", "", str(s), flags=re.DOTALL)
    s = re.sub(r"\n```$", "", s, flags=re.DOTALL)
    s = s.strip().strip('"').strip("'").strip()
    s = re.sub(r"\s+", " ", s).strip()
    if s.lower() in {"yes", "no", "noanswer"}:
        return s.lower()
    return s


def count_final_context_tokens(resp: Any, tokenizer) -> int:
    """Count tokens in the final context nodes sent to the QA prompt."""
    context_parts = []
    for source_node in getattr(resp, "source_nodes", []) or []:
        node = getattr(source_node, "node", source_node)
        try:
            text = node.get_content()
        except Exception:
            text = getattr(node, "text", "")
        if text:
            context_parts.append(str(text))
    final_context = "\n\n".join(context_parts)
    return len(tokenizer.encode(final_context)) if final_context else 0


def count_final_context_nodes(resp: Any) -> int:
    return len(getattr(resp, "source_nodes", []) or [])


# --------- 主程序 ---------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=r"D:\vscode\KG2RAG-main\data\hotpotqa\hotpot_dev_distractor_small.json")
    parser.add_argument("--kg_dir", type=str, default=r"D:\vscode\KG2RAG-main\data\hotpotqa\kgs_small")
    parser.add_argument("--persist_dir", type=str, default=r"D:\vscode\KG2RAG-main\data\hotpotqa\ollama_index_small")
    parser.add_argument("--questions_path", type=str, default=r"D:\vscode\KG2RAG-main\data\hotpotqa\hotpot_questions_small.txt")
    parser.add_argument("--answers_path", type=str, default=r"D:\vscode\KG2RAG-main\data\hotpotqa\hotpot_answers_small.json")

    # 🚨 新的输出结果路径
    parser.add_argument("--out_path", type=str, default=r"D:\vscode\KG2RAG-main\data\hotpotqa\hotpot_predictions_small_adaptive_llm_v2.json")
    parser.add_argument("--llm_model", type=str, default="llama3:8b")
    parser.add_argument("--embed_model", type=str, default="mxbai-embed-large")
    parser.add_argument("--seed_top_k", type=int, default=10)
    parser.add_argument("--final_ppr_mass", type=float, default=0.90)
    parser.add_argument("--max_final_nodes", type=int, default=50)
    parser.add_argument("--max_context_tokens", type=int, default=3000)
    parser.add_argument("--reranker_path", type=str, default=r"D:\vscode\KG2RAG-main\model\bge-reranker-large")
    parser.add_argument("--llm_timeout", type=int, default=200)
    parser.add_argument("--test_limit", type=int, default=0)
    args = parser.parse_args()

    ents, ent2paras = load_knowledge_points(args.data_path)
    doc_chunks, chunks_index, ents_set = build_chunks(ent2paras)
    doc2kg = load_doc2kg(args.kg_dir, list(ents_set))

    # Token监听器与LLM设置
    token_encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    token_counter = TokenCountingHandler(tokenizer=token_encoding.encode)
    Settings.callback_manager = CallbackManager([token_counter])
    Settings.llm = Ollama(model=args.llm_model, request_timeout=args.llm_timeout)
    Settings.embed_model = SafeOllamaEmbedding(model_name=args.embed_model)
    bge_reranker = FlagReranker(model_name_or_path=args.reranker_path)

    pdir = Path(args.persist_dir)
    if pdir.exists() and any(pdir.iterdir()):
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=args.persist_dir))
    else:
        index = VectorStoreIndex(doc_chunks, show_progress=True)
        index.storage_context.persist(persist_dir=args.persist_dir)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=args.seed_top_k)

    qa_prompt = PromptTemplate(
        "You are a question-answering system. Answer the question using the provided Context.\n\n"
        "---Instructions---\n"
        "- Return only the final short answer.\n"
        "- Do not include explanations, citations, markdown, or references.\n"
        "- Use the Context to infer the answer when multiple facts must be combined.\n"
        "- If the question is yes/no, return exactly \"yes\" or \"no\".\n"
        "- For questions beginning with \"Were\", \"Was\", \"Are\", \"Is\", \"Do\", \"Does\", or \"Did\", return only \"yes\" or \"no\".\n"
        "- If the answer is a person, place, organization, date, number, or title, return the shortest answer string that is supported by the Context.\n"
        "- Do not answer with a full sentence unless the answer itself is a sentence.\n"
        "- If several candidates appear in the Context, choose the one that best answers the Question.\n\n"
        "Context:\n{context_str}\n\n"
        "Question: {query_str}\n"
        "Answer: "
    )

    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT,
        text_qa_template=qa_prompt
    )

    # ==========================================
    # 🚨 挂载创新的自适应大模型图过滤器
    # ==========================================
    # 实例化大模型边权打分器
    llm_edge_scorer = LLMSemanticEdgeScorerv2(llm=Settings.llm)

    # 实例化自适应 PPR 剪枝器
    kg_adaptive_ppr = AdaptiveLLMSemanticPPRGraphFilterv2(
        doc2kg=doc2kg,
        ents=ents_set,
        chunks_index=chunks_index,
        llm_scorer=llm_edge_scorer,  # 传入打分器
        reranker=bge_reranker,
        top_k=args.seed_top_k,
        final_ppr_mass=args.final_ppr_mass,
        max_final_nodes=args.max_final_nodes,
        max_context_tokens=args.max_context_tokens,
        max_hops=3  # 增量扩展上限设为3跳
    )

    engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[kg_adaptive_ppr],  # 只有这一个集成处理器
    )

    questions, answers_list = load_qa_pairs(args.questions_path, args.answers_path)
    if args.test_limit and args.test_limit > 0:
        questions = questions[:args.test_limit]
    answers_dict = {}
    for ans in answers_list:
        if isinstance(ans, dict) and "query" in ans:
            clean_k = re.sub(r'^-\s*Question\s*\d+:\s*', '', ans["query"]).strip()
            answers_dict[clean_k] = ans

    # [小范围测试截断 - 如果你想全量跑，请注释这两行]
    # TEST_LIMIT = 10
    # questions = questions[:TEST_LIMIT]

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                outputs = json.load(f)
        except:
            pass

    # start_idx = len(outputs)
    # for i in tqdm(range(start_idx, len(questions)), desc="Adaptive LLM PPR Running", initial=start_idx,
    #               total=len(questions)):
    #     q = questions[i]
    #     clean_q = re.sub(r'^-\s*Question\s*\d+:\s*', '', q).strip()
    #
    #     try:
    #         token_counter.reset_counts()
    #         resp = engine.query(q)
    #         ans = normalize_result_text(getattr(resp, "response", ""))
    #
    #         # 获取后台沉默计算的推理路径
    #         reasoning_path = getattr(kg_adaptive_ppr, "latest_reasoning_path", "")
    #
    #         # 由于大模型判别边权也会消耗LLM Token，这里统计的总Token将包含判别器消耗和最后QA消耗之和
    #         token_usage = {
    #             "prompt_tokens": token_counter.prompt_llm_token_count,
    #             "completion_tokens": token_counter.completion_llm_token_count,
    #             "total_tokens": token_counter.total_llm_token_count
    #         }
    #
    #     except Exception as e:
    #         traceback.print_exc()
    #         ans = ""
    #
    #     outputs.append({
    #         "query": q,
    #         "prediction": ans,
    #         "ground_truth": answers_dict.get(clean_q),
    #         "reasoning_path": reasoning_path,
    #         "token_usage": token_usage
    #     })
    #
    #     with open(out_path, "w", encoding="utf-8") as f:
    #         json.dump(outputs, f, ensure_ascii=False, indent=2)
    start_idx = len(outputs)
    for i in tqdm(range(start_idx, len(questions)), desc="Adaptive LLM PPR Running", initial=start_idx,
                  total=len(questions)):
        q = questions[i]
        clean_q = re.sub(r'^-\s*Question\s*\d+:\s*', '', q).strip()

        # 🚨 修复：在 try 之前就初始化默认值，防止异常抛出时变量未定义
        reasoning_path = ""
        token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "final_context_tokens": 0
        }
        final_context_nodes = 0
        ppr_debug = {}

        try:
            token_counter.reset_counts()
            resp = engine.query(q)
            ans = normalize_result_text(getattr(resp, "response", ""))
            final_context_tokens = count_final_context_tokens(resp, token_encoding)
            final_context_nodes = count_final_context_nodes(resp)

            # 获取后台沉默计算的推理路径
            reasoning_path = getattr(kg_adaptive_ppr, "latest_reasoning_path", "")
            ppr_debug = getattr(kg_adaptive_ppr, "latest_debug", {}) or {}

            # 获取 Token
            token_usage = {
                "prompt_tokens": token_counter.prompt_llm_token_count,
                "completion_tokens": token_counter.completion_llm_token_count,
                "total_tokens": token_counter.total_llm_token_count,
                "final_context_tokens": final_context_tokens
            }

        except Exception as e:
            # 异常发生时，保留错误信息，并确保循环继续
            print("\n" + "=" * 80)
            print(f"[ERROR] Exception occurred on question: {clean_q}")
            traceback.print_exc()
            print("=" * 80 + "\n")
            ans = ""

        outputs.append({
            "query": q,
            "prediction": ans,
            "ground_truth": answers_dict.get(clean_q),
            "reasoning_path": reasoning_path,
            "final_context_nodes": final_context_nodes,
            "ppr_debug": ppr_debug,
            "token_usage": token_usage
        })

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)

    print(f"\n[Done] Adaptive predictions saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
