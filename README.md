# AdaPPR-RAG

**Query-adaptive, vector-seeded knowledge-graph retrieval for multi-hop question answering**

[English](#english) | [简体中文](#简体中文)

## English

### Overview

AdaPPR-RAG is a query-adaptive, graph-augmented retrieval framework for multi-hop question answering. It begins with vector-retrieved passages, expands a knowledge graph from the entities in those passages, and uses question-conditioned edge weighting and Personalized PageRank (PPR) to identify useful evidence. An adaptive stopping strategy determines whether graph expansion should continue. The selected passages and graph facts are then reranked and provided to a language model for answer generation.

### Framework

![AdaPPR-RAG framework](docs/model_framework.png)

### Workflow

1. **Seed entity generation.** Given a question, a vector index retrieves the top-10 text chunks. Entities associated with the retrieved chunks are extracted as the initial graph seeds.
2. **Hop-wise knowledge graph expansion.** The graph is expanded incrementally from the seed entities, up to three hops. Newly discovered edges receive question-relevance scores from a lightweight language model. The scores are used as edge weights for graph propagation.
3. **Personalized PPR and guarded stopping.** A cross-encoder reranker scores seed entities against the question and uses the top-ranked seeds to form the personalized restart distribution. Weighted PPR combines edge relevance with graph connectivity. From the second hop onward, expansion stops when the PPR distribution is sufficiently concentrated, restart entities are represented among highly ranked entities, and either query-relevant entities are covered or the top-entity ranking is stable. Otherwise, the next hop is explored, subject to the three-hop maximum.
4. **Evidence union and reranking.** Candidate evidence is assembled from the original seed chunks, chunks associated with retained entities, and related knowledge-graph facts. Retained entities are drawn from the restart set, the entities covering the configured PPR mass, and the most query-relevant entities. A cross-encoder reranks candidate evidence to form the final context.
5. **Answer generation.** The final context and question are passed to the QA prompt and response synthesizer, which generate a concise answer. The implementation also records graph paths and PPR diagnostics for inspection.

**Implementation note.** Edge relevance scores weight graph propagation; low-scoring edges are not explicitly deleted at each expansion hop. The final context selector applies a node limit and a lightweight whitespace-based token-budget estimate.

### Code structure

```text
.
├── kg_post_processor_main.py       # Edge scoring, graph expansion, weighted PPR,
│                                   # adaptive stopping, and evidence selection
├── kg_rag_sppr_new_hotpot.py       # HotpotQA pipeline entry point
├── kg_rag_sppr_new_multihop.py     # Multihop pipeline entry point
├── kg_response_synthesizer.py      # Context synthesis and answer generation
├── prompt_helper.py                # Prompt-aware context repacking helper
└── docs/
    └── model_framework.png         # Model framework figure
```

### Code availability

This repository provides the first release of the model code accompanying the paper. Upon acceptance of the paper, we will release the complete codebase together with the full parameter configurations.

---

## 简体中文

### 模型简介

AdaPPR-RAG 是一种面向多跳问答的问题自适应知识图谱增强检索框架。模型从向量检索得到的文本片段出发，以片段中的实体为种子扩展知识图谱，并结合问题条件化边权与 Personalized PageRank（PPR）识别有用证据。自适应停止策略决定是否继续扩展。随后，系统对选出的文本片段和图谱事实进行重排，并将其交给大语言模型生成答案。

### 模型框架图

![AdaPPR-RAG 模型框架图](docs/model_framework.png)

### 模型流程

1. **种子实体生成。** 输入问题后，向量索引检索 top-10 文本块，并从检索块中提取关联实体作为图检索的初始种子。
2. **逐跳知识图谱扩展。** 从种子实体出发逐步扩展知识图谱，最多三跳。轻量级语言模型对新发现的边进行问题相关性评分，并将分数用于图传播的边权。
3. **个性化 PPR 与受约束的自适应停止。** 交叉编码器重排器计算种子实体与问题的相关性，并使用排名靠前的种子构造个性化重启分布。加权 PPR 联合考虑边相关性和图结构连通性。从第二跳开始，当 PPR 分布足够集中、重启实体出现在高排名实体中，且查询相关实体得到覆盖或头部实体排名趋于稳定时，停止扩展；否则继续下一跳，最多扩展至三跳。
4. **证据合并与重排。** 候选证据由初始种子文本块、保留实体关联的文本块及相关知识图谱事实组成。保留实体来自重启实体集合、覆盖指定 PPR 权重的实体，以及问题相关性排名靠前的实体。随后使用交叉编码器对候选证据重排，构建最终上下文。
5. **答案生成。** 将最终上下文与问题输入 QA 提示词和响应合成器，生成简洁答案。实现中还会记录图路径和 PPR 诊断信息，供检查分析。

**实现说明：** 边相关性分数用于图传播加权；每轮扩展时不会显式删除低分边。最终上下文选择器设有节点数量上限，并使用基于空白字符分词的轻量预算估计。

### 代码结构

```text
.
├── kg_post_processor_main.py       # 边评分、图扩展、加权 PPR、
│                                   # 自适应停止与证据选择
├── kg_rag_sppr_new_hotpot.py       # HotpotQA 流程入口
├── kg_rag_sppr_new_multihop.py     # Multihop 流程入口
├── kg_response_synthesizer.py      # 上下文合成与答案生成
├── prompt_helper.py                # 面向提示词限制的上下文重组工具
└── docs/
    └── model_framework.png         # 模型框架图
```

### 代码开放计划

本仓库提供与论文配套的第一版模型代码。论文录用后，我们将完整公开代码库及全部参数配置。
