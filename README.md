# RAG Framework

本地可跑的检索增强生成框架。核心只依赖 **Python 标准库 + NumPy**，不需要 LangChain、不需要外网、不需要 GPU 也能完整跑通；同时把 embedding 模型、LLM、联网检索都做成了可插拔适配器，一旦环境里有真模型就自动升级。

## 快速开始

```powershell
python -m rag.cli --config config.yaml doctor          # 看当前能用到什么（含实际生效的融合权重）
python -m rag.cli --config config.yaml demo            # 索引内置语料并回答示例问题
python -m rag.cli --config config.yaml index examples/corpus
python -m rag.cli --config config.yaml query "RRF 融合相比加权分数融合有什么优缺点" --trace
python -m rag.cli --config config.yaml search "late interaction maxsim" --k 6
python -m rag.cli --config config.yaml eval examples/eval.jsonl --k 5 --generation
python -m unittest discover -s tests -t .
```

**一个 `db_path` = 一个知识库。** 默认 `data/rag.db` 是内置示例语料的库。要做自己的知识库，复制一份配置改 `db_path`，否则不同语料会混在同一个库里互相干扰：

```powershell
# config.work.yaml
# extends: config.yaml
# db_path: data/work.db
python -m rag.cli --config config.work.yaml index D:/my-notes
python -m rag.cli --config config.work.yaml query "灰度发布要观察多久" --k 3
```

`query` / `search` / `eval` 只在库为空时才会自动索引内置示例语料，不会污染你已经建好的库。

## 架构

```
ingest -> chunk -> embed -> index(BM25 / dense / late-interaction)
       -> hybrid fuse -> rerank -> context assemble -> grounded answer -> self-check
                                    ^                                      |
                                    +---- agentic grading + query rewrite -+
```

| 模块 | 作用 |
| --- | --- |
| `rag/text.py` | 中英混合分词（CJK 字符二元组）、句子切分、token 计数 |
| `rag/chunking.py` | structural / semantic / hierarchical(小到大) / sentence-window |
| `rag/embed/` | `hashed`(离线兜底) / `lsa`(TF-IDF+随机化 SVD) / `ollama` / `openai_compat` / 缓存 |
| `rag/index/` | BM25+ 倒排、稠密 ANN、ColBERT 风格 late interaction、SQLite 持久化 |
| `rag/retrieve/` | RRF 与加权融合、元数据过滤、联网检索适配器 |
| `rag/rerank.py` | late interaction / 启发式 / LLM listwise(RankGPT) / 链式组合 |
| `rag/transform.py` | HyDE / multi-query / step-back / 问题分解 |
| `rag/generate.py` | 去重、父块扩展、U 形上下文排序、带引用生成、抽取式兜底 |
| `rag/agentic.py` | CRAG 式相关性分级、自适应改写、groundedness 自检重试 |
| `rag/eval.py` | Recall@k / Precision@k / MRR / nDCG@k / faithfulness / relevancy |

## 内置的当代 RAG 技术

- **混合检索**：BM25+ 与稠密通道并行召回，RRF 或加权归一化融合，保留通道贡献便于调试。
- **按嵌入模型自动选融合档**：词面型嵌入偏 BM25，真嵌入模型偏稠密，权重不再一刀切（`WEIGHT_PROFILES`）。
- **CJK 分词**：中文按字符二元组建索引（Lucene CJKBigramFilter 思路），比单字索引稳。
- **层级切块（small-to-big）**：只索引子块，命中后上溯父块作为生成上下文。
- **语义切块**：相邻句子嵌入相似度骤降处断开。
- **Late interaction 重排**：token 级向量 MaxSim（每个查询 token 取最佳匹配 token 再平均），无需模型推理即可离线构建。
- **LLM listwise 重排**：RankGPT 式编号候选 + 排列输出。
- **查询改写**：HyDE、multi-query、step-back、问题分解。
- **Agentic / Corrective RAG**：对候选做 relevant/ambiguous/irrelevant 分级，质量不足则改写查询或转联网检索。
- **Groundedness 自检**：把答案拆成断言，逐条查证据支持度，低于阈值重试或降级。
- **Lost in the middle 缓解**：U 形上下文排列，最强证据放首尾。
- **证据充分性判定**：相关不等于能回答。按疑问词推出"所需事实的形状"（数字/日期/版本/因果/步骤/比较），再检查候选里是否同时出现被问实体与承载该事实的片段（`rag/evidence.py`）。
- **成本记账与预算止损**：一次回答的轮数、分级候选数、送达/丢弃的上下文、LLM 调用与 token 全部进 ledger，超限即停（`rag/usage.py`）。
- **"检索到但没送达"诊断**：上下文预算吃掉了多少候选，直接打在 trace 里，不再静默丢弃（`delivery_report`）。
- **语料退化评测**：错字/截断/重复/句序打乱/无关文档/互相矛盾的更正文档，按等级注入后重跑评测，输出退化曲线（`rag/stress.py`）。
- **增量索引**：文档指纹 + 按 (模型名, 维度, 文本哈希) 的嵌入缓存（`data/embeddings.db`，跨索引共享；`lsa` 因为要拟合语料所以不进缓存）。

## 配置

见 `config.yaml`。关键项：

```yaml
embedder: auto        # auto | hashed | lsa | ollama:bge-m3 | openai:text-embedding-3-small
llm: auto             # auto | none | ollama:qwen3.8:27b | openai:gpt-4o-mini
reranker: heuristic   # none | heuristic | late | llm | 链式如 "heuristic,late"
transform: none       # none | hyde | multi-query | step-back | decompose
retrieval:
  fusion: weighted    # rrf | weighted
  channels: [bm25, dense]
  weights:            # 留空 = 按嵌入模型自动选档；写死则完全覆盖
  max_per_document: 0 # 0=不限；案例库这类「一篇一个案例」的语料设 1，否则 top-k 全是同一篇
```

### 复用已加载的本地服务（推荐，不会加载第二个模型）

`config.strata.yaml` 演示了如何指向已经在显存里的服务：

```yaml
llm: auto
llm_base_url: http://127.0.0.1:8080/v1
llm_model: qwen3.8-flash-next-iq3_xxs
llm_wire_api: responses     # Strata/vLLM 用 responses；LM Studio/OpenAI 用 chat
```

```powershell
python -m rag.cli --config config.strata.yaml query "RRF 融合相比加权分数融合有什么优缺点" --trace
python -m rag.cli --config config.strata.fast.yaml query "..."   # 关掉 agentic 循环的快档
```

这条路径只发 HTTP 请求，不会再加载一份权重，所以显存已被占满时依然可用。

### 接真模型

```powershell
ollama pull bge-m3                                   # 之后 embedder: auto 会自动选中它
python -m rag.cli --config config.bge-m3.yaml doctor # 真嵌入档，融合权重已按 bge-m3 校准
# 或走 OpenAI 兼容端点（vLLM / LM Studio / Strata / OpenAI）
$env:RAG_EMBED_BASE_URL = "http://127.0.0.1:8080/v1"
$env:RAG_EMBED_MODEL   = "text-embedding-3-small"
$env:RAG_LLM_BASE_URL  = "http://127.0.0.1:8080/v1"
$env:RAG_LLM_MODEL     = "qwen3.8-flash-next-iq3_xxs"
$env:RAG_LLM_WIRE_API  = "responses"
```

## 联网检索（已实测）

`rag/retrieve/web.py` 支持四个后端，按可用性自动选择：`TAVILY_API_KEY` → Tavily，`BRAVE_API_KEY` → Brave，然后 Bing，最后 DuckDuckGo。另有一个 **arXiv 后端**，是拉取最新方法论文最可靠的路径。抓回的网页变成普通 `Document`，走与本地文件完全相同的切块、索引、重排与引用路径。

```powershell
python -m rag.cli --config config.yaml webtest "GraphRAG multi-hop retrieval" --backend bing --k 6
python -m rag.cli --config config.yaml webtest 'abs:"retrieval-augmented generation" AND cat:cs.IR' --backend arxiv --k 10
python examples/live_web_search.py
```

国内网络实测：DuckDuckGo / Google / Brave 不可达，Bing 可达但对英文技术查询会退化成词典结果，arXiv API 稳定。**所以查最新技术优先用 `--backend arxiv`。**

arXiv 查询语法直接透传：`abs:"..."`、`ti:"..."`、`cat:cs.IR`、`AND` / `OR`。结果按提交日期倒序。

PowerShell 5.1 会把带引号的参数拆开，用停止解析符 `--%` 原样传参：

```powershell
python -m rag.cli --config config.yaml webtest --% "abs:\"retrieval-augmented generation\" AND cat:cs.IR" --backend arxiv --k 5
```

### 实测：联网增强对检索指标的影响

内置语料 7 篇为基线，联网抓 32 篇（26 篇可用）注入同一个索引：

| 指标 | 仅本地语料 | 本地 + 联网 | 变化 |
| --- | --- | --- | --- |
| Recall@5 | 0.934 | **0.978** | +0.044 |
| Precision@5 | **0.720** | 0.600 | -0.120 |
| MRR@5 | 0.933 | **1.000** | +0.067 |
| nDCG@5 | 0.954 | **0.982** | +0.028 |

索引规模 7 → 37 篇文档，117 → 641 chunk，73 → 470 向量。

Precision@5 下降是预期的：候选池变大后，本地问题的 top-5 里会混入网页结果。Recall 和 MRR 上升说明真正相关的证据排名更靠前了。

### 实测：本地语料答不了的问题

联网注入后，三个问题在 top-5 里分别有 5/5、4/5、5/5 个网页来源，答案全部带引用。例如"微软 GraphRAG 如何构图"直接命中了抓回的中文技术长文，答出了 `microsoft/graphrag`、论文 `From Local to Global: A Graph RAG Approach to Query-Focused Summarization`、以及"后续版本把 token 成本降低了约 77%"。

按来源过滤可用（切块继承文档元数据）：

```python
pipeline.retrieve(q, filters={"metadata": {"origin": "web"}})
pipeline.retrieve(q, filters={"exclude_doc_ids": ["web:..."]})
```

### 抓取的现实问题

- 知乎一类页面会带登录、注册、邀请码等导航噪声。`extract_main_region` 优先取 `<article>` / `<main>` 区域，再用 `_BOILERPLATE_RE` 剥掉这类行；实测知乎长文 14617 字符、噪声行 0。
- arXiv 的 `http://` 会返回 502，`https://` 正常，已在解析时规范化。
- arXiv 的 abs 页面没有 `<article>` / `<main>` 区域，整页抽取会把 `Computer Science > ...`、`[Submitted on ...]`、`View a PDF` 这类导航灌进索引，把 chunk 向量稀释掉。`extract_arxiv_abstract` 只取 `<blockquote class="abstract">`，实测单篇正文从整页降到 1332-1542 字符的纯摘要。
- 部分站点（Datawhale 文档站）正文靠 JS 渲染，抓到的正文只有 32 字符。这类页面会被 `search_documents` 退回标题+摘要。
- 搜索引擎摘要本身已经是人工提炼过的正文，质量常常高于抓回来的 HTML。`web.prefer_snippet: true`（默认）在摘要够长（`min_snippet_chars: 160`）时直接用摘要，不再抓整页；摘要太短才回落到抓页。

### 实测：中文查询命中英文网页（跨语言检索）

`bge-m3` 的嵌入本身是跨语言的。直接测余弦相似度：

| 中文查询 | 对照英文文本 | 相似度 |
| --- | --- | --- |
| 重排和后期交互对检索质量有什么影响 | 该论文英文摘要 | 0.6132 |
| graphrag 什么时候比向量检索更合适 | 该论文英文摘要 | 0.5461 |
| 向量数据库存储嵌入向量 | 对应英文句 | 0.5796 |
| 今天天气很好适合出去散步（无关对照） | 同上 | 0.3480 |

但端到端跑起来时，中文查询一度完全够不到网页文档。根因不在嵌入模型，而在上面那条 arXiv 导航噪声。修完之后（`python examples/cross_lingual_sweep.py`，本地 7 篇 + arXiv 4 篇，`reranker: none`）：

| 融合方式 | 本地 Recall@5 | 本地 nDCG@5 | 5 个中文查询的网页命中数（top-5） |
| --- | --- | --- | --- |
| weighted 1.2 : 0.8 | 0.723 | 0.826 | 0, 0, 0, **2**, 0 |
| weighted 1.0 : 1.0 | 0.789 | 0.871 | 0, 0, 0, **2**, 0 |
| weighted 0.5 : 1.5 | **0.818** | **0.930** | 0, 0, 1, **2**, 0 |
| dense only | 0.800 | 0.921 | 0, 0, 1, **2**, 0 |
| rrf 1 : 1 | 0.757 | 0.922 | 0, 0, 0, **2**, 0 |

同一脚本还对比了两种把 arXiv 结果变成 `Document` 的方式：抓整页（清掉噪声后 1115-1542 字符纯摘要）与直接用 Atom 摘要（1202-1657 字符）。两者跨语言命中基本一致，摘要优先少 4 次页面抓取，所以默认走摘要优先。

其中"校园 AI 导师在硬件和软件之间如何权衡"是纯中文、本地语料完全没有对应内容，答案只在英文论文摘要里：`web_best=2.0000`、`local_best=0.4220`，网页 chunk 直接进 top-5。跨语言通路确认打通。

另一条"多跳问答中小块和大块应该怎么安排才能省成本"网页命中为 0 —— 因为本地语料里已经有一篇讲层级切块的文档，`local_best=1.88` 压过了网页。这是正确行为，不是失败。

### 联网抓到的近期工作（2026-09 至 2026-10）

这几篇与框架的设计选择直接相关，也是这套框架内置特性的当前实证：

- **A Matryoshka Hierarchical RAG for Efficient Multi-Hop QA**（2026-10-01）— 层级检索，对应 `strategy: hierarchical`。
- **Re-ranking and Late Interaction Drive Retrieval Quality: A Controlled Comparison of RAG Strategies for Scientific QA**（2026-09-29）— 在 463,971 篇 arXiv 论文上做对照实验，结论是重排与 late interaction 主导检索质量。对应 `rag/index/late.py` 与 `rag/rerank.py`。
- **ARCagent: An Adaptive Retrieval Calibration Agent**（2026-09-28）— 自适应检索校准，对应 `rag/agentic.py`。
- **Towards Semi-Automatically Comparing Keyword-Based and Semantic Retrieval**（2026-09-29）— 词面 vs 语义检索对照，对应混合检索通道。

本轮新增能力直接来源（2026-09 至 2026-10）：

- **Relevance Is Not Sufficient Evidence: Detecting Evidence Gaps Before Generation**（2609.37469）— 段落"提到了正确实体但不含回答问题所需的事实"；即使要求弃答，生成器仍对 40.0–99.3% 的证据不足问题作答。→ `rag/evidence.py`。
- **From Topical Relevance to Answerability: Entailment Distillation for Conversational Retrieval**（2609.03482）— 系统性 answerability gap。→ 分级器（relevant/ambiguous/irrelevant）与充分性判定分离。
- **BELIEFRAG: Making Adaptive RAG State-Aware under Evolving Evidence**（2609.39139）— 多步检索需要维护"当前证据支持什么、还缺什么"。→ trace 里的 `evidence` 阶段与 `missing_entities`。
- **RAGStress: A controlled benchmark for evaluating RAG under knowledge-base corruption**（2610.04691）— RAG 通常在"干净 KB"假设下评测。→ `rag/stress.py`。
- **Agentic RAG Evaluation: Budget Allocation Across Questions, Trajectories, and Reads**（2610.05034）— 要测 reading efficiency 与 cost boundary。→ `rag/usage.py`。
- **Judged Useless, Queried Anyway**（2610.06191）— agent 很少把"这结果没用"转成停止决策。→ `evidence_stop_on_no_entity`。
- **Programmatic Search Agents**（2610.06689）与 **Found but Not Read**（2610.02880）— 支持性段落被检索到却从未送达。→ `delivery_report`。
- **Mapping the RAG Landscape: A Four-Axis Taxonomy**（2610.01936）— 效率/防御/交互性/推理四轴分类，可用来给上面的特性归类。
- **TeleTune: Evolving Agent Skills From Offline Telemetry**（2610.05437）— 从离线遥测演化 agent skill，与本项目"把会话踩坑沉淀成 casebook + skill"的做法同构。
## 实测：证据充分性、成本与语料退化（2026-10）

三项都是纯 CPU 实现，不需要额外加载模型。

### 证据不足判定（`agentic.evidence`）

对 `examples/eval.jsonl` 的 10 个语料内问题，判定"证据不足"的数量随实现收敛：

| 实现版本 | 误拒（语料内被判不足） |
|---|---|
| 中文整段当硬实体 | 4/10 |
| 中文短语改软匹配 + 硬实体只认标识符 | 1/10 |
| 无硬实体时退回词面重合锚定 | **0/10** |

对 4 个语料外问题（Kubernetes HPA、PostgreSQL 逻辑复制、surface code 阈值、布洛芬剂量），默认配置**一个都不拒**，全部照常作答 —— 这正是 arXiv:2609.37469 测到的行为（12 个生成器对 40.0–99.3% 的证据不足问题仍然作答）。

打开闸门后在**内置玩具语料**上看着很好，但在**真实案例语料**上完全不是那回事。
下面是同一套代码在两个语料上的四档对照（30 个语料内问句 + 8 个语料外问句；
内置语料是 10 + 4）：

| 语料 | 档位 | 语料内被误拒 | 语料外被抓到 | 轮数 | 分级候选 |
|---|---|---|---|---|---|
| casebook（19 案例） | 全关（默认） | 0/30 | 0/8 | 82 | 410 |
| casebook | **词面止损 only** | **0/30** | **5/8** | 73 | 365 |
| casebook | 止损 + 闸门 | 5/30 | 7/8 | 73 | 365 |
| casebook | 闸门 only | 5/30 | 7/8 | 82 | 410 |
| 内置语料（7 文档） | 全关 | 0/10 | 0/4 | 24 | 144 |
| 内置语料 | 词面止损 only | 0/10 | 3/4 | 18 | 108 |
| 内置语料 | 止损 + 闸门 | 0/10 | 4/4 | 18 | 108 |
| 内置语料 | 闸门 only | 0/10 | 3/4 | 24 | 144 |

三条结论：

1. **`stop_when_no_lexical_hit` 是干净的**：两个语料上误拒都是 0，语料外抓到 5/8 与 3/4，
   成本降 11%（真实语料）到 25%（内置语料）。判据是"BM25 通道对任何候选都没有贡献" ——
   语料里连词面都碰不上的时候，重试同一个语料没有意义。
2. **`evidence_gate` 在真实语料上不划算**：多抓 2 个语料外问题，代价是把 5/30 个语料内可答
   问题也拒了。内置语料上它 0 误伤，是因为那 10 个问句本来就是照文档标题写的。
3. **support 分数在真实语料上没有区分力**：语料内被标记的问题 support 落在 0.503-0.685，
   语料外落在 0.550-0.700，完全重叠。之前"0 误拒"是内置语料的运气。

`evidence_stop_on_no_entity`（按硬实体覆盖率止损）在真实语料上**一次都没触发**：
中文问句没有硬实体标识符，`entity_coverage` 退回 1.0。它精确但召回极低，只在含英文标识符的
提问上起作用。

坑：`stop_when_no_lexical_hit` 依赖 BM25 有命中。纯英文语料 + 中文提问的跨语言场景里
BM25 合法地为 0，这个止损会误停，那种语料不要开。推荐档位见 `config.casebook.evidence.yaml`。

### 语料退化曲线（`rag.cli stress`）

`python -m rag.cli --config config.yaml stress examples/eval.jsonl --k 5`

| 退化等级 | 文档数 | recall@5 | Δ | ndcg@5 |
|---|---|---|---|---|
| 0（干净） | 7 | 0.8714 | — | 0.9723 |
| 1 | 10 | 0.8673 | -0.0041 | 0.9854 |
| 2 | 13 | 0.8294 | -0.0420 | 1.0000 |
| 3 | 16 | 0.8161 | -0.0553 | 0.9163 |

结论：这套混合检索对轻度语料污染相当鲁棒（等级 1 几乎无损），到等级 3（同时注入错字、截断、重复、句序打乱、无关文档、矛盾更正文档）才掉 5.5 个点。`mrr@5` 到等级 3 才第一次下滑，说明排序质量比召回更抗污染。

### 成本与送达

`query --trace` 现在会打印：

```
evidence: type=count sufficient=False support=0.375 entities_missing=['bge-m3', '向量维度']
delivery: retrieved=8 delivered=6 dropped=2 context_tokens=1415 dropped_tokens=1076
usage: rounds=3 candidates_retrieved=24 graded_candidates=18 delivered_units=5 dropped_units=1
```

`dropped_tokens=1076` 就是 arXiv:2610.06689 / 2610.02880 说的"检索到了但没送到模型眼前"，以前完全不可见。

## 实测结果

内置语料 7 篇（中英混合）、10 条问答、`hashed` 离线嵌入、`llm: none`（纯 CPU）。

| 配置 | Recall@5 | Precision@5 | MRR@5 | nDCG@5 |
| --- | --- | --- | --- | --- |
| BM25 only | 0.819 | 0.700 | 0.933 | 0.879 |
| dense only (hashed) | 0.767 | 0.560 | 0.781 | 0.765 |
| hybrid RRF 1:1 | 0.788 | 0.700 | 0.933 | 0.859 |
| hybrid weighted 1.2:0.8 | 0.826 | 0.660 | 0.933 | 0.890 |
| hybrid weighted + heuristic rerank | **0.934** | **0.720** | 0.933 | **0.954** |
| LSA embedder + heuristic rerank | 0.671 | **0.920** | 0.933 | 0.918 |

### 真嵌入模型（bge-m3，1.2 GB，CPU 推理）

`ollama pull bge-m3` 之后 `embedder: auto` 会自动选中它（1024 维）。同一份语料与评测集，权重 1.2:0.8：

| 配置 | Recall@5 | Precision@5 | MRR@5 | nDCG@5 |
| --- | --- | --- | --- | --- |
| bge-m3 + 无重排 | 0.770 | 0.780 | 0.933 | 0.885 |
| bge-m3 + heuristic | 0.810 | 0.820 | 0.933 | 0.917 |
| bge-m3 + heuristic（权重 0.5:1.5） | **0.871** | **0.880** | **1.000** | **0.972** |

### 融合权重必须随嵌入模型重新校准

`python examples/embedder_comparison.py` 里的权重扫描（本地语料，bm25+dense，格式 `recall / ndcg`）：

| 嵌入 + 重排 | 1.2:0.8 | 1.0:1.0 | 0.8:1.2 | 0.5:1.5 | 实测最优 |
| --- | --- | --- | --- | --- | --- |
| hashed + 无重排 | 0.826/0.890 | **0.884/0.908** | 0.839/0.870 | 0.797/0.795 | 1.0:1.0 |
| hashed + heuristic | **0.934/0.954** | 0.884/0.913 | 0.864/0.900 | 0.856/0.887 | 1.2:0.8 |
| bge-m3 + 无重排 | 0.769/0.885 | 0.781/0.890 | 0.828/0.923 | **0.855/0.958** | 0.5:1.5 |
| bge-m3 + heuristic | 0.809/0.917 | 0.838/0.934 | 0.865/0.955 | **0.871/0.972** | 0.5:1.5 |

`hashed` 的最优是 `bm25 1.2 : dense 0.8`，`bge-m3` 的最优是 `bm25 0.5 : dense 1.5` —— 把前者套到后者，nDCG 从 0.972 掉到 0.917。

所以 `retrieval.weights` 默认留空，由 pipeline 按实际构建出来的嵌入模型自动选档（`rag/retrieve/hybrid.py` 的 `WEIGHT_PROFILES`）：词面型嵌入（`hashed` / `lsa`）走 `lexical` 档，真嵌入模型（`ollama:*` / OpenAI 兼容）走 `semantic` 档。写死权重则完全覆盖自动档，`doctor` 会打印实际生效的权重。

`bge-m3` 在这个词面重合度很高的评测集上 Recall 仍低于 `hashed + heuristic`（0.871 vs 0.934），但 Precision 更高（0.88 vs 0.72）。这份评测集的问句和文档用词高度重叠，天然偏袒 BM25；跨语言那一节才是真嵌入模型真正拉开差距的地方。

### 生成侧

| 生成方式 | faithfulness | citation_coverage | answer_relevancy | answerable_rate |
| --- | --- | --- | --- | --- |
| 抽取式兜底（`llm: none`） | 1.000 | 0.343 | 0.132 | 1.000 |
| Strata qwen3.8-flash（agentic） | 0.634 | 0.615 | 0.225 | 1.000 |
| Strata qwen3.8-flash（单次检索） | 0.691 | 0.431 | 0.206 | 1.000 |

抽取式答案直接复制证据句，所以忠实度必然为 1；换成真 LLM 后忠实度掉到 0.63-0.69，说明这套离线代理指标确实能区分"照抄证据"和"自己发挥"。agentic 循环把引用覆盖率从 0.43 提到 0.62。

`answer_relevancy` 偏低（0.2 左右）是 `hashed` 嵌入的局限：它对"长答案 vs 短问题"的余弦区分度很弱。换真嵌入模型后这个指标才有意义。

### 延迟与成本

在只有 Strata 一个本地推理服务、且该服务同时在给别的会话供推理的环境下，10 条问答的生成评测跑了约 20 分钟。瓶颈是模型服务并发，不是检索——检索部分（10 条 × 全量索引）不到 1 秒。

三条实用建议：

- 检索质量用 `eval`（纯 CPU，秒级）反复迭代，生成质量少跑、小样本跑。
- `generation.max_output_tokens` 调小（默认 1400，本地 27B 建议 700）比调大上下文更划算。
- agentic 循环的分级已改成**一次调用批量分级**；早期版本对每个候选单独分级，10 条问答会打出上百次模型调用。

复现：`python examples/ablation.py`（更多变体）、`python examples/embedder_comparison.py`（嵌入模型与融合权重扫描）、`python examples/cross_lingual_sweep.py`（跨语言 + 网页正文抽取方式）与 `python -m rag.cli --config config.yaml eval examples/eval.jsonl --k 5 --generation`。

**注意**：这份评测集词面与文档高度重合，所以词面类方法（BM25、启发式重排）在这里占优。换成真实语义问句 + 真嵌入模型后，稠密通道与 late interaction 的收益会显著上升——这也是为什么默认 `reranker` 选了实测最优的 `heuristic`，而不是看起来更"高级"的那个。

## 已知边界

- `hashed` / `lsa` 是**离线兜底**，语义能力远不如真嵌入模型。它们的作用是让框架在任何环境下都能跑通并被评测，不是最终形态。
- `lsa` 需要语料拟合，语料变化后要重新 fit（pipeline 已自动处理）。
- 无 LLM 时生成走抽取式兜底（MMR 选句 + 引用），不会编造内容。
- 精确稠密检索在百万级 chunk 以上需要换 ANN 索引。
- 证据充分性判定是**词面启发式**，不是蕴含模型。它对"被问实体是否出现 + 候选里有没有所需事实的形状"敏感，对需要跨段落推理才能回答的问题会误判。因此 `evidence_gate` 与 `evidence_stop_on_no_entity` 默认关闭；实测在语料内问题上有过 4/10 的误拒率，收敛到 0/10 之后才建议按语料情况打开。
- 退化评测里的 `truncate` 会真的删掉证据，recall 下降是预期行为而不是 bug —— 它测的就是"语料缺了一段"时的表现。
- `late` 重排依赖 token 级向量。`hashed` / `lsa` 能直接给出 token 向量；只暴露整句嵌入的模型（`ollama:*`）会退回 hashed token 向量兜底，所以 `bge-m3 + late` 与 `bge-m3 + 无重排` 指标完全相同（0.770/0.780/0.933/0.885）。要拿到真正的 ColBERT 效果需要接一个多向量模型，接口已经留好了（`Embedder.embed_tokens`）。