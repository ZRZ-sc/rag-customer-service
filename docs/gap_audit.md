# 需求设计文档 → 实现差距审计（项目1：企业级智能客服 RAG 系统）

> 审计基准：`D:\AIagent\项目一RAG智能客服.txt`（需求设计文档·修订版）
> 参考：`D:\AIagent\电商AI应用开发体系_可落地项目方案.txt`（项目 1 的更大范围描述，见文末 §4）
> 审计时间：2026-09-30 ｜ 结论：**需求文档内的功能项全部落地；2 项非功能指标受本机硬件限制未达标；1 项监控能力（LangSmith）因网络不可达未实测**

图例：✅ 已完成并验证 ｜ 🟡 完成但有偏差/替代方案 ｜ 🔴 未完成 ｜ ⚠️ 未达标（环境限制）

## 1. 功能需求（2.1）

| 编号 | 需求 | 状态 | 实现与证据 |
|---|---|---|---|
| F1 | API 或 Web 界面输入自然语言问题 | ✅ | `POST /api/v1/chat`；`app/static/index.html`（http://localhost:8000/），容器模式经 Nginx http://localhost:8080 |
| F2 | 自动识别精确查询类 vs 知识类 | ✅ | `app/services/intent.py`：政策标记 → 正则（ORD/SF 单号）→ 关键词 + LLM few-shot → 默认知识类。N4 意图准确率 **100%（62/62）** |
| F3 | 精确查询优先查 PostgreSQL | ✅ | `app/services/precise_query.py`：归属校验（403）+ 30s 短缓存；`source=database`，confidence=1.0 |
| F4 | 知识类走 FAQ RAG 检索 | ✅ | `app/services/rag/`：BM25 ∥ Milvus 向量 → RRF → Cross-Encoder 重排 → DeepSeek 生成 |
| F5 | 多轮对话结合上下文 | ✅ | Redis `session:{id}`（10 轮 / 30 分钟）+ `query_rewrite.py` 指代消解 + 会话实体记忆（与用户绑定） |
| F6 | 无法回答时自动转人工（预留接口） | ✅ | `POST /api/v1/handoff`、`GET /api/v1/handoff/{id}`；低置信度自动建单；`HANDOFF_WEBHOOK_URL` 预留回调 |
| F7 | 管理后台查看问答日志与统计 | ✅ | `GET /api/v1/admin/{logs,handoffs,stats}` + `app/static/admin.html`（独立 `ADMIN_TOKEN` 鉴权） |

## 2. 非功能需求（2.2）

| 编号 | 目标 | 状态 | 实测与说明 |
|---|---|---|---|
| N1 | QPS > 200 | ✅ | 缓存画像 **350 RPS**、精确查询 312 RPS（`docs/loadtest_report.md`） |
| N2 | P99 < 500ms | ⚠️ | 实测 P99 1.5~3.5s。文档性能预算明确依赖 **vLLM + GPU 重排**，本机为 CPU-only（i7-12650H / 16GB 无 GPU）。分级数据：L1 缓存命中 2-7ms、L2/L3/L4 复用 550-890ms、冷启动 0.8-2.9s。GPU 化或流式返回可解 |
| N3 | 可用性 99.9% | 🟡 | 架构能力齐备：`/health` + `/ready`（含熔断状态）、`restart: unless-stopped` + 容器 healthcheck、依赖降级、LLM 熔断；**未做长时间稳定性/混沌演练**，故不给 99.9% 的实测结论 |
| N4 | 意图 >90%、答案 >85% | ✅ | 意图 **100%**、答案 **100%**、HTTP 状态符合率 **100%**（`docs/eval_report.md`） |
| N5 | ≥500 并发 | ✅ | 500 并发 → 331 RPS、**0 个 5xx**；P99 由 3.6s 劣化到 8.2s（排队） |
| N6 | 链路追踪 + 指标 + 日志 | 🟡 | Prometheus 20+ 指标 / Grafana 15 面板 / **8 条告警规则** / loguru JSON 日志（`request_id`+`trace_id` 贯穿）。**LangSmith 未实测**：代码已接入开关（`LANGSMITH_TRACING`），但网络不可达且无 Key，现以 `request_id` + Prometheus 面板替代 |
| N7 | 只查自己的订单、API 鉴权、防越权 | ✅ | Bearer 鉴权；查他人订单 403（`authz_*` 9 条用例全绿）；管理接口独立 token；日志脱敏且禁记明文 token |
| N8 | 依赖故障有降级、系统不整体不可用 | ✅ | LLM／Milvus／Redis／PostgreSQL 四条降级链 + 本轮新增 **LLM 熔断**（详见 §3） |

## 3. 缺陷修复：LLM 不可用时的降级（本轮补齐）

流程6 要求「LLM 服务超时/不可用 → 返回固定话术 + 可选转人工」，审计时发现**未实现**：
生成阶段的 `llm.ainvoke` 异常会直接冒泡成 **HTTP 500**，与文档要求不符。已修复并验证：

| 项 | 修复前 | 修复后 |
|---|---|---|
| 接口行为 | HTTP 500「服务器内部错误」 | **HTTP 200** + 「抱歉，系统繁忙，请稍后再试」+ `source=fallback` + **自动转人工工单号** |
| 缓存 | — | 失败结果**不写缓存**（避免污染 1h 答案缓存） |
| 首次失败耗时 | — | 61s（受 SDK 超时+重试影响，≈2×`LLM_TIMEOUT`） |
| 后续请求耗时 | 每次都 60s+ | **0.2~0.6s**（熔断短路，毫秒级） |

实现要点：
- `app/services/llm.py`：新增 `ainvoke(messages, stage)` 统一入口 + **连续失败熔断**（`LLM_CIRCUIT_THRESHOLD`=3，**超时/连接类异常一次即打开**；`LLM_CIRCUIT_COOLDOWN`=30s 后放行探测）
- `app/services/rag/pipeline.py`：生成失败 → 固定话术 + `llm_failed` 标记
- `app/api/routes/chat.py`：`llm_failed` → 走兜底话术分支（复用既有自动转人工逻辑）
- 观测：新增 `llm_failure_total{stage}`（含 `:circuit_open`）、`/ready` 的 `llm` 组件新增 `circuit_open` 状态、新增 2 条 Prometheus 告警
- 回归：单元测试 **95 个全通过**；N4 评测 79 条 / 91 轮次仍为 意图 100% / 答案 100%

## 4. 技术架构与流程（第三章）

| 文档条目 | 状态 | 说明 |
|---|---|---|
| 3.1 架构（Nginx→FastAPI→Redis/PG/Milvus + 可观测性） | ✅ | 与实现一致；Nginx 已容器化（`nginx/nginx.conf`） |
| 3.2 FastAPI / Milvus / PostgreSQL / Redis / bge-small-zh / 混合检索+RRF / asyncio+httpx / 批处理 | ✅ | 批处理：Embedding `batch_size=32`、Rerank 分批 32 |
| 3.2 LangChain + **LCEL** | 🟡 | 使用 `langchain-core` / `langchain-openai`，但编排为自研 async（需要插入多级缓存、快速路径、熔断与降级），**未用 LCEL 声明式管道** |
| 3.2 LLM：vLLM 部署 Llama 3 / OpenAI API | 🟡 | 采用 **DeepSeek API**（按你的指定）；compose 中无 vllm 服务 |
| 3.2 重排序 bge-reranker-v2-m3 | 🟡 | 本机 16GB 内存装不下 fp32 v2-m3（加载即崩），改 **bge-reranker-base + int8 量化（~700MB）**；`.env.local` 中一行即可换回 v2-m3 |
| 3.2 LangSmith | 🔴 | 见 N6（网络不可达，未实测） |
| 流程1 意图识别三层 + 高频缓存直返 | ✅ | 正则 → 关键词/LLM → 默认知识类；L1 早期探测命中 2-7ms |
| 流程2 RAG（HyDE + Multi-Query + 混合检索 + Rerank + 生成 + 缓存） | 🟡 | 全部实现；差异：**Multi-Query 与 HyDE 合并为一次 LLM 调用**（省 ~1s）；送入 Cross-Encoder 的候选上限默认 12（文档写 50）——CPU 重排开销所致，`RERANK_MAX_CANDIDATES` 可调；召回默认 50（`TOP_K_CANDIDATES`，本机下调为 20） |
| 流程2 性能预算 <500ms | ⚠️ | 同 N2 |
| 流程3 缓存（知识类 TTL 1h、精确查询 30-60s、预热、主动失效） | ✅ | 四级缓存 `rag:answer:`(1h) / `query_rewrite:` / `query_transform:` / `rerank:`(各 24h)；`ingest_kb` 主动清缓存；`scripts/warmup_cache.py` + 启动预热开关 |
| 流程3 缓存键 = 问题 + 会话上下文摘要 | 🟡 | 实现为「**改写后的问题**」（改写已吸收上下文与指代），效果等价且键更稳定 |
| 流程4 多轮（Redis、10 轮、30 分钟） | ✅ | `session_max_turns=10`、`session_ttl_seconds=1800` |
| 流程5 知识库更新（脚本/API、全量+增量、清缓存、记日志） | 🟡 | `scripts/ingest_kb.py`：`ON CONFLICT` 幂等更新 + Milvus 删旧插新 + 清三级缓存 + 日志；**未集成进管理后台**（文档原文即"可先做脚本，后续再集成"）；无长文档分块（知识库是结构化 FAQ，一条问答一条向量） |
| 流程6 降级策略（LLM/Milvus/PG/Redis） | ✅ | 四条均有实现；LLM 一条为本轮补齐（见 §3） |
| 3.4 PostgreSQL 5 张表 | ✅ | `users / orders / products / faq_documents(embedding VECTOR(512), version, status) / dialogue_logs(retrieval_results JSON, feedback, rewritten_question)`，另加 `handoff_tickets` |
| 3.4 Milvus Collection（id/embedding/metadata JSON/HNSW） | 🟡 | id INT64 + embedding FLOAT_VECTOR(512) + **faq_id**（关联 PG 取原文，避免双写不一致）；索引 HNSW/COSINE(M=16, efConstruction=200) |
| 3.5 API 契约（请求/响应字段、confidence 规则、错误码） | ✅ | 字段全含并扩展（request_id/intent/rewritten_question/handoff_ticket_id/sources）；confidence：精确查询=1.0、RAG=Top-1 重排分、<0.5 → 兜底 + 转人工；错误码 0/401/403/429/500/503（另加 404 未找到） |
| 3.6 Prometheus 指标 | ✅ | 文档 7 项全部具备（`cache_hit_ratio` 以 `cache_hits_total`/`cache_misses_total`/`cache_operation_total` 三个指标表达），另扩展 10+ 项 |
| 3.6 Grafana 面板（6 个） | ✅ | `rag-overview.json` 15 个面板（QPS/P99/错误率/缓存命中率/快速路径/各环节耗时/Token/转人工/反馈…） |
| 3.6 日志规范（JSON、request_id/trace_id/…、脱敏、禁明文 token） | ✅ | 字段齐备；手机号/邮箱/身份证/地址/sk-*/Bearer/JWT 全脱敏 |
| 3.7 docker-compose 部署（含 vllm） | 🟡 | 9 服务合并单一 `docker-compose.yml`（etcd/minio/milvus/postgres/redis/**app**/nginx/prometheus/grafana）；**无 vllm**（改用 API）。镜像已构建验证：`rag-customer-service:latest` 2.55GB |
| 3.7 nginx 80/443 | 🟡 | 监听 80（宿主 `${NGINX_PORT:-8080}`）+ 限流 429 + `/metrics` 仅内网；**未配置 443/TLS 证书** |
| 3.7 生产多副本高可用 | ✅(方案) | `docs/production_plan.md` + `nginx.conf` 预留多副本 upstream；当前按文档单机 compose 运行 |

## 5. 方案书（父文档）中项目1的更大范围——**未纳入需求文档，当前未实现**

需求设计文档（修订版）聚焦文本 RAG 客服；父方案书对「项目 1：AI 智能客服（多模态 RAG 应用）」还有更大范围，以下均**未实现**，如需可排期：

| 类别 | 未实现项 |
|---|---|
| 多模态 | 图片输入（Qwen-VL 理解）、图片应答（商品主图/尺码表）、视频应答（直播切片）、以图搜图（CLIP）、商品卡片（ID/价格/库存） |
| 多渠道 | 微信/抖音私信模拟接入（当前仅 Web 页面 + API） |
| 实时交互 | WebSocket / SSE 流式输出（当前一次性返回；N2 的 P99 目标也依赖流式首字优化） |
| 对话管理 | 更早轮次的 **LLM 摘要压缩**（当前仅滑动窗口 10 轮） |
| 安全 | 敏感话题拦截（政治/色情/竞品对比）、事实一致性校验（回复是否严格来自检索内容，当前仅靠 prompt 约束） |
| 迭代闭环 | 未解决问题自动入库、A/B 测试（prompt/模型/检索策略） |
| 检索 | Elasticsearch 倒排索引（当前用进程内 rank-bm25） |

## 6. 结论与建议优先级

已对齐需求文档的功能项，剩余可做的（按性价比排序）：

1. **N2 延迟达标**：接入 GPU 重排或 `vLLM` 本地推理 + SSE 流式返回首字（文档的 P99<500ms 预算按此设计）
2. **N6 LangSmith**：网络可用后填 `LANGSMITH_API_KEY` 即可点亮（开关与代码已就绪）
3. **N3 可用性验证**：多副本 + 长稳/混沌演练（`docs/production_plan.md` 已给方案）
4. **部署完善**：Nginx 443/TLS、compose 中补 vllm 服务（如改本地推理）
5. **可选对齐**：`RERANK_MAX_CANDIDATES` 调回 50（GPU 环境）、知识库更新 API 集成进管理后台、长文档分块支持
