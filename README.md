# 企业级智能客服 RAG 系统

基于需求设计文档实现的生产级高并发 RAG 客服系统：自动识别**精确查询类**（订单/物流）与**知识类**（FAQ）问题，
前者走 PostgreSQL 实时查询，后者走 混合检索 + Rerank + DeepSeek 生成 的完整 RAG 链路。

## 技术栈

| 层 | 选型 |
|---|---|
| Web 框架 | FastAPI (async) + uvicorn |
| LLM | DeepSeek API（OpenAI 兼容接口，经 LangChain ChatOpenAI 接入） |
| 嵌入模型 | BAAI/bge-small-zh-v1.5（本地 CPU，512 维） |
| 重排模型 | BAAI/bge-reranker-v2-m3（本地 CPU，Cross-Encoder） |
| 向量库 | Milvus (HNSW + COSINE) |
| 关系库 | PostgreSQL 15 + pgvector |
| 缓存/会话 | Redis 7 |
| 检索 | BM25(rank-bm25) + 向量 混合检索，RRF 融合 |
| 可观测性 | Loguru(JSON) + Prometheus + （可选 LangSmith） |
| 部署 | Docker Compose |

## 目录结构

```
app/
├── main.py                  # FastAPI 入口（lifespan、中间件、异常处理）
├── config.py                # pydantic-settings 配置（.env）
├── api/
│   ├── deps.py              # Bearer Token 认证
│   └── routes/              # chat / health
├── core/                    # 错误码、结构化日志、Prometheus 指标
├── db/                      # PostgreSQL(asyncpg) / Redis / Milvus 客户端
├── models/                  # 请求/响应 Pydantic 模型
└── services/
    ├── llm.py               # DeepSeek 客户端
    ├── embeddings.py        # bge 本地向量化
    ├── intent.py            # 意图识别：正则 → 关键词 → LLM few-shot
    ├── precise_query.py     # 精确查询（含归属校验）
    ├── cache.py             # Redis 答案缓存
    ├── conversation.py      # 多轮会话（Redis，30 分钟过期）
    └── rag/                 # query_transform / retriever / reranker / pipeline
scripts/
├── init_db.py               # 建表 + 种子数据（用户/商品/订单）
├── ingest_kb.py             # FAQ → PG → 向量化 → Milvus → 清缓存
└── download_models.py       # 预下载本地模型
data/faq/sample_faq.json     # 43 条示例 FAQ
tests/                       # 单元测试
docker-compose.infra.yml     # Milvus + PG + Redis 基础设施
```

## 快速开始

### 0. 前置条件

- Python 3.12+，Docker Desktop 运行中
- DeepSeek API Key（填入 `.env` 的 `DEEPSEEK_API_KEY`）

### 1. 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\download_models.py   # 预下载嵌入/重排模型（约 2-3GB）
```

### 2. 启动基础设施

```powershell
docker compose -f docker-compose.infra.yml up -d
# 等待 milvus healthy（首次拉镜像较慢）
docker compose -f docker-compose.infra.yml ps
```

### 3. 初始化数据

```powershell
.\.venv\Scripts\python.exe scripts\init_db.py                     # 建表 + 订单/用户种子数据
.\.venv\Scripts\python.exe scripts\ingest_kb.py --file data\faq\sample_faq.json  # FAQ 导入向量库
```

### 4. 启动服务

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. 调用

```powershell
# 健康检查
curl http://localhost:8000/health
curl http://localhost:8000/ready

# 知识类问题（RAG）
curl -X POST http://localhost:8000/api/v1/chat -H "Authorization: Bearer dev-token-change-me" -H "Content-Type: application/json" `
  -d '{"session_id":"s1","question":"七天无理由退货的条件是什么？","user_id":"user_001"}'

# 精确查询（订单，注意归属校验）
curl -X POST http://localhost:8000/api/v1/chat -H "Authorization: Bearer dev-token-change-me" -H "Content-Type: application/json" `
  -d '{"session_id":"s1","question":"我的订单ORD20240100001到哪了？","user_id":"user_001"}'

# 指标
curl http://localhost:8000/metrics
```

## 核心流程

```
用户问题
  ├─ 意图识别（正则快路径 → 关键词 → LLM few-shot）
  │    ├─ 精确查询 → PostgreSQL（归属校验，30s 短缓存）→ 实时答案
  │    └─ 知识类 → Redis 缓存命中？→ 是：直接返回
  │                              └─ 否：Multi-Query + HyDE（并行）
  │                                    → 混合检索(BM25∥向量, RRF 融合)
  │                                    → bge-reranker 精排 Top-5
  │                                    → DeepSeek 生成 → 写缓存(TTL 1h)
  └─ 响应（含 confidence / latency / tokens）
```

## 降级策略

| 故障点 | 降级行为 |
|---|---|
| LLM 不可用 | 503 + "抱歉，系统繁忙，请稍后再试" |
| Milvus 故障 | 降级为 PostgreSQL ILIKE 全文检索 |
| Redis 故障 | 跳过缓存/会话，直连 PG + 完整 RAG |
| PostgreSQL 故障 | 500 + 错误日志与告警 |

## 验证测试

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Roadmap（后续阶段）

- [x] 阶段1：核心问答链路（意图路由 + 精确查询 + 混合检索 RAG + DeepSeek）
- [ ] 阶段2：可观测性完善（Prometheus/Grafana 面板、LangSmith 追踪、request_id 全链路）
- [ ] 阶段3：缓存命中率优化 + 多轮会话强化
- [ ] 阶段4：降级演练与容灾脚本
- [ ] 阶段5：知识库管理 API（管理后台）
- [ ] 阶段6：Docker Compose 全栈一键部署（含 app/nginx/prometheus/grafana）
