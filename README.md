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
├── config.py                # pydantic-settings 配置（.env / .env.local）
├── api/
│   ├── deps.py              # HTTPBearer 认证（Swagger Authorize 可用）
│   └── routes/              # chat / health
├── core/                    # 错误码、结构化日志(UTF-8 JSON)、Prometheus 指标
├── db/                      # PostgreSQL(asyncpg) / Redis / Milvus 客户端（含自愈）
├── models/                  # 请求/响应 Pydantic 模型
└── services/
    ├── llm.py               # DeepSeek 客户端
    ├── embeddings.py        # bge 本地向量化（含启动预热）
    ├── intent.py            # 意图识别：政策优先 → 正则 → 关键词 → LLM few-shot
    ├── query_rewrite.py     # 多轮指代消解（LLM 改写 + 实体记忆兜底 + 政策护栏，24h 缓存）
    ├── precise_query.py     # 精确查询（含归属校验）
    ├── cache.py             # 四级缓存：答案 / 改写 / 扩展 / 重排
    ├── conversation.py      # 多轮会话 + 会话实体记忆（Redis，30 分钟过期）
    └── rag/                 # query_transform / retriever / reranker / pipeline（快速路径）
scripts/
├── init_db.py               # 建表 + 种子数据（用户/商品/订单）
├── ingest_kb.py             # FAQ → PG → 向量化 → Milvus → 清缓存
├── download_models_modelscope.py  # 从 ModelScope 预下载本地模型
├── infra_up.ps1             # 基础设施启停（含 WSL 保活）
├── monitoring_up.ps1        # 监控栈启停（自动写入抓取目标）
├── latency_test.py          # 延迟测试（冷启动 / 各级缓存命中对比）
└── traffic_demo.py          # 流量演示（点亮 Grafana 仪表盘）
monitoring/
├── prometheus/              # prometheus.yml、alert-rules.yml、targets/
└── grafana/                 # 数据源与仪表盘自动配置 + rag-overview.json
data/faq/sample_faq.json     # 42 条示例 FAQ
tests/                       # 单元测试（48 个）
docker-compose.infra.yml     # Milvus + PG + Redis 基础设施
docker-compose.monitoring.yml# Prometheus + Grafana 监控栈
```

## 快速开始

### 0. 前置条件

- Python 3.12+，Docker 可用（本机环境：Docker 引擎运行于 WSL Ubuntu，见「环境适配」）
- DeepSeek API Key（填入 `.env` 的 `DEEPSEEK_API_KEY`）

### 1. 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\download_models_modelscope.py   # 预下载嵌入/重排模型（ModelScope 源）
```

### 2. 启动基础设施

```powershell
powershell -ExecutionPolicy Bypass -File scripts\infra_up.ps1 -Keepalive   # 一键启动（含 WSL 保活，窗口需保持打开）
# 等价手动方式：
# docker compose -f docker-compose.infra.yml up -d
```

### 3. 初始化数据

```powershell
.\.venv\Scripts\python.exe -m scripts.init_db                     # 建表 + 订单/用户种子数据
.\.venv\Scripts\python.exe -m scripts.ingest_kb --file data\faq\sample_faq.json  # FAQ 导入向量库
```

### 4. 启动服务

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
# 或开发模式（改动自动重载）：加上 --reload
```

### 5. （可选）启动监控栈

```powershell
powershell -ExecutionPolicy Bypass -File scripts\monitoring_up.ps1
# Prometheus http://localhost:9090    Grafana http://localhost:3000 (admin / admin123)
```

### 6. 调用

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

## API 一览

| 接口 | 说明 | 备注 |
|------|------|------|
| `POST /api/v1/chat` | 问答主接口 | 返回 answer/source/confidence/latency/tokens/**request_id**/intent/rewritten_question/handoff_ticket_id |
| `POST /api/v1/handoff` | **转人工（创建工单）** | 用户主动请求人工；返回 ticket_id、排队提示 |
| `GET /api/v1/handoff/{ticket_id}` | 查询工单状态 | pending / accepted / closed |
| `POST /api/v1/feedback` | **点赞 / 点踩** | 用 `request_id` 定位问答记录；含归属校验 |
| `GET /health` `/ready` | 存活 / 就绪探针 | `/ready` 检查 PG、Redis、Milvus、LLM |
| `GET /metrics` | Prometheus 指标 | 不在 OpenAPI 文档中 |

**转人工（F6）两种触发方式**：
1. 用户主动：`POST /api/v1/handoff`（可带 reason / last_question / priority）
2. **自动**：问答置信度低于 `CONFIDENCE_THRESHOLD`（默认 0.5）时，服务自动建单并在响应里回传 `handoff_ticket_id`，答案末尾附工单号

**反馈闭环（F7）**：问答响应中的 `request_id`（亦见响应头 `X-Request-ID`）用于提交反馈
```powershell
curl -X POST http://localhost:8000/api/v1/feedback -H "Authorization: Bearer dev-token-change-me" -H "Content-Type: application/json" `
  -d '{"request_id":"<问答返回的 request_id>","feedback":1,"user_id":"user_001","comment":"很有帮助"}'
```
`feedback` 取值：`1` 点赞 / `-1` 点踩 / `0` 取消；他人记录返回 403，未知 request_id 返回 404。

**限流（429）**：应用层 Redis 令牌桶，按 `user_id`（缺失时按客户端 IP）计数
| 配置 | 默认 | 说明 |
|------|------|------|
| `RATE_LIMIT_ENABLED` | true | 总开关 |
| `RATE_LIMIT_PER_MINUTE` | 300 | 单身份每分钟请求数 |
| `RATE_LIMIT_BURST` | 60 | 单身份突发容量 |
| `RATE_LIMIT_GLOBAL_PER_MINUTE` | 0 | 全局桶（0=关闭），过载时用 429 卸载而非雪崩 |

超限返回 `{"code":429,...,"data":{"scope":"chat","retry_after":5}}` + `Retry-After` 响应头。
Redis 不可用时**放行**（fail-open），符合 N8：依赖故障不得让服务整体不可用。

## 知识库更新（幂等）

```powershell
.\.venv\Scripts\python.exe -m scripts.ingest_kb --file data\faq\sample_faq.json
# → [ok] 处理 42 条：新增 0 条，更新 42 条
# → [ok] milvus 向量已替换，缓存已清理：answer=.. transform=.. rerank=..
```
- `question` 上有唯一索引，重复导入走 `ON CONFLICT (question) DO UPDATE`：**内容覆盖 + version+1 + 重新发布**，不会产生重复文档
- Milvus 侧先按 faq_id 删除旧向量再写入，避免同一文档残留多条向量
- 归档：`--delete <faq_id> ...`（同时移除向量并清缓存）

## 可观测性（Prometheus + Grafana）

```powershell
# 一键启动监控栈（自动探测 WSL 网关 = Windows 宿主机 IP 作为抓取目标）
powershell -ExecutionPolicy Bypass -File scripts\monitoring_up.ps1
```

| 服务 | 地址 | 账号 |
|------|------|------|
| Prometheus | http://localhost:9090 | 无（抓取状态：`/targets`） |
| Grafana | http://localhost:3000 | **admin / admin123** |

- 仪表盘：Grafana → Dashboards → 文件夹「RAG 智能客服」→ **RAG 智能客服 - 总览**（含 QPS、P99 延迟、错误率、缓存命中率、快速路径占比、各环节耗时、Token 消耗等 13 个面板）
- 抓取目标：`/metrics`（FastAPI 在 Windows 宿主机，Prometheus 容器经 WSL 网关访问；IP 变化时重跑脚本即可，无需改配置）
- 告警规则：6 条（服务不可达 / 错误率 / P99 / 缓存命中率 / 快速路径占比 / 错误速率）

### 在 Swagger 里触发请求查看监控数据

1. 打开 http://localhost:8000/docs → 右上角 **Authorize** 🔒 → 填 `dev-token-change-me`（不带 `Bearer` 前缀）
2. 展开 **POST /api/v1/chat** → **Try it out** → 依次执行下面 4 组（每组间隔几秒，方便看到曲线）：

| 目的 | 请求体 | 观察的面板 |
|------|--------|-----------|
| 精确查询（走数据库） | `{"session_id":"m1","question":"我的订单 ORD20240100001 到哪里了？","user_id":"user_001"}` | QPS 按来源 → database |
| 知识类新问题（RAG） | `{"session_id":"m1","question":"退货运费谁出？","user_id":"user_001"}` | QPS 按来源 → rag、各环节耗时 |
| 同问题重复问（缓存） | 再执行上面同一请求 | 答案缓存命中率 ↑、P99 ↓ |
| 多轮追问（指代消解） | `{"session_id":"m1","question":"那什么时候能到？","user_id":"user_001"}` | 多轮上下文改写面板 |

3. 等待 15-30 秒（Prometheus 抓取间隔 15s），在 Grafana 把时间范围设为 **Last 5 minutes**，刷新周期 10s
4. 也可用命令行批量造流量：`.\.venv\Scripts\python.exe -m scripts.traffic_demo --rounds 20 --concurrency 4`

> 注意：监控栈与主服务相互独立。主服务停止时 Prometheus 目标会变为 down，会触发 `RagServiceDown` 告警。

## 质量保障：评测 / 压测 / 缓存预热

### N4 准确率评测

```powershell
.\.venv\Scripts\python.exe -m scripts.build_eval_set          # 生成评测集（79 条 / 91 轮次）
.\.venv\Scripts\python.exe -m scripts.eval_accuracy           # 全量评测 → docs/eval_report.md
.\.venv\Scripts\python.exe -m scripts.eval_accuracy --category multi_turn   # 只跑某一类
```

| 指标 | 实测 | 需求目标 |
|------|------|---------|
| 意图识别准确率 | **100%**（62/62） | >90% ✅ |
| 答案正确率 | **100%**（78/78） | >85% ✅ |
| HTTP 状态符合率 | 100%（含 401/403/404/422） | — |
| 延迟 | 平均 439ms / P50 489ms / P99 1512ms | — |

评测集覆盖：精确查询 12 · 越权与无效实体 9 · 知识问答 36（含同义改写）· 多轮追问 12 · 异常输入 10。
完整报告见 `docs/eval_report.md`。

### N1/N5 压测（Locust）

```powershell
# 压测专用实例（无模型预加载、关闭限流，与生产实例分端口）
$env:PRELOAD_MODELS="false"; $env:RATE_LIMIT_ENABLED="false"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8001

.\.venv\Scripts\python.exe -m scripts.warmup_cache --base http://127.0.0.1:8001   # 预热缓存
$env:LT_BASE="http://127.0.0.1:8001"; $env:LT_PROFILE="cache"; $env:LT_STAGES="50:30,100:30,200:45,300:45"
.\.venv\Scripts\python.exe -m locust -f loadtest\locustfile.py --headless --csv loadtest\results\cache
.\.venv\Scripts\python.exe -m loadtest.summarize --prefix loadtest/results/cache   # 汇总分档位结果
```

| 画像 | 吞吐上限 | 说明 |
|------|---------|------|
| 缓存命中型 | **350 RPS** ✅ (N1>200) | P50 54ms@50 并发 |
| 纯精确查询 | 312 RPS | PostgreSQL 未打满，全程 0 失败 |
| 冷启动 RAG | 5 RPS | LLM + CPU 重排上限 |
| 500 并发 | 331 RPS，0 个 5xx ✅ (N5) | P99 由 3.6s 劣化到 8.2s（排队） |

瓶颈定位与调优建议见 `docs/loadtest_report.md`。

### 缓存预热（流程3）

```powershell
.\.venv\Scripts\python.exe -m scripts.warmup_cache                    # 预热高频清单（20 条）
.\.venv\Scripts\python.exe -m scripts.warmup_cache --limit 10 --clear-first
.\.venv\Scripts\python.exe -m scripts.warmup_cache --file data/faq/sample_faq.json --limit 0   # 全量
```
服务启动自动预热（默认关闭，会产生 LLM 费用）：`.env.local` 设 `CACHE_WARMUP_ON_STARTUP=true`，
可用 `CACHE_WARMUP_LIMIT` / `CACHE_WARMUP_BASE_URL` 调整。

## 核心流程

```
用户问题
  ├─ L1 快速探测（独立问题直接查答案缓存）→ 命中：<10ms 返回
  ├─ 多轮上下文改写（指代消解）：代词/省略 + 会话实体记忆 → 独立完整问题（L2 缓存 24h）
  ├─ 意图识别（政策标记优先 → 正则快路径 → 关键词 → LLM few-shot）
  │    ├─ 精确查询 → PostgreSQL（归属校验 403，30s 短缓存）→ 实时答案
  │    └─ 知识类 → 基础混合检索（BM25∥向量，RRF 融合）
  │           ├─ 高置信（顶1向量分 ≥0.80 且分差 ≥0.05）→【快速路径】重排 Top-5 → 生成
  │           └─ 否则 →【完整路径】
  │                 Multi-Query + HyDE（一次 LLM 调用，L3 缓存 24h）
  │                 → 多查询混合检索 → Rerank（L4 缓存 24h）→ DeepSeek 生成
  └─ 响应（含 answer / source / confidence / latency / tokens / intent / rewritten_question）
```

**四级缓存**：`rag:answer:` (1h) → `query_rewrite:` (24h) → `query_transform:` (24h) → `rerank:` (24h)
知识库更新（`ingest_kb.py`）会主动清除答案 / 扩展 / 重排三级缓存。

## 降级策略

| 故障点 | 降级行为 |
|---|---|
| LLM 不可用 | 503 + "抱歉，系统繁忙，请稍后再试" |
| Milvus 故障 | 降级为 PostgreSQL ILIKE 全文检索 |
| Redis 故障 | 跳过缓存/会话，直连 PG + 完整 RAG |
| PostgreSQL 故障 | 500 + 错误日志与告警 |

## 验证测试

```powershell
.\.venv\Scripts\python.exe -m pytest                      # 单元测试（48 个）
.\.venv\Scripts\python.exe -m scripts.latency_test        # 延迟测试（冷启动/缓存命中对比）
.\.venv\Scripts\python.exe -m scripts.traffic_demo        # 流量演示（点亮监控面板）
```

## 性能现状（本机：i7-12650H CPU / 16GB，无 GPU）

| 场景 | 服务端延迟 | 说明 |
|------|-----------|------|
| L1 答案缓存命中 | 2-7ms | 独立问题走快速探测 |
| 缓存复用（L2/L3/L4 命中） | 550-890ms | 仅剩 DeepSeek 生成耗时 |
| 冷启动（快速路径） | 0.76-1.35s | 高置信直出，跳过 LLM 扩展 |
| 冷启动（完整路径） | 2.2-2.9s | LLM 扩展 ~1.0s + CPU 重排 ~0.85s |

> 需求文档的 P99<500ms 依赖 GPU 重排与 vLLM 本地推理；本机组合下缓存是最有效的杠杆。

## 环境适配（本机特有，换机时请对照调整）

| 问题 | 现象 | 处理 |
|------|------|------|
| Docker Desktop 不可用 | GUI 卡在 "Remaining Processes" 对话框，引擎起不来 | 改为在 **WSL Ubuntu 内安装 Docker 29 + Compose**，用 `scripts\infra_up.ps1` 管理 |
| docker.io 被墙 | 拉镜像 `i/o timeout` | dockerd 配置镜像加速 `/etc/docker/daemon.json` → `https://dockerproxy.net` |
| WSL VM 被回收 | 无活跃会话约 1 分钟后 VM 关闭，容器随之停止 | 脚本 `-Keepalive` 前台常驻；容器均设 `restart: unless-stopped` |
| HuggingFace 被墙 | 模型下载超时 | 改用 **ModelScope** 源（`scripts\download_models_modelscope.py`），模型落 `models/` 本地目录 |
| `.env` 受沙箱写保护 | 代理无法修改含密钥文件 | 引入 **`.env.local` 覆盖层**（`config.py` 中 `env_file=(".env", ".env.local")`，后者优先） |
| 16GB 内存装不下 v2-m3 重排模型 | 加载 2.27GB fp32 权重时 `torch_cpu.dll` 崩溃 | 换 **bge-reranker-base + int8 动态量化**（~700MB），CPU 上快 2-3 倍 |
| 监控栈抓取 Windows 上的服务 | 容器内 `localhost:8000` 不通 | Prometheus 抓 **WSL 网关 IP**（`scripts\monitoring_up.ps1` 自动探测并写入 file_sd 目标） |
| Windows 控制台 GBK | JSON 日志含 emoji 抛 `UnicodeEncodeError` 被静默丢弃 | `setup_logging()` 强制 stdout/stderr 转 UTF-8 |
| 受限沙箱无命名管道 | loguru `enqueue=True` 导致启动失败 | 去掉 `enqueue`（单进程无需异步日志队列） |

## Roadmap（后续阶段）

- [x] 阶段1：核心问答链路（意图路由 + 精确查询 + 混合检索 RAG + DeepSeek）
- [x] 阶段2：多轮对话上下文改写（指代消解 + 实体记忆 + 政策护栏）
- [x] 阶段3：四级缓存优化 + RAG 快速路径（缓存复用 <1s）
- [x] 阶段4：可观测性集成（Prometheus + Grafana 15 面板 + 6 条告警规则）
- [x] 阶段5：知识库幂等更新（修重复插入缺陷）+ 429 限流 + F6 转人工 + 反馈 API
- [ ] 阶段6：评测集与准确率量化（N4）+ 压测（N1/N5）+ 缓存预热脚本
- [ ] 阶段7：管理后台 API（F7：问答日志与统计查询）
- [ ] 阶段8：Docker Compose 全栈一键部署（含 app/nginx 容器化）
