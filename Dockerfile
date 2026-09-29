# RAG 智能客服 —— 应用镜像
#
# 构建：docker build -t rag-customer-service:latest .
# 说明：
#   - 使用 CPU 版 PyTorch（镜像体积从 ~6GB 降到 ~2GB）
#   - 本地模型（嵌入/重排）体积大且会随知识库换代，**不打进镜像**，通过卷挂载 ./models
#   - 非 root 用户运行；健康检查走 /health
#   - 国内网络可选镜像加速（默认官方源；实测 aliyun 的 pytorch-wheels 快一个数量级）：
#       docker build \
#         --build-arg TORCH_INDEX_URL=https://mirrors.aliyun.com/pytorch-wheels/cpu \
#         --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
#         -t rag-customer-service:latest .
FROM python:3.12-slim AS base

# 依赖源（可覆盖，见文件头注释）
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG PIP_INDEX_URL=https://pypi.org/simple

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TMPDIR=/tmp

# libgomp1: torch/transformers 运行时依赖；tini 用于正确转发信号
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 tini \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/app

# ---- 依赖层（先装 torch CPU 再装其余，充分利用构建缓存） ----
COPY requirements.txt ./
RUN pip install torch --index-url "$TORCH_INDEX_URL" \
 && pip install -r requirements.txt --index-url "$PIP_INDEX_URL" \
 && pip install "psutil>=5.9" --index-url "$PIP_INDEX_URL"

# ---- 代码层 ----
COPY app ./app
COPY scripts ./scripts
COPY data ./data

# 容器内默认配置：中间件按服务名寻址；模型走挂载目录；关闭模型预加载以控制内存
ENV APP_NAME=rag-customer-service \
    PG_HOST=postgres \
    PG_PORT=5432 \
    REDIS_URL=redis://redis:6379/0 \
    MILVUS_URI=http://milvus:19530 \
    EMBEDDING_MODEL=/srv/models/bge-small-zh-v1.5 \
    RERANK_MODEL=/srv/models/bge-reranker-base \
    PRELOAD_MODELS=false \
    PROMETHEUS_URL=http://prometheus:9090

RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /srv/models \
 && chown -R appuser:appuser /srv
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status==200 else 1)"

ENTRYPOINT ["/usr/bin/tini", "--"]
# 单副本默认 1 worker；多副本方案见 docs/production_plan.md（模型需外置为独立推理服务）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "15"]
