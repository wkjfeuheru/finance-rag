# ========== Stage 1: 构建前端 ==========
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ========== Stage 2: 后端运行时 ==========
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    # HuggingFace 模型缓存（嵌入模型 + BGE reranker）指向持久卷，
    # 避免容器重建/重启后重新下载约 2GB 权重
    HF_HOME=/app/onnx_cache/hf_home

# 系统依赖：gcc 编译 sentence-transformers/onnxruntime 扩展，libgomp1 供 onnxruntime 运行
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 复制后端代码
COPY finance_rag/ ./finance_rag/
COPY main.py .

# 复制前端构建产物（来自 Stage 1）
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# 以非 root 用户运行，挂载的文件与模型缓存目录保持可写
RUN groupadd --system app && useradd --system --gid app --home-dir /app app \
    && mkdir -p /app/files/docs /app/onnx_cache /app/assets/state \
    && chown -R app:app /app

USER app

EXPOSE 8000

# 生产模式启动（不开 reload）
CMD ["uvicorn", "finance_rag.src.main:app", "--host", "0.0.0.0", "--port", "8000"]
