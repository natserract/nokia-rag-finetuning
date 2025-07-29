FROM python:3.11-slim-bookworm

RUN grep -rl 'deb.debian.org' /etc/apt/ | xargs sed -i 's|http[s]*://deb.debian.org|https://mirrors.tuna.tsinghua.edu.cn|g' && \
    apt-get update && \
    apt-get install -y curl gcc && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.7.5 /uv /uvx /bin/
ENV UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app

COPY pyproject.toml /app/
RUN uv venv --python=3.11
RUN . .venv/bin/activate
RUN uv pip install --verbose -r pyproject.toml -i https://mirrors.aliyun.com/pypi/simple/
ENV PATH="/app/.venv/bin:$PATH"

COPY main.py /app/main.py
COPY dataset /app/dataset

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Create required directories
RUN mkdir -p /root/.ollama/models

# Download model during build with better handling
RUN ollama serve > /dev/null 2>&1 & \
    sleep 25 && \
    ollama pull deepseek-r1:8b && \
    ollama pull mxbai-embed-large

COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["sh", "/start.sh"]
