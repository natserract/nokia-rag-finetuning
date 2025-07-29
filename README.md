# Nokia User Guide - RAG, Finetuning

## Overview

![Overview1](https://res.cloudinary.com/dqo6txtrv/image/upload/v1750841103/Screenshot_2025-06-25_at_16.41.03_mr9qhl.png)

![Overview2](https://res.cloudinary.com/dqo6txtrv/image/upload/v1750841103/Screenshot_2025-06-25_at_16.42.05_jtgore.png)

![Overview3](https://res.cloudinary.com/dqo6txtrv/image/upload/v1753766346/finetuning_fvtcn3.jpg)

![Overview4](https://res.cloudinary.com/dqo6txtrv/image/upload/v1753766038/Screenshot_2025-06-26_at_16.13.52_l9c3e8.png)

**Dataset Information**:
```bash
DatasetDict({
    train: Dataset({
        features: ['id', 'question', 'answer'],
        num_rows: 2470
    })
})
```

## Requirements

- Python 3.11+
- uv (package manager) https://docs.astral.sh/uv/getting-started/installation/
- Docker
- Ollama

**Windows**:
- Enable WSL
- Install Ollama `curl -fsSL https://ollama.com/install.sh | sh`

**When Ollama is installed, you can start it using the following command:**

```bash
ollama pull deepseek-r1:8b

ollama pull mxbai-embed-large # embedding model
```

**Setup environment:**
```bash
cp .env.example .env
```

> Add OPEN_AI_KEY=<your_openai_key> if you want to use OpenAI model.

**Python environment:**
```bash
uv venv
source .venv/bin/activate
```

**Install required packages:**
```bash
uv pip install .
```

## Run Server

**Run Server**
```bash
uv run main.py --run_server
```

Visit http://0.0.0.0:8000/chat

**Payload**:
```json
{
    "model": "model-202507071447",
    "question": "What type of information can you expect to find in the Nokia 8 User Guide?"
}
```

**Sample Response**:
```json
{
    "response": "<pad> You can expect to find usage instructions, safety information, and device specifications in the Nokia 8 User Guide.</s>"
}
```

## Usage (CLI command)

**CLI:**
```bash
# default
uv run main.py

# debug
uv run main.py --debug

# model
uv run main.py --model=ollama

# training
uv run main.py --train_from_scratch

# inference
uv run main.py --model_name_or_checkpoint_path=<name> # Check on training/* directory to get latest checkpoint

# run server
uv run main.py --run_server

# generate qa dataset
uv run main.py --generate
```

## Usage (with docker)

```bash
docker compose up --build -d # or
docker compose up --build --force-recreate -d

# run
docker-compose run -it nokiaguideai

# with arguments
docker-compose run -it nokiaguideai python main.py --model=openai
```

**Start Ollama and its dependencies using Docker Compose:**

if gpu is configured:
```bash
docker compose -f docker-compose-ollama-gpu.yaml up -d
```

else
```bash
docker compose -f docker-compose-ollama.yaml up -d
```

Visit http://localhost:7869 to verify Ollama is running.

## Troubleshooting

- `chromadb.errors.InvalidArgumentError`: Collection expecting embedding with dimension ..

Remove db directory, and type: `uv run main.py --mode=embed`

- **RuntimeError: MPS backend out of memory**

```bash
RuntimeError: MPS backend out of memory (MPS allocated: 17.93 GB, other allocations: 162.97 MB, max allowed: 18.13 GB). Tried to allocate 60.00 MB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).
```

Run:

```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

uv run main.py
```

> Remember this may cause system failure

- **Dataset doesn't update**

Clear cache https://huggingface.co/docs/datasets/cache
