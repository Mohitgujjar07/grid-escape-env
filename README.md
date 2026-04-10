# Grid Escape — OpenEnv

A 6x6 grid navigation game built for RL agent training using Meta's OpenEnv framework.

The agent must navigate from start → goal tile (G), avoiding traps (T) and walls (W).

## Quick Start

```bash
pip install -r server/requirements.txt
uvicorn server.app:app --reload --port 8000
```

Open **http://localhost:8000/web** — play manually or click "Run AI Agent" to watch BFS solve it.

## Docker

```bash
docker build -t grid-escape-env .
docker run -p 7860:7860 grid-escape-env
```

Open **http://localhost:7860/web**

## Hugging Face Spaces

1. Create a new Space → SDK: Docker
2. Push this repo to the Space
3. Access at `https://YOUR_USERNAME-grid-escape-env.hf.space/web`

## Run BFS Demo Agent

```bash
python client.py
```

## Reward Function

| Event | Reward |
|-------|--------|
| Reach goal | +10.0 + speed bonus |
| Hit trap | -1.0 |
| Hit wall/boundary | -0.1 |
| Each step | -0.05 |

## WebSocket API

Connect to `ws://host/ws`

```json
{ "method": "reset", "level": 0 }
{ "method": "step", "action": { "move": "right" } }
{ "method": "state" }
```
