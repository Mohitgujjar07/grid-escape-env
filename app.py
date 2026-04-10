import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import json, traceback

from models import GridAction
from server.grid_environment import GridEnvironment

app = FastAPI(title="Grid Escape OpenEnv")
env = GridEnvironment()


@app.get("/")
async def root():
    return {"status": "ok", "env": "grid_escape", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            method = data.get("method")

            if method == "reset":
                level = data.get("level", None)
                obs, state = env.reset(level_idx=level)
                await websocket.send_text(json.dumps({
                    "observation": obs.model_dump(),
                    "state": state.model_dump(),
                }))

            elif method == "step":
                action = GridAction(**data["action"])
                obs, reward, done, state = env.step(action)
                await websocket.send_text(json.dumps({
                    "observation": obs.model_dump(),
                    "reward": reward,
                    "done": done,
                    "state": state.model_dump(),
                }))

            elif method == "state":
                await websocket.send_text(json.dumps({
                    "state": env.get_state().model_dump()
                }))

            else:
                await websocket.send_text(json.dumps({"error": f"Unknown method: {method}"}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"error": str(e), "trace": traceback.format_exc()}))
        except:
            pass


@app.get("/web", response_class=HTMLResponse)
async def web_interface():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_ui.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
