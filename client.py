import asyncio
import json
import websockets
from collections import deque
from models import GridAction, GridObservation, GridState, MoveAction


class GridEscapeEnv:
    """
    OpenEnv-compatible client for Grid Escape.

    Async:
        async with GridEscapeEnv("ws://localhost:8000") as env:
            obs, state = await env.reset()
            obs, reward, done, state = await env.step(GridAction(move=MoveAction.RIGHT))

    Sync:
        with GridEscapeEnv("ws://localhost:8000").sync() as env:
            obs, state = env.reset()
    """

    def __init__(self, base_url="ws://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._ws = None

    async def __aenter__(self):
        url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        self._ws = await websockets.connect(f"{url}/ws")
        return self

    async def __aexit__(self, *args):
        if self._ws:
            await self._ws.close()

    async def reset(self, level: int = 0):
        await self._ws.send(json.dumps({"method": "reset", "level": level}))
        data = json.loads(await self._ws.recv())
        return GridObservation(**data["observation"]), GridState(**data["state"])

    async def step(self, action: GridAction):
        await self._ws.send(json.dumps({"method": "step", "action": action.model_dump()}))
        data = json.loads(await self._ws.recv())
        return GridObservation(**data["observation"]), data["reward"], data["done"], GridState(**data["state"])

    async def state(self):
        await self._ws.send(json.dumps({"method": "state"}))
        data = json.loads(await self._ws.recv())
        return GridState(**data["state"])

    def sync(self):
        return _SyncWrapper(self)


class _SyncWrapper:
    def __init__(self, env):
        self._env = env
        self._loop = asyncio.new_event_loop()

    def __enter__(self):
        self._loop.run_until_complete(self._env.__aenter__())
        return self

    def __exit__(self, *args):
        self._loop.run_until_complete(self._env.__aexit__(*args))
        self._loop.close()

    def reset(self, level=0):
        return self._loop.run_until_complete(self._env.reset(level))

    def step(self, action):
        return self._loop.run_until_complete(self._env.step(action))


def bfs_agent(obs: GridObservation):
    """BFS pathfinding agent — finds shortest safe path to goal."""
    grid = obs.grid
    start = tuple(obs.agent_pos)
    goal = tuple(obs.goal_pos)
    R, C = len(grid), len(grid[0])
    dirs = [("up", -1, 0), ("down", 1, 0), ("left", 0, -1), ("right", 0, 1)]

    queue = deque([(start, [])])
    visited = {start}

    while queue:
        (r, c), path = queue.popleft()
        for move, dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < R and 0 <= nc < C):
                continue
            if (nr, nc) in visited:
                continue
            cell = grid[nr][nc]
            if cell in ("W", "T"):
                continue
            visited.add((nr, nc))
            new_path = path + [move]
            if (nr, nc) == goal:
                return new_path[0]
            queue.append(((nr, nc), new_path))

    # fallback: random safe move
    import random
    safe = [m for m, dr, dc in dirs
            if 0 <= start[0]+dr < R and 0 <= start[1]+dc < C
            and grid[start[0]+dr][start[1]+dc] not in ("W", "T")]
    return random.choice(safe) if safe else "right"


async def demo():
    print("Grid Escape — BFS Agent Demo\n" + "="*30)
    async with GridEscapeEnv("ws://localhost:8000") as env:
        for level in range(3):
            obs, state = await env.reset(level=level)
            print(f"\nLevel {level + 1} — Start: {obs.agent_pos} → Goal: {obs.goal_pos}")

            while not state.done:
                move = bfs_agent(obs)
                obs, reward, done, state = await env.step(GridAction(move=MoveAction(move)))
                row = "".join(obs.grid[r][c] for r in range(len(obs.grid)) for c in range(len(obs.grid[r])))
                print(f"  Step {state.step:2d} | move={move:5s} | reward={reward:+.2f} | score={state.score:.2f} | {obs.message}")

            result = "WIN!" if state.won else "LOST"
            print(f"  → {result} Final score: {state.score:.2f}")


if __name__ == "__main__":
    asyncio.run(demo())
