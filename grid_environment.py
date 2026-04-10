import uuid
import random
from models import GridAction, GridObservation, GridState, MoveAction

SIZE = 6
MAX_STEPS = 30

LEVELS = [
    {
        "agent": [0, 0],
        "goal": [5, 5],
        "walls": [[1,1],[1,2],[2,4],[3,1],[3,2],[4,3],[4,4]],
        "traps": [[2,1],[3,4],[4,0]],
    },
    {
        "agent": [0, 0],
        "goal": [5, 5],
        "walls": [[0,2],[1,2],[2,2],[2,3],[3,3],[4,3],[4,2]],
        "traps": [[1,3],[2,5],[3,1],[4,5]],
    },
    {
        "agent": [0, 0],
        "goal": [5, 5],
        "walls": [[0,3],[1,1],[1,3],[2,1],[2,3],[3,3],[3,5],[4,1],[4,3]],
        "traps": [[1,4],[2,5],[3,0],[4,4]],
    },
]


class GridEnvironment:
    def __init__(self):
        self.episode_id = ""
        self.agent = [0, 0]
        self.goal = [5, 5]
        self.walls = []
        self.traps = []
        self.step_count = 0
        self.score = 0.0
        self.done = False
        self.won = False
        self.level = None

    def reset(self, level_idx: int = None):
        self.episode_id = str(uuid.uuid4())
        self.step_count = 0
        self.score = 0.0
        self.done = False
        self.won = False

        lvl = LEVELS[level_idx % len(LEVELS)] if level_idx is not None else random.choice(LEVELS)
        self.level = lvl
        self.agent = list(lvl["agent"])
        self.goal = list(lvl["goal"])
        self.walls = [list(w) for w in lvl["walls"]]
        self.traps = [list(t) for t in lvl["traps"]]

        return self._obs("Episode started! Reach the G tile. Avoid T traps!"), self._state()

    def step(self, action: GridAction):
        if self.done:
            return self._obs("Episode already done."), 0.0, True, self._state()

        dr, dc = {"up": (-1,0), "down": (1,0), "left": (0,-1), "right": (0,1)}[action.move]
        nr, nc = self.agent[0] + dr, self.agent[1] + dc

        if not (0 <= nr < SIZE and 0 <= nc < SIZE):
            reward = -0.1
            self.score += reward
            self.step_count += 1
            if self.step_count >= MAX_STEPS:
                self.done = True
            return self._obs("Bumped into boundary!"), reward, self.done, self._state()

        if [nr, nc] in self.walls:
            reward = -0.1
            self.score += reward
            self.step_count += 1
            if self.step_count >= MAX_STEPS:
                self.done = True
            return self._obs("Blocked by a wall!"), reward, self.done, self._state()

        self.agent = [nr, nc]
        self.step_count += 1

        if [nr, nc] in self.traps:
            reward = -1.0
            self.score += reward
            self.done = True
            self.won = False
            return self._obs("TRAP! You stepped on T. Episode over."), reward, True, self._state()

        if [nr, nc] == self.goal:
            bonus = max(0, (MAX_STEPS - self.step_count) * 0.1)
            reward = 10.0 + bonus
            self.score += reward
            self.done = True
            self.won = True
            return self._obs(f"GOAL! Reached G! Speed bonus: +{bonus:.1f}"), reward, True, self._state()

        reward = -0.05
        self.score += reward

        if self.step_count >= MAX_STEPS:
            self.done = True
            return self._obs("Out of steps!"), reward, True, self._state()

        return self._obs("Moved " + action.move + "."), reward, False, self._state()

    def _build_grid(self):
        grid = [['.' for _ in range(SIZE)] for _ in range(SIZE)]
        for w in self.walls:
            grid[w[0]][w[1]] = 'W'
        for t in self.traps:
            grid[t[0]][t[1]] = 'T'
        grid[self.goal[0]][self.goal[1]] = 'G'
        grid[self.agent[0]][self.agent[1]] = 'A'
        return grid

    def _obs(self, msg: str) -> GridObservation:
        return GridObservation(
            grid=self._build_grid(),
            agent_pos=list(self.agent),
            goal_pos=list(self.goal),
            step=self.step_count,
            message=msg,
        )

    def _state(self) -> GridState:
        return GridState(
            episode_id=self.episode_id,
            step=self.step_count,
            max_steps=MAX_STEPS,
            score=round(self.score, 2),
            done=self.done,
            won=self.won,
        )

    def get_state(self) -> GridState:
        return self._state()
