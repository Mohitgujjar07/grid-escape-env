from enum import Enum
from typing import List, Optional
from pydantic import BaseModel


class MoveAction(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class GridAction(BaseModel):
    move: MoveAction


class GridObservation(BaseModel):
    grid: List[List[str]]   # 6x6 grid: 'A'=agent, 'G'=goal, 'T'=trap, '.'=empty, 'W'=wall
    agent_pos: List[int]    # [row, col]
    goal_pos: List[int]
    step: int
    message: str


class GridState(BaseModel):
    episode_id: str
    step: int
    max_steps: int
    score: float
    done: bool
    won: bool
