from dataclasses import dataclass
from enum import Enum

import numpy as np


class Color(Enum):
    """
    Represents the color of a ball in the pool game.
    """
    WHITE = (255, 255, 255)
    YELLOW = (0, 255, 255)
    BLUE = (255,0,0)
    RED = (0,0,255)
    PURPLE = (255,0,255)
    ORANGE = (0,165,255)
    GREEN = (0,255,0)
    BROWN = (42,42,165)
    BLACK = (0,0,0)


@dataclass
class Ball:
    """
    Represents a ball in the pool game.
    """
    position: np.array
    stripped: bool = False
    radius: int = 15
    color: Color = Color.WHITE
    in_pocket: bool = False


@dataclass
class WhiteBall(Ball):
    direction: np.array = np.array([0, 0])

@dataclass
class VelocityVector:
    position: np.array
    direction: np.array


@dataclass
class Cue:
    """
    Represents the cue in the pool game.
    """
    position: np.array
    direction: np.array


@dataclass
class Board:
    """
    Represents the board in the pool game.
    """
    width: int
    height: int