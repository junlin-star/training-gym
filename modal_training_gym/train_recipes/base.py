from abc import ABC
from enum import Enum


class RecipeType(Enum):
    SLIME = "slime"
    MILES = "miles"
    STITCH = "stitch"


class BaseTrainRecipe(ABC):
    recipe_type: RecipeType
