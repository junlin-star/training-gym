from abc import ABC
from enum import Enum


class RecipeType(Enum):
    SLIME = "slime"
    MILES = "miles"
    VIME = "vime"


class BaseTrainRecipe(ABC):
    recipe_type: RecipeType
