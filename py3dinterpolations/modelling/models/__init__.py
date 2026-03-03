"""Model registry for 3D interpolation."""

from ...core.types import ModelType
from .base import BaseModel
from .idw import IDWModel
from .kriging import KrigingModel
from .sklearn_model import SklearnModel

MODEL_REGISTRY: dict[ModelType, type[BaseModel]] = {
    ModelType.ORDINARY_KRIGING: KrigingModel,
    ModelType.IDW: IDWModel,
}

__all__ = [
    "MODEL_REGISTRY",
    "BaseModel",
    "IDWModel",
    "KrigingModel",
    "SklearnModel",
    "get_model",
]


def get_model(model_type: ModelType | str, **kwargs: object) -> BaseModel:
    """Instantiate a model by type.

    Args:
        model_type: Model identifier, either a ModelType enum or its string value.
        **kwargs: Parameters passed to the model constructor.

    Returns:
        An instantiated model ready for fit().

    Raises:
        ValueError: If model_type is not in the registry.
    """
    model_type = ModelType(model_type)
    cls = MODEL_REGISTRY.get(model_type)
    if cls is None:
        available = list(MODEL_REGISTRY.keys())
        msg = f"Model {model_type!r} not in registry. Available: {available}"
        raise ValueError(msg)
    return cls(**kwargs)
