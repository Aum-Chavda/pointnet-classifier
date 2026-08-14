# src/models/registry.py
"""
Model registry — factory pattern for creating PointNet-style models by name.

OOP  : Factory pattern — centralised object creation, caller never imports
       concrete classes directly
       Registry pattern — hash map of name -> class, O(1) lookup
DS&A : dict (hash map) — O(1) register, O(1) build, O(N) list
Role : Single place to register and instantiate all model variants
"""

from __future__ import annotations
from typing import Type
from src.models.base import BasePointNetBackbone
from src.utils.config import PointNetConfig


class ModelRegistry:
    """
    Central registry for all PointNet-style backbone models.

    OOP  : Registry pattern — _registry is a CLASS variable (shared across
           all instances). @classmethod decorators mean you call methods on
           the class itself, not an instance:
               ModelRegistry.build("pointnet", cfg)   # correct
               ModelRegistry().build("pointnet", cfg) # also works but wasteful

    DS&A : _registry is a dict (hash map)
           - register() : O(1) insert  — dict[name] = cls
           - build()    : O(1) lookup  — dict[name]
           - list()     : O(N) scan    — list(dict.keys())

    Design decision — why class methods not module-level functions?
        Class methods give us a namespace (ModelRegistry.build vs just build)
        and make it easy to subclass the registry for testing or extension.
        The _registry dict is shared state — class variable is the right home.
    """

    # Class variable — ONE dict shared across the entire program
    # Maps string name → concrete class (subclass of BasePointNetBackbone)
    # DS&A : hash map, O(1) average case insert and lookup
    _registry: dict[str, Type[BasePointNetBackbone]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        model_cls: Type[BasePointNetBackbone],
    ) -> None:
        """
        Register a model class under a given name.

        OOP  : @classmethod — cls is the class itself (ModelRegistry), not an instance
               This lets subclasses override registration behaviour if needed
        DS&A : O(1) dict insert

        Args:
            name      : string key (e.g. "pointnet", "pointnet_vanilla")
            model_cls : the class to register (not an instance — the class itself)

        Raises:
            TypeError  : if model_cls is not a subclass of BasePointNetBackbone
            ValueError : if name is already registered (prevents silent overwrites)

        Example:
            ModelRegistry.register("pointnet", PointNet)
        """
        # Guard — only register valid backbone subclasses
        if not (isinstance(model_cls, type)
                and issubclass(model_cls, BasePointNetBackbone)):
            raise TypeError(
                f"model_cls must be a subclass of BasePointNetBackbone, "
                f"got {model_cls}"
            )

        # Guard — prevent silent overwrites (hard to debug if two files
        # register different classes under the same name)
        if name in cls._registry:
            raise ValueError(
                f"Model '{name}' is already registered. "
                f"Use a different name or call ModelRegistry.remove('{name}') first."
            )

        cls._registry[name] = model_cls

    @classmethod
    def build(
        cls,
        name: str,
        config: PointNetConfig,
    ) -> BasePointNetBackbone:
        """
        Build and return a model instance by name.

        OOP  : Factory method — caller asks for "pointnet", gets back a fully
               constructed PointNet object without knowing the class name
        DS&A : O(1) hash map lookup + O(1) object construction

        Args:
            name   : registered model name (e.g. "pointnet")
            config : PointNetConfig instance passed to model constructor

        Returns:
            Instantiated model — typed as BasePointNetBackbone

        Raises:
            KeyError : if name is not registered (with helpful error message
                       listing what IS registered)

        Example:
            model = ModelRegistry.build("pointnet", cfg)
            # returns PointNet(cfg) without importing PointNet directly
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(
                f"Model '{name}' not found in registry. "
                f"Available models: {available}"
            )

        model_cls = cls._registry[name]   # O(1) lookup
        return model_cls(config)          # call constructor — returns instance

    @classmethod
    def list_models(cls) -> list[str]:
        """
        Return sorted list of all registered model names.

        DS&A : O(N) — iterates over dict keys where N = number of registered models
               sorted() = O(N log N) — small N in practice

        Useful for:
            - Debugging ("what models are available?")
            - CLI argument validation ("--model must be one of [...]")
        """
        return sorted(cls._registry.keys())

    @classmethod
    def remove(cls, name: str) -> None:
        """
        Remove a model from the registry.

        DS&A : O(1) dict delete
        Used in tests — lets us register/unregister test models
        without polluting the real registry between test runs.

        Raises:
            KeyError : if name not found
        """
        if name not in cls._registry:
            raise KeyError(f"Model '{name}' not found in registry.")
        del cls._registry[name]

    @classmethod
    def clear(cls) -> None:
        """
        Remove ALL registered models.

        DS&A : O(1) — replaces dict with empty dict
        Used in tests to reset registry state between test cases.
        """
        cls._registry.clear()


# ======================================================================
# Register all models here — one line per model
# ======================================================================
# Import here (not at top of file) to avoid circular imports:
#   registry.py imports pointnet.py
#   pointnet.py imports base.py
#   base.py does NOT import registry.py  → no circle
from src.models.pointnet import PointNet

ModelRegistry.register("pointnet", PointNet)