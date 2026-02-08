# models/registry.py
import pkgutil
import importlib
from typing import Dict, List, Type, Any, Set

NN_Registry = Dict[str, Dict[str, Type]] # category -> name -> class
Trad_Registry = Dict[str, Type]

class NNModelLibrary:
    _registry: NN_Registry = {}
    _discovered_packages: Set[str] = set()

    @classmethod
    def register(cls, category: str, name: str = None):
        def decorator(model_cls: Type):
            key = name or model_cls.__name__
            if key in cls._registry.get(category, {}):
                raise KeyError(f"Model '{key}' already registered in category '{category}'")
            cls._registry.setdefault(category, {})[key] = model_cls
            return model_cls
        return decorator

    @classmethod
    def autodiscover(cls, package: str):
        """
        Import all modules in a package exactly once per package.
        Safe to call multiple times with different packages.
        This method MUST be executed in a main file to register all models.
        """
        if package in cls._discovered_packages:
            return

        pkg = importlib.import_module(package)

        for _, modname, ispkg in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            if not ispkg:
                importlib.import_module(modname)

        cls._discovered_packages.add(package)

    @classmethod
    def items(cls) -> NN_Registry:
        return cls._registry
    
    @classmethod
    def list_categories(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def list_models(cls, category: str) -> List[str]:
        return list(cls._registry.get(category, {}).keys())

    @classmethod
    def get(cls, category: str, name: str) -> Type:
        return cls._registry[category].get(name)

    @classmethod
    def instantiate(cls, category: str, name: str, *args, **kwargs) -> Any:
        return cls.get(category, name)(*args, **kwargs)

class TradModelLibrary:
    _registry: Trad_Registry = {}
    _discovered_packages: Set[str] = set()

    @classmethod
    def register(cls, name: str = None):
        def decorator(model_cls: Type):
            key = name or model_cls.__name__
            if key in cls._registry:
                raise KeyError(f"Model '{key}' already registered in")
            cls._registry[key] = model_cls
            return model_cls
        return decorator

    @classmethod
    def autodiscover(cls, package: str):
        """
        Import all modules in a package exactly once per package.
        Safe to call multiple times with different packages.
        This method MUST be executed in a main file to register all models.
        """
        if package in cls._discovered_packages:
            return

        pkg = importlib.import_module(package)

        for _, modname, ispkg in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            if not ispkg:
                importlib.import_module(modname)

        cls._discovered_packages.add(package)

    @classmethod
    def items(cls) -> Trad_Registry:
        return cls._registry

    @classmethod
    def list_models(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def get(cls, name: str) -> Type:
        return cls._registry.get(name)

    @classmethod
    def instantiate(cls, name: str, *args, **kwargs) -> Any:
        return cls.get(name)(*args, **kwargs)