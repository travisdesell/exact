# models/registry.py
import pkgutil
import importlib
from typing import Dict, Type, Any, Set

NN_Registry = Dict[str, Dict[str, Type]] # category -> name -> class
Trad_Registry = Dict[str, Type]

class NNModelLibrary:
    """
    Neural Network Model Library to contain all available models 
    like a library that can be queried.
    """
    _registry: NN_Registry = {}
    _discovered_packages: Set[str] = set()

    @classmethod
    def register(cls, category: str, name: str | None = None):
        """
        Register a neural network model using this decorator.

        Args:
            category (str): Name of category the model architecture belongs to.
            name (str | None): Name of the model. Default = None. 
                If None, name of the class will be used as default.
        
        Raise:
            KeyError: If a model already exists in the library.
        """
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

        Args:
            package (str): Name or pythonic module path to detect models.
                Example: 'src.models'
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
        """
        Get entire library of neural network models.

        Returns:
            NN_Registry: Dictionary of all categories and their models.
        """
        return cls._registry
    
    @classmethod
    def list_categories(cls) -> list[str]:
        """
        Get a list of available categories.

        Returns:
            list[str]: List of categories present in the library.
        """
        return list(cls._registry.keys())

    @classmethod
    def list_models(cls, category: str) -> list[str]:
        """
        Get a list of all available models in a particular category.

        Args:
            category (str): Category name of the required neural networks.
        
        Returns:
            list[str]: List of all available neural network models 
                which belong to the give category.
        """
        return list(cls._registry.get(category, {}).keys())

    @classmethod
    def get(cls, category: str, name: str) -> Type | None:
        """
        Get a particular model for the given category name.

        Args:
            category (str): Category name of the required model.
            name (str): Name of the required model.
        
        Returns:
            Type | None: Class of the required neural network model.    
        """
        category_dict = cls._registry.get(category)
        if category_dict is None:
            return None
        return category_dict.get(name)

    @classmethod
    def instantiate(cls, category: str, name: str, *args, **kwargs) -> Any:
        """
        Instantiate a particular model with its arguments and hyperparameters.

        Args:
            category (str): Category name of the required model.
            name (str): Name of the required model.
            *args: Positional arguments to be passed on to the model object.
            **kwargs: Key word arguments to be passed on to the model object.
        
        Returns:
            Instantiated model object.
        """
        return cls.get(category, name)(*args, **kwargs)

class TradModelLibrary:
    """
    Tradional Model Library to contain all available tradional portfolio 
    optimization models/methods like a library that can be queried.
    """
    _registry: Trad_Registry = {}
    _discovered_packages: Set[str] = set()

    @classmethod
    def register(cls, name: str = None):
        """
        Register a tradional portfolio optimization model/method using this decorator.

        Args:
            name (str | None): Name of the model. Default = None. 
                If None, name of the class will be used as default.
        
        Raise:
            KeyError: If a model already exists in the library.
        """
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

        Args:
            package (str): Name or pythonic module path to detect models.
                Example: 'src.models'
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
        """
        Get entire library of tradional portfolio optimization models/methods.

        Returns:
            Trad_Registry: Dictionary of all categories and their models.
        """
        return cls._registry

    @classmethod
    def list_models(cls) -> list[str]:
        """
        Get a list of all available models.        
        Returns:
            list[str]: List of all available tradional portfolio optimization models.
        """
        return list(cls._registry.keys())

    @classmethod
    def get(cls, name: str) -> Type | None:
        """
        Get a particular model by its name.

        Args:
            name (str): Name of the reqired tradional model.
        
        Returns:
            Type | None: Class of the required tradional model.    
        """
        return cls._registry.get(name)

    @classmethod
    def instantiate(cls, name: str, *args, **kwargs) -> Any:
        """
        Instantiate a particular model with its arguments and hyperparameters.

        Args:
            name (str): Name of the required model.
            *args: Positional arguments to be passed on to the model object.
            **kwargs: Key word arguments to be passed on to the model object.
        
        Returns:
            Instantiated model object.
        """
        return cls.get(name)(*args, **kwargs)