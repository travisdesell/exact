import pytest
# from unittest.mock import patch, MagicMock
from src.models.registry import NNModelLibrary, TradModelLibrary

# Dummy model classes for testing
class DummyModel1:
    pass

class DummyModel2:
    pass

# -------------------- Tests for NNModelLibrary -------------------- #
def test_nn_register_and_get():
    NNModelLibrary._registry.clear()
    
    @NNModelLibrary.register(category='test_cat', name='dummy1')
    class Dummy1:
        pass
    
    @NNModelLibrary.register(category='test_cat')
    class Dummy2:
        pass
    
    cls1 = NNModelLibrary.get('test_cat', 'dummy1')
    assert cls1 is Dummy1
    
    cls2 = NNModelLibrary.get('test_cat', 'Dummy2')
    assert cls2 is Dummy2

def test_nn_register_duplicate_raises():
    NNModelLibrary._registry.clear()
    
    @NNModelLibrary.register(category='test_cat', name='dup')
    class Dup1:
        pass
    
    with pytest.raises(KeyError, match="Model 'dup' already registered in category 'test_cat'"):
        @NNModelLibrary.register(category='test_cat', name='dup')
        class Dup2:
            pass

def test_nn_items_and_list_categories():
    NNModelLibrary._registry.clear()
    
    @NNModelLibrary.register(category='catA')
    class ModelA: pass
    
    @NNModelLibrary.register(category='catB')
    class ModelB: pass
    
    items = NNModelLibrary.items()
    assert 'catA' in items
    assert 'catB' in items
    assert items['catA']['ModelA'] is ModelA
    assert items['catB']['ModelB'] is ModelB
    
    categories = NNModelLibrary.list_categories()
    assert set(categories) == {'catA', 'catB'}

def test_nn_list_models():
    NNModelLibrary._registry.clear()
    
    @NNModelLibrary.register(category='catX', name='mod1')
    class Mod1: pass
    
    @NNModelLibrary.register(category='catX', name='mod2')
    class Mod2: pass
    
    models = NNModelLibrary.list_models('catX')
    assert models == ['mod1', 'mod2']
    
    assert NNModelLibrary.list_models('nonexistent') == []

def test_nn_get_nonexistent_returns_none():
    NNModelLibrary._registry.clear()
    # Category missing
    assert NNModelLibrary.get('cat', 'missing') is None
    # Category exists but model missing
    @NNModelLibrary.register(category='cat', name='exists')
    class Exists: pass
    assert NNModelLibrary.get('cat', 'missing') is None

def test_nn_instantiate():
    NNModelLibrary._registry.clear()
    
    @NNModelLibrary.register(category='test', name='MyModel')
    class MyModel:
        def __init__(self, a, b=10):
            self.a = a
            self.b = b
    
    obj = NNModelLibrary.instantiate('test', 'MyModel', 5, b=20)
    assert isinstance(obj, MyModel)
    assert obj.a == 5
    assert obj.b == 20

# @patch('pkgutil.walk_packages')
# @patch('importlib.import_module')
# def test_nn_autodiscover(mock_import_module, mock_walk_packages):
#     NNModelLibrary._discovered_packages.clear()
#     # Create a mock package with __path__ and __name__
#     mock_pkg = MagicMock()
#     mock_pkg.__path__ = ['/fake/path']
#     mock_pkg.__name__ = 'test.package'
#     mock_import_module.return_value = mock_pkg

#     # walk_packages must return full module names (including prefix)
#     mock_walk_packages.return_value = iter([(None, 'test.package.submod', False)])

#     # First call should import the package and walk
#     NNModelLibrary.autodiscover('test.package')
#     assert 'test.package' in NNModelLibrary._discovered_packages
#     mock_import_module.assert_called_with('test.package')
#     # Check that the submodule was imported
#     mock_import_module.assert_any_call('test.package.submod')
    
#     # Second call should skip
#     mock_import_module.reset_mock()
#     NNModelLibrary.autodiscover('test.package')
#     mock_import_module.assert_not_called()

# -------------------- Tests for TradModelLibrary -------------------- #
def test_trad_register_and_get():
    TradModelLibrary._registry.clear()
    
    @TradModelLibrary.register(name='trad1')
    class Trad1:
        pass
    
    @TradModelLibrary.register()
    class Trad2:
        pass
    
    cls1 = TradModelLibrary.get('trad1')
    assert cls1 is Trad1
    
    cls2 = TradModelLibrary.get('Trad2')
    assert cls2 is Trad2

def test_trad_register_duplicate_raises():
    TradModelLibrary._registry.clear()
    
    @TradModelLibrary.register(name='dup')
    class DupA:
        pass
    
    with pytest.raises(KeyError, match="Model 'dup' already registered in"):
        @TradModelLibrary.register(name='dup')
        class DupB:
            pass

def test_trad_items_and_list_models():
    TradModelLibrary._registry.clear()
    
    @TradModelLibrary.register(name='alpha')
    class Alpha: pass
    
    @TradModelLibrary.register(name='beta')
    class Beta: pass
    
    items = TradModelLibrary.items()
    assert items['alpha'] is Alpha
    assert items['beta'] is Beta
    
    models = TradModelLibrary.list_models()
    assert set(models) == {'alpha', 'beta'}

def test_trad_get_nonexistent_returns_none():
    TradModelLibrary._registry.clear()
    assert TradModelLibrary.get('missing') is None

def test_trad_instantiate():
    TradModelLibrary._registry.clear()
    
    @TradModelLibrary.register(name='MyTrad')
    class MyTrad:
        def __init__(self, x, y=100):
            self.x = x
            self.y = y
    
    obj = TradModelLibrary.instantiate('MyTrad', 42, y=200)
    assert isinstance(obj, MyTrad)
    assert obj.x == 42
    assert obj.y == 200

# @patch('pkgutil.walk_packages')
# @patch('importlib.import_module')
# def test_trad_autodiscover(mock_import_module, mock_walk_packages):
#     TradModelLibrary._discovered_packages.clear()
#     mock_pkg = MagicMock()
#     mock_pkg.__path__ = ['/fake/path']
#     mock_pkg.__name__ = 'trad.package'
#     mock_import_module.return_value = mock_pkg

#     mock_walk_packages.return_value = iter([(None, 'trad.package.submod', False)])

#     TradModelLibrary.autodiscover('trad.package')
#     assert 'trad.package' in TradModelLibrary._discovered_packages
#     mock_import_module.assert_called_with('trad.package')
#     mock_import_module.assert_any_call('trad.package.submod')
    
#     mock_import_module.reset_mock()
#     TradModelLibrary.autodiscover('trad.package')
#     mock_import_module.assert_not_called()