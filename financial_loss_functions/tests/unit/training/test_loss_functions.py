import pytest
from src.training.loss_functions import LossLibrary

# ----------------------------------------------------------------------
# Fixture to populate registry with dummy functions
# ----------------------------------------------------------------------
@pytest.fixture
def populated_registry():
    # Clear any previous registrations
    LossLibrary._registry.clear()
    
    # Register dummy functions inside the fixture
    @LossLibrary.register(category='objectives', name='sharpe')
    def dummy_sharpe(weights, returns):
        return 0.5
    
    @LossLibrary.register(category='objectives', subcategory='risk', name='cvar')
    def dummy_cvar(weights, returns):
        return 0.2
    
    @LossLibrary.register(category='regularizers', name='risk_parity')
    def dummy_risk_parity(weights):
        return 0.1
    
    @LossLibrary.register(category='regularizers', subcategory='diversification', name='hhi')
    def dummy_hhi(weights):
        return 0.05
    
    yield LossLibrary._registry
    # Optionally clear after test (not necessary if we clear before each test in fixture)

# ----------------------------------------------------------------------
# Fixture to ensure empty registry
# ----------------------------------------------------------------------
@pytest.fixture
def empty_registry():
    LossLibrary._registry.clear()
    yield

# ----------------------------------------------------------------------
# Tests that require populated registry
# ----------------------------------------------------------------------
def test_register_and_get(populated_registry):
    # Get by category, name, default subcategory
    fn = LossLibrary.get('objectives', 'sharpe')
    assert fn is not None
    assert fn(1,2) == 0.5  # call to verify

    fn = LossLibrary.get('objectives', 'cvar', subcategory='risk')
    assert fn is not None
    assert fn(1,2) == 0.2

    fn = LossLibrary.get('regularizers', 'risk_parity')
    assert fn is not None
    assert fn(1) == 0.1

    fn = LossLibrary.get('regularizers', 'hhi', subcategory='diversification')
    assert fn is not None
    assert fn(1) == 0.05

def test_duplicate_registration_raises(populated_registry):
    with pytest.raises(KeyError, match="already registered"):
        @LossLibrary.register(category='objectives', name='sharpe')
        def another_sharpe():
            pass

def test_items(populated_registry):
    items = LossLibrary.items()
    assert 'objectives' in items
    assert 'regularizers' in items
    assert '__default__' in items['objectives']
    assert 'risk' in items['objectives']
    assert 'sharpe' in items['objectives']['__default__']
    assert 'cvar' in items['objectives']['risk']
    assert '__default__' in items['regularizers']
    assert 'diversification' in items['regularizers']
    assert 'risk_parity' in items['regularizers']['__default__']
    assert 'hhi' in items['regularizers']['diversification']

def test_list_categories(populated_registry):
    categories = LossLibrary.list_categories()
    assert set(categories) == {'objectives', 'regularizers'}

def test_list_subcategories(populated_registry):
    subcats = LossLibrary.list_subcategories('objectives')
    assert set(subcats) == {'__default__', 'risk'}
    subcats = LossLibrary.list_subcategories('regularizers')
    assert set(subcats) == {'__default__', 'diversification'}
    # Non-existent category returns empty list
    assert LossLibrary.list_subcategories('nonexistent') == []

def test_list_functions(populated_registry):
    funcs = LossLibrary.list_functions('objectives')
    assert set(funcs) == {'sharpe'}
    funcs = LossLibrary.list_functions('objectives', subcategory='risk')
    assert set(funcs) == {'cvar'}
    funcs = LossLibrary.list_functions('regularizers')
    assert set(funcs) == {'risk_parity'}
    funcs = LossLibrary.list_functions('regularizers', subcategory='diversification')
    assert set(funcs) == {'hhi'}
    # Non-existent category returns empty list
    assert LossLibrary.list_functions('nonexistent') == []
    # Non-existent subcategory returns empty list
    assert LossLibrary.list_functions('objectives', subcategory='missing') == []

# ----------------------------------------------------------------------
# Tests for empty registry (no functions)
# ----------------------------------------------------------------------
def test_empty_registry(empty_registry):
    assert LossLibrary.items() == {}
    assert LossLibrary.list_categories() == []
    assert LossLibrary.list_subcategories('objectives') == []
    assert LossLibrary.list_functions('objectives') == []
    with pytest.raises(KeyError):
        LossLibrary.get('objectives', 'sharpe')