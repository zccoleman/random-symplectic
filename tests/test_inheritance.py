import pytest

from randomsymplectic import DnaryArray, SymplecticArray


def test_needs_d():
    with pytest.raises(TypeError):
        t = DnaryArray([0])

def test_can_create_child():
    class Child(DnaryArray):
        ...

def test_child_needs_d():
    class Child(DnaryArray):
        ...
    with pytest.raises(TypeError):
        Child([0])

def test_second_child_needs_d():
    class Child(DnaryArray):
        ...
    class Child2(Child):
        ...
    with pytest.raises(TypeError):
        Child2([0])

def test_cannot_reassign_classvar():
    class Child(DnaryArray):
        d=5
    with pytest.raises(AttributeError):
        Child.d=10

def test_cannot_reassign_instance_var():
    class Child(DnaryArray):
        d=5
    t=Child([0])
    with pytest.raises(AttributeError):
        t.d=10

def test_requires_prime():
    with pytest.raises(ValueError):
        class D4(DnaryArray):
            _validate_prime=True
            d=4
    
def test_dnary_identical_instances():
    D3 = DnaryArray.set_d(3)
    other_D3 = DnaryArray.set_d(3)
    assert D3 is other_D3

def test_symplectic_identical_instances():
    S3 = SymplecticArray.set_d(3)
    other_S3 = SymplecticArray.set_d(3)
    assert S3 is other_S3

def test_separate_instances():
    D3 = DnaryArray.set_d(89)
    assert 89 not in SymplecticArray._class_instances