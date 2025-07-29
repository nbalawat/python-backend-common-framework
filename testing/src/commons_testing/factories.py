"""Factory classes for test data generation."""

from typing import Any, Dict, Type, TypeVar, Optional, Callable
from unittest.mock import MagicMock
import factory
from factory import Factory as FactoryBoyFactory

T = TypeVar('T')

class Factory(FactoryBoyFactory):
    """Base factory class."""
    
    class Meta:
        abstract = True

class SubFactory(factory.SubFactory):
    """Sub-factory for related objects."""
    pass

class LazyAttribute(factory.LazyAttribute):
    """Lazy attribute that's computed when the object is built."""
    pass

class Sequence(factory.Sequence):
    """Sequence for generating unique values."""
    pass

class Faker(factory.Faker):
    """Faker for generating fake data."""
    pass

def create_mock_factory(cls: Type[T], **defaults) -> Callable[..., T]:
    """Create a factory that returns mock objects."""
    def _factory(**kwargs):
        mock_obj = MagicMock(spec=cls)
        # Set defaults
        for key, value in defaults.items():
            setattr(mock_obj, key, value)
        # Set provided kwargs
        for key, value in kwargs.items():
            setattr(mock_obj, key, value)
        return mock_obj
    return _factory