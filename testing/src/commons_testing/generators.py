"""Data generators for testing."""

from typing import Any, Dict, List, Optional, Union, Type
from faker import Faker
import random
import string
from datetime import datetime, timedelta
from decimal import Decimal

# Global faker instance
fake = Faker()

class DataGenerator:
    """Utility class for generating test data."""
    
    def __init__(self, seed: Optional[int] = None):
        self.faker = Faker()
        if seed:
            self.faker.seed_instance(seed)
    
    def random_string(self, length: int = 10) -> str:
        """Generate random string."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    def random_email(self) -> str:
        """Generate random email."""
        return self.faker.email()
    
    def random_phone(self) -> str:
        """Generate random phone number."""
        return self.faker.phone_number()
    
    def random_date(self, start_date: datetime = None, end_date: datetime = None) -> datetime:
        """Generate random date."""
        if not start_date:
            start_date = datetime.now() - timedelta(days=365)
        if not end_date:
            end_date = datetime.now() + timedelta(days=365)
        return self.faker.date_time_between(start_date=start_date, end_date=end_date)
    
    def random_number(self, min_val: int = 0, max_val: int = 1000) -> int:
        """Generate random number."""
        return random.randint(min_val, max_val)
    
    def random_decimal(self, left_digits: int = 5, right_digits: int = 2) -> Decimal:
        """Generate random decimal."""
        return self.faker.pydecimal(left_digits=left_digits, right_digits=right_digits)
    
    def random_boolean(self) -> bool:
        """Generate random boolean."""
        return random.choice([True, False])
    
    def random_choice(self, choices: List[Any]) -> Any:
        """Pick random choice from list."""
        return random.choice(choices)
    
    def random_dict(self, keys: List[str], value_type: Type = str) -> Dict[str, Any]:
        """Generate random dictionary."""
        result = {}
        for key in keys:
            if value_type == str:
                result[key] = self.random_string()
            elif value_type == int:
                result[key] = self.random_number()
            elif value_type == bool:
                result[key] = self.random_boolean()
            else:
                result[key] = None
        return result