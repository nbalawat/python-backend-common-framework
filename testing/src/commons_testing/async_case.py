"""Async test case utilities."""

import asyncio
import pytest
from typing import Any, Callable, Optional
from unittest import TestCase

class AsyncTestCase(TestCase):
    """Base class for async test cases."""
    
    def setUp(self):
        """Set up test case."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Tear down test case."""
        if self.loop.is_running():
            self.loop.stop()
        self.loop.close()
    
    def run_async(self, coro):
        """Run an async coroutine in the test loop."""
        return self.loop.run_until_complete(coro)
    
    async def assert_async_raises(self, exception_type, coro):
        """Assert that an async coroutine raises an exception."""
        with self.assertRaises(exception_type):
            await coro

@pytest.mark.asyncio
async def async_test_fixture():
    """Basic async test fixture."""
    return {"status": "ready", "async": True}

def async_test(func: Callable) -> Callable:
    """Decorator to mark a test as async."""
    return pytest.mark.asyncio(func)