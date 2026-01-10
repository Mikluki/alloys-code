"""
conftest.py - Pytest configuration
"""

import sys
from pathlib import Path

# Add project root to path so imports work
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    """Configure pytest before running tests"""
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test (requires mocks)"
    )
