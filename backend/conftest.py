"""Pytest configuration for backend tests.

Ensures the backend directory is importable regardless of the invocation
directory, and prevents the manual API smoke client from being collected as a
test module.
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

collect_ignore = ["test_client.py"]
