"""Makes the helpers in this directory importable under pytest's importlib import mode.

Nothing else belongs here: the tests are plain unittest.TestCase classes so that they run
under `python -m unittest` just as well as under pytest.
"""

import sys
from pathlib import Path

TESTS_DIR = str(Path(__file__).resolve().parent)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)
