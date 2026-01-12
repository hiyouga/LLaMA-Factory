"""Test package marker.

This file exists to make `tests_v1` importable when using multiprocessing with
the "spawn" start method (e.g., torch.multiprocessing), which requires test
modules to be importable by module path during pickling/unpickling.
"""


