"""Keep historical RAHFL imports working in a checkout named PWHFL.

The existing Dataset, HHF and Network packages live one directory above this
compatibility namespace. No modules are eagerly imported here.
"""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent)]
