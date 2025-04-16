"""
lfd/utils/constants.py \n
Constants, Paths
"""

import os

__root__ = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
with open(os.path.join(__root__, "VERSION"), "r") as f:
    __version__ = f.read().strip()


base_dir = os.path.join(__root__, "lfd")
exps_dir = os.path.join(base_dir, "experiments")

for dir_ in [exps_dir]:
    os.makedirs(dir_, exist_ok=True)
