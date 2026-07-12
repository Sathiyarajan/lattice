import os
import sys

PROJECT_ROOT = os.path.expanduser("~/projects/lattice")
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
