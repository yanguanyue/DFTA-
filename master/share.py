import sys
from pathlib import Path

import config

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cldm.hack import disable_verbosity, enable_sliced_attention


disable_verbosity()

if config.save_memory:
    enable_sliced_attention()
