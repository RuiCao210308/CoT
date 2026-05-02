import os
import sys


PLANNER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openemma", "planner")
if PLANNER_DIR not in sys.path:
    sys.path.insert(0, PLANNER_DIR)

from action_chunk_sanity import main


if __name__ == "__main__":
    raise SystemExit(main())
