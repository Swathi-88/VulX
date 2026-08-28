import sys
from pathlib import Path

# Add project root and src/ directory to sys.path for test resolution
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
