import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname (__file__), "."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Now you can import from src
from src import app
