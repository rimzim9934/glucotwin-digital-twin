"""Launch with the project-local Streamlit configuration from any directory."""
import os,sys
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)
os.execv(sys.executable,[sys.executable,'-m','streamlit','run','app.py',*sys.argv[1:]])
