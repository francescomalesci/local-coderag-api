import os
import sys
import subprocess

# Prende la directory corrente di lavoro da cui viene lanciato lo script/exe
base_dir = os.path.dirname(sys.executable if getattr(sys, 'frozen', False) else os.path.abspath(__file__))

# Se l'exe è rimasto dentro 'dist', risale alla cartella di progetto
if os.path.basename(base_dir).lower() == "dist":
    base_dir = os.path.abspath(os.path.join(base_dir, ".."))

python_bin = os.path.join(base_dir, "venv", "Scripts", "python.exe")

if not os.path.exists(python_bin):
    # Fallback all'interprete di sistema se il venv non viene trovato
    python_bin = sys.executable

# Avvia FastAPI/Uvicorn
subprocess.Popen([python_bin, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"], cwd=base_dir)

# Avvia Streamlit
subprocess.Popen([python_bin, "-m", "streamlit", "run", "app.py"], cwd=base_dir)