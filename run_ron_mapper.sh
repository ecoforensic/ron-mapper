#!/usr/bin/env bash
set -euo pipefail

# === Change to the folder this script lives in ===
cd "$(dirname "$0")"

# === Find Python (prefer python3) ===
if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "ERROR: Python 3.8+ not found. Please install Python from https://www.python.org and retry."
  read -rp "Press Enter to exit..."
  exit 1
fi

# === Create a virtual environment if it doesn't exist ===
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  "$PYTHON" -m venv .venv
fi

# === Activate the venv ===
# shellcheck disable=SC1091
source .venv/bin/activate

# === Upgrade pip quietly ===
python -m pip install --upgrade pip --quiet

# === Install dependencies ===
if [ -f "requirements.txt" ]; then
  echo "Installing requirements from requirements.txt ..."
  pip install -r requirements.txt
else
  echo "requirements.txt not found; installing core packages..."
  pip install streamlit pandas numpy matplotlib plotly openpyxl XlsxWriter
fi

# === Run the app ===
echo "Launching Streamlit app..."
streamlit run ron_streamlit_v1_0.py

# === Keep window open if launched by double-click (some desktops) ===
echo ""
read -rp "Streamlit exited. Press Enter to close this window..."
