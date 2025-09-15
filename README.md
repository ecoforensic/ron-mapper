# RoN-Mapper

**RoN-Mapper** is a Streamlit application for mapping species-level ecological evidence onto Ecuador’s Rights of Nature (RoN, Articles 71–74). It integrates global species distribution datasets, field surveys, and legal criteria to identify Number of species that trigger Rights of nature Articles, Constitutionally Imperilled Species (CIS) and Ambassador Species (CIAS) and other metrics to support decision-making under Ecuador’s constitutional framework.

---

## Quick Start

Clone or download this repository to your computer.

### Option 1 — One-click launcher (recommended)

**Windows**
1. Double-click `run_ron_mapper.bat`.  
2. The script will:
   - Create a virtual environment (`.venv`) if needed.
   - Install all required dependencies.
   - Launch the Streamlit app.  
3. The app will open automatically in your browser at [http://localhost:8501](http://localhost:8501).

**macOS / Linux**
1. Open a Terminal in this folder.  
2. Run:
   ```bash
   chmod +x run_ron_mapper.sh   # first time only
   ./run_ron_mapper.sh
   ```
3. The script will:
   - Create a virtual environment (`.venv`) if needed.
   - Install all required dependencies.
   - Launch the Streamlit app.  
4. The app will open automatically in your browser at [http://localhost:8501](http://localhost:8501).

---

### Option 2 — Manual setup (advanced users)

```bash
# 1) (optional) create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) install dependencies
pip install -r requirements.txt

# 3) run the app
streamlit run ron_streamlit_v1_0.py
```

---

## Profiles

Parameter profiles for scalars and weights are stored in the `profiles/` folder:

- `default_profile.json` – balanced defaults.  
- `precautionary_profile.json` – precautionary scalars (+5% IUCN, +10% other criteria, capped at 1.0).  

These can be loaded directly through the app sidebar.

---

## Troubleshooting

- **`streamlit` is not recognized** (Windows): activate the virtual environment, or use `run_ron_mapper.bat`.  
- **Permission denied** (macOS/Linux): run `chmod +x run_ron_mapper.sh` once, then `./run_ron_mapper.sh`.  
- **Python not found**: install Python 3.8+ from [python.org](https://www.python.org) and retry.  
- **Firewall prompt**: allow local network access so the browser can connect to the app.

---

## License

This repository is licensed under the **Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0)** licence.

- ✅ Free for research, academic, and non-commercial use.  
- ✅ You may copy, share, and adapt the code with proper attribution.  
- 🚫 Commercial use (including resale or integration into proprietary products) is prohibited without explicit permission.  

For commercial licensing enquiries, please contact **Kinseed Limited**.  
Full licence text: [LICENSE](LICENSE).

---

## Citation

If you use RoN-Mapper in your research, please cite:

```
Peck, M. (2025). RoN-Mapper (v1.0). Ecoforensic CIC.
Available at: https://github.com/<your-username>/ron-mapper
DOI: 10.5281/zenodo.TBD
```

The `CITATION.cff` file in this repository provides citation metadata for GitHub and reference managers.
