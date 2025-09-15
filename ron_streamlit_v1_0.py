# to run: streamlit run ron_streamlit_v1_0.py
# ===== Bootstrap: ensure required packages are available (one-time install if missing) =====
def _ensure_deps():
    import importlib, subprocess, sys
    required = {
        "streamlit": "streamlit",
        "pandas": "pandas",
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "plotly": "plotly",
        "openpyxl": "openpyxl",
        "xlsxwriter": "XlsxWriter",
    }
    for mod, pip_name in required.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

_ensure_deps()
# ===========================================================================================


import json
import copy
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="RoN Mapper — Ecoforensic CIC", layout="wide")
st.title("RoN Mapper — Ecoforensic CIC")
APP_VERSION = "v1.0"

# ----------------------
# Model configuration
# ----------------------
ARTICLES = ["Article_71","Article_72","Article_73","Article_74"]

# Per-trigger criteria (species-level)
CRITERIA = [
    "IUCN",
    "KBA_A1","KBA_B1","KBA_B2","KBA_D1","KBA_D2","KBA_D3","KBA_E",
    "EDGE","Keystone","Biocultural",
    "Endemism_National","Endemism_Regional"
]


DEFAULT_SCALARS = {
    "IUCN": 0.0,  # base if no category present
    "KBA_A1": 1.0,
    "KBA_B1": 0.60, "KBA_B2": 0.60,
    "KBA_D1": 0.60, "KBA_D2": 0.60, "KBA_D3": 0.60,
    "KBA_E":  0.60,
    "EDGE": 1.0, "Keystone": 1.0, "Biocultural": 1.0,
    "Endemism_National": 0.0, "Endemism_Regional": 0.0,
}

# IUCN Red List category → scalar (overrides base IUCN scalar of 0 when present)
IUCN_CATEGORY_SCALAR_DEFAULT = {
    "CR": 1.00,
    "EN": 0.85,
    "VU": 0.70,
    "NT": 0.30,
    "LC": 0.00,
    "DD": 0.50,
    "NE": 0.50,  # Not Evaluated → precautionary mid
    "NA": 0.00,  # Not Applicable
}

# Simple, binary default mapping (criterion → article weights)
DEFAULT_WEIGHTS_CRIT_TO_ART = {
    "IUCN": [0.0,1.0,1.0,0.0],     # Article 72 and 73
    "KBA_A1": [0.0,1.0,1.0,0.0],   # Article 72 and 73
    "KBA_B1": [1.0,1.0,0.0,0.0],   # Article 71 and 72
    "KBA_B2": [1.0,1.0,0.0,0.0],   # Article 71 and 72
    "KBA_D1": [1.0,1.0,0.5,0.0],   # Article 71, 72 and 73 (0.50)
    "KBA_D2": [1.0,1.0,0.5,0.0],   # Article 71, 72 and 73 (0.50)
    "KBA_D3": [1.0,1.0,0.5,0.0],   # Article 71, 72 and 73 (0.50)
    "KBA_E":  [1.0,1.0,0.5,0.0],   # Article 71, 72 and 73 (0.50)
    "EDGE": [1.0,1.0,1.0,0.0],     # Article 71, 72 and 73
    "Keystone": [1.0,0.0,0.0,0.0], # Article 71
    "Biocultural": [0.0,0.0,0.0,1.0], # Article 74
    "Endemism_National": [1.0,1.0,0.25,0.0], # Article 71, 72 and 73 (0.25)
    "Endemism_Regional": [1.0,1.0,0.5,0,0.0], # Article 71, 72 and 73 (0.5)
}

def crit_to_article_weights(crit_map: dict) -> dict:
    out = {a:{} for a in ARTICLES}
    for idx, art in enumerate(ARTICLES):
        for c, arr in crit_map.items():
            out[art][c] = float(arr[idx])
    return out

DEFAULT_WEIGHTS = crit_to_article_weights(DEFAULT_WEIGHTS_CRIT_TO_ART)

# ----------------------
# Session state
# ----------------------
if "weights" not in st.session_state:
    st.session_state["weights"] = {a: DEFAULT_WEIGHTS[a].copy() for a in ARTICLES}
if "scalars" not in st.session_state:
    st.session_state["scalars"] = DEFAULT_SCALARS.copy()
if "iucn_category_map" not in st.session_state:
    st.session_state["iucn_category_map"] = dict(IUCN_CATEGORY_SCALAR_DEFAULT)
if "combiner_mode" not in st.session_state:
    st.session_state["combiner_mode"] = "Evidence-union"

# ----------------------
# Sidebar controls
# ----------------------
st.sidebar.header("Inputs & Settings")
expert_mode = st.sidebar.checkbox("Show advanced options", value=False)

# ---- Profile presets (default Binary strict) ----
def get_profile_presets():
    # Base from current defaults (keeps Binary strict identical to defaults)
    base_w = {a: {c: float(w) for c, w in DEFAULT_WEIGHTS[a].items()} for a in ARTICLES}
    base_s = {k: float(v) for k, v in DEFAULT_SCALARS.items()}
    base_m = {k: float(v) for k, v in IUCN_CATEGORY_SCALAR_DEFAULT.items()}

    # Binary strict = defaults
    bin_w = {a: d.copy() for a, d in base_w.items()}
    bin_s = base_s.copy()
    bin_m = base_m.copy()

    # Balanced default = same as binary for now (workshop can tune later)
    bal_w = {a: d.copy() for a, d in base_w.items()}
    bal_s = base_s.copy()
    bal_m = base_m.copy()

    # Precautionary: +0.10 to scalars, +0.05 to IUCN map (cap at 1.0)
    pre_w = {a: d.copy() for a, d in base_w.items()}
    pre_s = {k: min(1.0, v + 0.10) for k, v in base_s.items()}
    pre_m = {k: min(1.0, v + 0.05) for k, v in base_m.items()}

    return {
        "Binary strict":  (bin_w, bin_s, bin_m),
        "Balanced default": (bal_w, bal_s, bal_m),
        "Precautionary": (pre_w, pre_s, pre_m),
    }

st.sidebar.markdown("### Profile presets")
_presets = get_profile_presets()
_preset_choice = st.sidebar.selectbox("Choose preset", list(_presets.keys()), index=list(_presets.keys()).index("Binary strict"))
if st.sidebar.button("Apply preset"):
    w,s,m = _presets[_preset_choice]
    st.session_state["weights"] = w
    st.session_state["scalars"] = s
    st.session_state["iucn_category_map"] = m
    st.success(f"Applied preset: {_preset_choice}")

# Aggregation selector (expert-only)
if expert_mode:
    st.sidebar.markdown("### Aggregation (expert only)")
    st.session_state["combiner_mode"] = st.sidebar.selectbox(
        "Combiner for contributions → article score",
        options=["Evidence-union","Weighted mean","Harmonic mean (conservative)"],
        index=["Evidence-union","Weighted mean","Harmonic mean (conservative)"].index(st.session_state["combiner_mode"])
    )

# Upload parameter profiles
up_w = st.sidebar.file_uploader("Upload custom weights JSON", type="json")
up_s = st.sidebar.file_uploader("Upload custom scalars JSON", type="json")
up_p = st.sidebar.file_uploader("Upload full profile (weights + scalars + IUCN map)", type="json")

if up_p is not None:
    try:
        prof = json.load(up_p)
        if "weights" in prof and "scalars" in prof:
            st.session_state["weights"] = prof["weights"]
            st.session_state["scalars"] = prof["scalars"]
            st.session_state["iucn_category_map"] = prof.get("iucn_category_map", st.session_state["iucn_category_map"])
            st.session_state["combiner_mode"] = prof.get("combiner_mode", st.session_state["combiner_mode"])
            st.success("Loaded profile.")
    except Exception as e:
        st.warning(f"Profile could not be loaded: {e}")

if up_w is not None:
    try:
        wobj = json.load(up_w)
        # Allow either criterion→[w71,w72,w73,w74] or article→criterion mapping
        if all(isinstance(v, (list,tuple)) for v in wobj.values()):
            st.session_state["weights"] = crit_to_article_weights(wobj)
        else:
            st.session_state["weights"] = wobj
        st.success("Loaded custom weights.")
    except Exception as e:
        st.warning(f"Weights JSON invalid: {e}")

if up_s is not None:
    try:
        sobj = json.load(up_s)
        st.session_state["scalars"].update(sobj)
        st.success("Loaded custom scalars.")
    except Exception as e:
        st.warning(f"Scalars JSON invalid: {e}")

# Download current profile
prof = {
    "weights": st.session_state["weights"],
    "scalars": st.session_state["scalars"],
    "iucn_category_map": st.session_state["iucn_category_map"],
    "combiner_mode": st.session_state["combiner_mode"],
    "app_version": APP_VERSION,
}
st.sidebar.download_button("Download current profile (JSON)",
    data=json.dumps(prof, indent=2), file_name="ron_profile_v7_2.json", mime="application/json"
)

# Thresholds
rli_threshold = st.sidebar.slider("RLI threshold (Article alert)", 0.0, 1.0, 0.6, 0.05)
species_alert_threshold = st.sidebar.slider("Species Article contribution alert", 0.0, 1.0, 0.6, 0.05)

# Advanced editors
if expert_mode:
    st.sidebar.markdown("### IUCN category → scalar")
    with st.sidebar.expander("Adjust IUCN category scalars (CR→LC, DD, NE, NA)", expanded=False):
        cmap = st.session_state["iucn_category_map"]
        for k in ["CR","EN","VU","NT","LC","DD","NE","NA"]:
            default_val = IUCN_CATEGORY_SCALAR_DEFAULT.get(k, 0.5)
            cmap[k] = st.sidebar.slider(f"{k}", 0.0, 1.0, float(cmap.get(k, default_val)), 0.05)
        st.sidebar.caption("These override the base IUCN scalar when a species has a Red List category.")

    st.sidebar.markdown("### Scalars")
    scalar_help = {
        "IUCN": "Used only if no category given; category mapping overrides.",
        "KBA_A1":"KBA A1 at site","KBA_B1":"KBA B1 restricted-range","KBA_B2":"KBA B2 co-occurring",
        "KBA_D1":"KBA D1 aggregations","KBA_D2":"KBA D2 refugia","KBA_D3":"KBA D3 recruitment","KBA_E":"KBA E quantitative",
        "EDGE":"Evolutionarily Distinct & Globally Endangered","Keystone":"Keystone species","Biocultural":"Biocultural value",
        "Endemism_National":"National endemic","Endemism_Regional":"Regional endemic",
    }
    for crit in CRITERIA:
        st.session_state["scalars"][crit] = st.sidebar.slider(
            f"{crit}", 0.0, 1.0, float(st.session_state["scalars"].get(crit, 0.5)), 0.05, help=scalar_help.get(crit,""))

    st.sidebar.markdown("### Weights (criterion → article)")
    with st.sidebar.expander("Adjust weights", expanded=False):
        for art in ARTICLES:
            st.write(f"**{art}**")
            for c in CRITERIA:
                st.session_state["weights"][art][c] = st.sidebar.slider(
                    f"{c} → {art.split('_')[1]}", 0.0, 1.0, float(st.session_state["weights"][art].get(c, 0.0)), 0.05
                )
        st.caption("Binary defaults; add small secondary weights with care to avoid double counting.")

# ----------------------
# Data ingest
# ----------------------
st.markdown("### Upload species data")
uploaded_file = st.file_uploader("Excel (.xlsx) with sheet 'Species_Input' (fallbacks to another sheet or first sheet)", type=["xlsx"])
work_df = None

if uploaded_file is not None:
    try:
        try:
            raw_df = pd.read_excel(uploaded_file, sheet_name="Species_Input")
        except Exception:
            try:
                raw_df = pd.read_excel(uploaded_file, sheet_name="cahuasqui_species_data")
            except Exception:
                raw_df = pd.read_excel(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(raw_df.head())

        st.markdown("### Column Mapper")
        expected_cols = [
            "Species",
            "IUCN_Category",
            "KBA_A1 (0/1)","KBA_B1 (0/1)","KBA_B2 (0/1)","KBA_D1 (0/1)","KBA_D2 (0/1)","KBA_D3 (0/1)","KBA_E (0/1)",
            "EDGE (0/1)","Keystone (0/1)","Biocultural (0/1)",
            "Endemism"
        ]
        mapper_cols = {}
        with st.expander("Map your columns (only if your headers differ)"):
            for col in expected_cols:
                options = ["(none)"] + list(raw_df.columns)
                default_idx = options.index(col) if col in raw_df.columns else 0
                mapper_cols[col] = st.selectbox(f"Map to '{col}'", options=options, index=default_idx, key=f"map_{col}")

        def _pick(colname):
            mapped = mapper_cols.get(colname, colname)
            if mapped and mapped != "(none)" and mapped in raw_df.columns:
                return raw_df[mapped]
            return pd.Series([np.nan]*len(raw_df), index=raw_df.index)

        work_df = pd.DataFrame({c: _pick(c) for c in expected_cols})

    except Exception as e:
        st.error(f"Could not read species sheet: {e}")

# ----------------------
# Helpers
# ----------------------
def safe_int01(v):
    try:
        return 1 if int(float(v)) >= 1 else 0
    except Exception:
        return 0

def parse_endemism(val):
    txt = str(val) if not pd.isna(val) else ""
    txt = txt.lower()
    en_nat = 1 if "national" in txt else 0
    en_reg = 1 if "regional" in txt or "local" in txt else 0
    known = bool(txt)
    return en_nat, en_reg, known

def extract_triggers_and_known(row: pd.Series):
    kba_a1 = 'KBA_A1 (0/1)'
    kba_b_cols = ['KBA_B1 (0/1)','KBA_B2 (0/1)']
    kba_d_cols = ['KBA_D1 (0/1)','KBA_D2 (0/1)','KBA_D3 (0/1)']
    kba_e_col = 'KBA_E (0/1)'

    IUCN_category = str(row.get('IUCN_Category', '')).strip().upper() if not pd.isna(row.get('IUCN_Category', np.nan)) else ''
    IUCN = 1 if IUCN_category in st.session_state["iucn_category_map"] else 0

    A1 = safe_int01(row.get(kba_a1, np.nan))
    B1 = safe_int01(row.get(kba_b_cols[0], np.nan))
    B2 = safe_int01(row.get(kba_b_cols[1], np.nan))
    D1 = safe_int01(row.get(kba_d_cols[0], np.nan))
    D2 = safe_int01(row.get(kba_d_cols[1], np.nan))
    D3 = safe_int01(row.get(kba_d_cols[2], np.nan))
    E  = safe_int01(row.get(kba_e_col, np.nan))

    EDGE = safe_int01(row.get('EDGE (0/1)', np.nan))
    Keystone = safe_int01(row.get('Keystone (0/1)', np.nan))
    Biocultural = safe_int01(row.get('Biocultural (0/1)', np.nan))
    EnNat, EnReg, endem_known = parse_endemism(row.get('Endemism', np.nan))

    triggers = {
        "IUCN": IUCN, "IUCN_Category": IUCN_category,
        "KBA_A1": A1, "KBA_B1": B1, "KBA_B2": B2,
        "KBA_D1": D1, "KBA_D2": D2, "KBA_D3": D3,
        "KBA_E": E,
        "EDGE": EDGE, "Keystone": Keystone, "Biocultural": Biocultural,
        "Endemism_National": EnNat, "Endemism_Regional": EnReg
    }
    known = {
        "IUCN": bool(IUCN_category),
        "KBA_A1": not pd.isna(row.get(kba_a1, np.nan)),
        "KBA_B1": not pd.isna(row.get(kba_b_cols[0], np.nan)),
        "KBA_B2": not pd.isna(row.get(kba_b_cols[1], np.nan)),
        "KBA_D1": not pd.isna(row.get(kba_d_cols[0], np.nan)),
        "KBA_D2": not pd.isna(row.get(kba_d_cols[1], np.nan)),
        "KBA_D3": not pd.isna(row.get(kba_d_cols[2], np.nan)),
        "KBA_E":  not pd.isna(row.get(kba_e_col, np.nan)),
        "EDGE": not pd.isna(row.get('EDGE (0/1)', np.nan)),
        "Keystone": not pd.isna(row.get('Keystone (0/1)', np.nan)),
        "Biocultural": not pd.isna(row.get('Biocultural (0/1)', np.nan)),
        "Endemism_National": endem_known, "Endemism_Regional": endem_known,
    }
    return triggers, known

def effective_scalars_for_species(triggers: dict, base_scalars: dict):
    scal = base_scalars.copy()
    cat = str(triggers.get("IUCN_Category","")).upper()
    if cat in st.session_state.get("iucn_category_map", IUCN_CATEGORY_SCALAR_DEFAULT):
        scal["IUCN"] = float(st.session_state["iucn_category_map"][cat])
    return scal

def compute_article_scores(triggers: dict, weights: dict, scalars: dict, combiner_mode: str):
    scores = {}
    for art in ARTICLES:
        v = np.array([triggers[c] * scalars[c] for c in CRITERIA], dtype=float)
        w = np.array([weights[art][c] for c in CRITERIA], dtype=float)
        x = v * w  # per-criterion contributions
        mode = combiner_mode
        if mode == "Weighted mean":
            denom = w.sum() + 1e-12
            scores[art] = float((w * v).sum() / denom)
        elif mode == "Harmonic mean (conservative)":
            mask = v > 0
            if not np.any(mask):
                scores[art] = 0.0
            else:
                w_pos = w[mask]
                v_pos = v[mask]
                scores[art] = float(w_pos.sum() / ((w_pos / np.clip(v_pos, 1e-9, 1.0)).sum()))
        else:  # Evidence-union
            scores[art] = float(1.0 - np.prod(1.0 - x))
    return scores

def compute_eci(known: dict, weights_by_art: dict):
    eci = {}
    for art in ARTICLES:
        crits = list(weights_by_art[art].keys())
        known_w = [weights_by_art[art][c] for c in crits if known.get(c, False)]
        tot_w = [weights_by_art[art][c] for c in crits]
        eci[art] = float(sum(known_w)) / (float(sum(tot_w)) + 1e-9) if tot_w else 0.0
    eci["ECI_overall"] = float(np.mean([eci[a] for a in ARTICLES]))
    return eci

def compute_scores(triggers: dict, rli_thr: float):
    scal_used = effective_scalars_for_species(triggers, st.session_state["scalars"])
    scores = compute_article_scores(triggers, st.session_state["weights"], scal_used, st.session_state.get("combiner_mode","Evidence-union"))
    SRRS = 100 * float(np.mean([scores[a] for a in ARTICLES]))
    RLI = int(sum(1 for a in ARTICLES if scores[a] >= rli_thr))
    return scores, SRRS, RLI

@st.cache_data(show_spinner=False)
def compute_results_cached(work_df: pd.DataFrame, weights: dict, scalars: dict,
                           iucn_map: dict, rli_thr: float, combiner_mode: str):
    rows = []
    for _, row in work_df.iterrows():
        species = str(row.get("Species","Unknown"))
        triggers, known = extract_triggers_and_known(row)
        scal_used = effective_scalars_for_species(triggers, scalars)
        scores = compute_article_scores(triggers, weights, scal_used, combiner_mode)
        srrs = 100 * float(np.mean([scores[a] for a in ARTICLES]))
        rli = int(sum(1 for a in ARTICLES if scores[a] >= rli_thr))
        eci = {}
        for art in ARTICLES:
            crits = list(weights[art].keys())
            known_w = [weights[art][c] for c in crits if known.get(c, False)]
            tot_w = [weights[art][c] for c in crits]
            eci[art] = float(sum(known_w)) / (float(sum(tot_w)) + 1e-9) if tot_w else 0.0
        rec = {"Species": species, **scores, "SRRS": srrs, "RLI": rli}
        for a in ARTICLES:
            rec[f"ECI_{a.split('_')[1]}"] = eci[a]
        rec["ECI_overall"] = float(np.mean([eci[a] for a in ARTICLES]))
        rows.append(rec)
    return pd.DataFrame(rows)

# ----------------------
# UI Tabs
# ----------------------
tab1, tab2, tab3, tab4 = st.tabs(["Results","Parameters","Sensitivity (Tornado)","Compare Profiles"])

with tab1:
    st.subheader("Species results")
    if uploaded_file is None or work_df is None:
        st.info("Upload a dataset to compute results.")
    else:
        # Cached computation of results
        results_df = compute_results_cached(
            work_df,
            st.session_state["weights"],
            st.session_state["scalars"],
            st.session_state["iucn_category_map"],
            rli_threshold,
            st.session_state.get("combiner_mode", "Evidence-union")
        ).sort_values("Species").reset_index(drop=True)

        # Quick metrics
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Species (n)", len(results_df))
        with c2:
            st.metric("Mean SRRS", f"{results_df['SRRS'].mean():.1f}")
        with c3:
            st.metric("Mean RLI (0–4)", f"{results_df['RLI'].mean():.2f}")

        st.dataframe(results_df)

        # Unique species triggering at least one Article at or above threshold
        triggered_any = (results_df[ARTICLES] >= rli_threshold).any(axis=1)
        total_triggered_any = int(triggered_any.sum())
        st.markdown(f"**Species with ≥1 Article ≥ threshold:** {total_triggered_any} / {len(results_df)}")

        # Species counts by Article threshold
        st.markdown("### Species counts by Article (≥ threshold)")
        counts = {a: int((results_df[a] >= rli_threshold).sum()) for a in ARTICLES}
        fig, ax = plt.subplots(figsize=(5,3))
        ax.bar(list(counts.keys()), list(counts.values()))
        for i, (k,v) in enumerate(counts.items()):
            ax.text(i, v + (max(counts.values()) if counts else 1)*0.02, str(v), ha='center', va='bottom', fontsize=9)
        ax.set_ylim(0, (max(counts.values()) if counts else 1) * 1.25)
        ax.set_ylabel("Species")
        ax.set_xticklabels(list(counts.keys()), rotation=90, ha='right')
        st.pyplot(fig, clear_figure=True)

        # Threshold sweep: threshold vs % species triggering by Article
        st.markdown("### Threshold sweep (governance sensitivity)")
        thr_grid = np.linspace(0, 1, 21)
        pct = {a: [] for a in ARTICLES}
        for t in thr_grid:
            mask = results_df[ARTICLES] >= t
            denom = max(1, len(results_df))
            for a in ARTICLES:
                pct[a].append(100.0 * float(mask[a].sum()) / denom)
        fig_sw, ax_sw = plt.subplots(figsize=(6, 3.2))
        for a in ARTICLES:
            ax_sw.plot(thr_grid, pct[a], label=a)
        ax_sw.axvline(rli_threshold, linestyle="--", linewidth=1)
        ax_sw.set_xlabel("Threshold")
        ax_sw.set_ylabel("% species ≥ threshold")
        ax_sw.set_ylim(0, 100)
        ax_sw.legend(ncol=2, fontsize=8)
        st.pyplot(fig_sw, clear_figure=True)

        # Optional overlay with comparison profile
        with st.expander("Overlay comparison profile on sweep"):
            mode = st.radio("Comparison source", ["None","Upload JSON","Preset"], horizontal=True)
            comp = None
            if mode == "Upload JSON":
                upC = st.file_uploader("Upload comparison profile JSON", type="json", key="sweep_prof")
                if upC is not None:
                    try:
                        profC = json.load(upC)
                        comp = (
                            profC.get("weights", st.session_state["weights"]),
                            profC.get("scalars", st.session_state["scalars"]),
                            profC.get("iucn_category_map", st.session_state["iucn_category_map"]),
                            profC.get("combiner_mode", st.session_state.get("combiner_mode","Evidence-union"))
                        )
                    except Exception as e:
                        st.warning(f"Could not read comparison profile: {e}")
            elif mode == "Preset":
                presets_loc = get_profile_presets()
                psel = st.selectbox("Preset to overlay", list(presets_loc.keys()), index=list(presets_loc.keys()).index("Balanced default"))
                wC, sC, mC = presets_loc[psel]
                comp = (wC, sC, mC, st.session_state.get("combiner_mode","Evidence-union"))

            if comp is not None:
                wC, sC, mC, combC = comp
                dfC = compute_results_cached(work_df, wC, sC, mC, rli_threshold, combC)
                pctC = {a: [] for a in ARTICLES}
                for t in thr_grid:
                    maskC = dfC[ARTICLES] >= t
                    denomC = max(1, len(dfC))
                    for a in ARTICLES:
                        pctC[a].append(100.0 * float(maskC[a].sum()) / denomC)
                fig_sw2, ax_sw2 = plt.subplots(figsize=(6, 3.2))
                for a in ARTICLES:
                    ax_sw2.plot(thr_grid, pct[a], label=f"{a} (A)")
                for a in ARTICLES:
                    ax_sw2.plot(thr_grid, pctC[a], linestyle="--", label=f"{a} (B)")
                ax_sw2.axvline(rli_threshold, linestyle="--", linewidth=1)
                ax_sw2.set_xlabel("Threshold")
                ax_sw2.set_ylabel("% species ≥ threshold")
                ax_sw2.set_ylim(0, 100)
                ax_sw2.legend(ncol=2, fontsize=8)
                st.pyplot(fig_sw2, clear_figure=True)

        # --- Visual summaries: SRRS (sorted), ECI, RLI ---
        st.markdown("### Visual summaries")
        max_n = min(30, len(results_df))
        vis_n = st.slider("Number of species to display (top by SRRS)", 5, max_n, min(25, max_n), 1)
        # Order species by SRRS descending and take top N
        srrs_sorted = results_df.sort_values("SRRS", ascending=False).head(vis_n)
        species_order = list(srrs_sorted["Species"])
        # Dynamic figure width and font size
        fig_w = max(6, 0.35 * len(species_order))
        font_sz = max(7, min(12, int(220 / max(1, len(species_order)))))

        # SRRS bar chart
        fig_s, ax_s = plt.subplots(figsize=(fig_w, 3.5))
        ax_s.bar(species_order, srrs_sorted["SRRS"].values)
        ax_s.set_ylabel("SRRS")
        ax_s.set_title("SRRS (top by SRRS)")
        ax_s.set_xticklabels(species_order, rotation=90, ha="right")
        ax_s.tick_params(axis="x", labelsize=font_sz)
        ax_s.tick_params(axis="y", labelsize=10)
        st.pyplot(fig_s, clear_figure=True)

        # ECI (overall) bar chart using same species ordering
        eci_vals = results_df.set_index("Species").loc[species_order, "ECI_overall"].values
        fig_e, ax_e = plt.subplots(figsize=(fig_w, 3.5))
        ax_e.bar(species_order, eci_vals)
        ax_e.set_ylabel("ECI (overall)")
        ax_e.set_title("Evidence Completeness Index (same species order as SRRS)")
        ax_e.set_xticklabels(species_order, rotation=90, ha="right")
        ax_e.tick_params(axis="x", labelsize=font_sz)
        ax_e.tick_params(axis="y", labelsize=10)
        st.pyplot(fig_e, clear_figure=True)

        # RLI bar chart using same species ordering
        rli_vals = results_df.set_index("Species").loc[species_order, "RLI"].values
        fig_r, ax_r = plt.subplots(figsize=(fig_w, 3.5))
        ax_r.bar(species_order, rli_vals)
        ax_r.set_ylabel("RLI (0–4)")
        ax_r.set_title("RoN Linkage Index (same species order as SRRS)")
        ax_r.set_xticklabels(species_order, rotation=90, ha="right")
        ax_r.tick_params(axis="x", labelsize=font_sz)
        ax_r.tick_params(axis="y", labelsize=10)
        st.pyplot(fig_r, clear_figure=True)

        # --- Species audit (selected species): per-article scores + stacked contributions ---
        st.markdown("### Species audit")
        species_sel = st.selectbox("Pick a species to audit", options=list(results_df["Species"]))
        row_sel = work_df[work_df["Species"].astype(str) == str(species_sel)].head(1)
        if len(row_sel):
            tr_sel, _ = extract_triggers_and_known(row_sel.iloc[0])
            scal_sel = effective_scalars_for_species(tr_sel, st.session_state["scalars"])
            scores_sel = compute_article_scores(tr_sel, st.session_state["weights"], scal_sel, st.session_state.get("combiner_mode", "Evidence-union"))

            # Per-article score bars with threshold line + hover (top contributors)
            try:
                import plotly.graph_objects as go
                arts = ARTICLES
                vals = [scores_sel[a] for a in arts]
                hover = []
                for art in arts:
                    comps = []
                    for c in CRITERIA:
                        v = tr_sel[c] * scal_sel[c] * st.session_state["weights"][art][c]
                        if v > 0:
                            comps.append((c, v))
                    comps.sort(key=lambda x: x[1], reverse=True)
                    top = comps[:5]
                    txt = "<br>".join([f"{c}: {v:.3f}" for c,v in top]) if top else "No contributors"
                    hover.append(txt)
                figas = go.Figure(go.Bar(x=arts, y=vals, hovertext=hover, hoverinfo="text"))
                figas.add_hline(y=rli_threshold, line_dash="dash")
                figas.update_yaxes(range=[0,1], title="Article score (0–1)")
                figas.update_layout(title=f"Per-article scores — {species_sel}")
                st.plotly_chart(figas, use_container_width=True)
            except Exception:
                fig_as, ax_as = plt.subplots(figsize=(5,3))
                arts = ARTICLES
                vals = [scores_sel[a] for a in arts]
                ax_as.bar(arts, vals)
                ax_as.axhline(rli_threshold, linestyle="--", linewidth=1)
                ax_as.set_ylim(0, 1)
                ax_as.set_ylabel("Article score (0–1)")
                ax_as.set_title(f"Per-article scores — {species_sel}")
                ax_as.set_xticklabels(arts, rotation=90, ha="right")
                st.pyplot(fig_as, clear_figure=True)

            # Stacked contributions with hover tooltips (Plotly; fallback to matplotlib)
            try:
                import plotly.graph_objects as go
                arts = ARTICLES
                crits = CRITERIA
                contrib = {c: [] for c in crits}
                hover = {c: [] for c in crits}
                for art in arts:
                    for c in crits:
                        t = tr_sel[c]
                        s = scal_sel[c]
                        w = st.session_state["weights"][art][c]
                        val = t * s * w
                        contrib[c].append(val)
                        hover[c].append(f"{c}: {val:.3f} = {t}×{s:.3f}×{w:.3f}")
                figp = go.Figure()
                for c in crits:
                    figp.add_trace(go.Bar(x=contrib[c], y=arts, orientation="h", name=c, hovertext=hover[c], hoverinfo="text"))
                figp.update_layout(barmode="stack", height=max(300, 60*len(arts)), title=f"Article contribution breakdown — {species_sel}")
                figp.update_xaxes(title="∑(trigger×scalar×weight)", range=[0,1])
                figp.update_yaxes(title="")
                st.plotly_chart(figp, use_container_width=True)
            except Exception:
                crits = CRITERIA
                fig_sc, ax_sc = plt.subplots(figsize=(7, max(3.5, 0.6*len(ARTICLES))))
                y_pos = np.arange(len(ARTICLES))
                left = np.zeros(len(ARTICLES))
                contrib = np.zeros((len(ARTICLES), len(crits)))
                for ai, art in enumerate(ARTICLES):
                    for ci, c in enumerate(crits):
                        v = tr_sel[c] * scal_sel[c]
                        w = st.session_state["weights"][art][c]
                        contrib[ai, ci] = v * w
                total_by_crit = contrib.sum(axis=0)
                top_idx = np.argsort(total_by_crit)[::-1]
                show_idx = top_idx[:10]
                for ci, c in enumerate(crits):
                    vals_c = contrib[:, ci]
                    ax_sc.barh(y_pos, vals_c, left=left, label=c if ci in show_idx else None)
                    left += vals_c
                ax_sc.set_yticks(y_pos)
                ax_sc.set_yticklabels([a for a in ARTICLES])
                ax_sc.set_xlabel("∑ (trigger × scalar × weight)")
                ax_sc.set_title(f"Article contribution breakdown — {species_sel}")
                ax_sc.set_xlim(0, 1)
                if any(ci in show_idx for ci in range(len(crits))):
                    ax_sc.legend(ncol=2, fontsize=8)
                st.pyplot(fig_sc, clear_figure=True)

        # Download results with provenance
        out = BytesIO()
        with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
            results_df.to_excel(writer, index=False, sheet_name="Results")
            # Provenance (hidden)
            prov = {
                "app_version": APP_VERSION,
                "threshold": rli_threshold,
                "combiner_mode": st.session_state.get("combiner_mode","Evidence-union"),
                "profile": {"weights": st.session_state["weights"], "scalars": st.session_state["scalars"], "iucn_category_map": st.session_state["iucn_category_map"]}
            }
            prov_df = pd.DataFrame({"Provenance":[json.dumps(prov, indent=2)]})
            prov_df.to_excel(writer, index=False, sheet_name="Provenance")
            try:
                workbook  = writer.book
                worksheet = writer.sheets["Provenance"]
                worksheet.hide()
            except Exception:
                pass
        st.download_button("Download results (Excel)", data=out.getvalue(), file_name="ron_results.xlsx")

with tab2:
    st.subheader("Active parameters (current run)")
    st.write("**IUCN category mapping**")
    st.json(st.session_state["iucn_category_map"])
    st.write("**Scalars**")
    st.json(st.session_state["scalars"])
    st.write("**Weights (criterion → article)**")
    st.json(st.session_state["weights"])

    st.markdown("### Validation checks")
    st.caption("Checks based on **current combiner** and threshold.")
    def _article_score_for(cat):
        tr = {k:0 for k in CRITERIA}
        tr["IUCN"] = 1
        tr["IUCN_Category"] = cat
        scal = effective_scalars_for_species(tr, st.session_state["scalars"])
        s = compute_article_scores(tr, st.session_state["weights"], scal, st.session_state.get("combiner_mode","Evidence-union"))
        return s["Article_73"]
    vu_ok = _article_score_for("VU") >= rli_threshold
    en_ok = _article_score_for("EN") >= rli_threshold
    cr_ok = _article_score_for("CR") >= rli_threshold
    st.write(f"IUCN VU triggers A73 ≥ threshold: **{vu_ok}**")
    st.write(f"IUCN EN triggers A73 ≥ threshold: **{en_ok}**")
    st.write(f"IUCN CR triggers A73 ≥ threshold: **{cr_ok}**")
    trE = {k:0 for k in CRITERIA}
    trE["EDGE"] = 1
    scalE = effective_scalars_for_species(trE, st.session_state["scalars"])
    sE = compute_article_scores(trE, st.session_state["weights"], scalE, st.session_state.get("combiner_mode","Evidence-union"))
    e71_ok = sE["Article_71"] >= rli_threshold
    e73_ok = sE["Article_73"] >= rli_threshold
    st.write(f"EDGE triggers A71 ≥ threshold: **{e71_ok}**")
    st.write(f"EDGE triggers A73 ≥ threshold: **{e73_ok}**")
    if not (vu_ok and en_ok and cr_ok and e71_ok and e73_ok):
        st.warning("One or more guarantees fail under the current combiner/threshold. Consider Evidence-union and/or adjust scalars/weights.")

with tab3:
    st.subheader("One-at-a-time (OAT) Tornado Sensitivity")
    if uploaded_file is None or work_df is None:
        st.info("Upload a dataset to run sensitivity analysis.")
    else:
        # Controls
        scope = st.radio("Scope", ["Profile (global)", "Species"], horizontal=True)
        if scope == "Species":
            species_list = list(pd.Series(work_df["Species"]).astype(str).fillna("Unknown").unique())
            species_name = st.selectbox("Select species", options=species_list)
        else:
            species_name = None
        outcome_opt = st.selectbox("Outcome",
            options=["SRRS (mean)","Article_71 (mean)","Article_72 (mean)","Article_73 (mean)","Article_74 (mean)"] if scope=="Profile (global)"
                    else ["SRRS (this species)","Article_71 (this species)","Article_72 (this species)","Article_73 (this species)","Article_74 (this species)"])
        low_bound = st.slider("Low bound for perturbation", 0.0, 1.0, 0.0, 0.05)
        high_bound = st.slider("High bound for perturbation", 0.0, 1.0, 1.0, 0.05)
        topn = st.slider("Show top N most influential", 5, 40, 20, 1)
        include_scalars = st.checkbox("Include scalars", value=True)
        include_weights = st.checkbox("Include weights", value=True)

        def parse_outcome_key(txt):
            if txt.startswith("SRRS"):
                return "SRRS"
            else:
                return txt.split()[0]

        outcome_key = parse_outcome_key(outcome_opt)

        def _compute_species_outcome(row, weights, scalars, outcome_key):
            triggers, known = extract_triggers_and_known(row)
            scal_used = effective_scalars_for_species(triggers, scalars)
            scores = compute_article_scores(triggers, weights, scal_used, st.session_state.get("combiner_mode","Evidence-union"))
            if outcome_key == "SRRS":
                return 100 * float(np.mean([scores[a] for a in ARTICLES]))
            else:
                return float(scores[outcome_key])

        def compute_outcome(weights, scalars, scope, species_name, outcome_key):
            if scope == "Species":
                row = work_df[work_df["Species"].astype(str) == str(species_name)].head(1)
                if len(row) == 0:
                    return np.nan
                return _compute_species_outcome(row.iloc[0], weights, scalars, outcome_key)
            else:
                vals = []
                for _, r in work_df.iterrows():
                    vals.append(_compute_species_outcome(r, weights, scalars, outcome_key))
                return float(np.nanmean(vals)) if len(vals) else np.nan

        # Baseline
        base_val = compute_outcome(st.session_state["weights"], st.session_state["scalars"], scope, species_name, outcome_key)

        # Build parameter list
        params = []
        if include_scalars:
            for c in CRITERIA:
                params.append(("SCALAR", c, None))
        if include_weights:
            for art in ARTICLES:
                for c in CRITERIA:
                    params.append(("WEIGHT", c, art))

        records = []
        for ptype, crit, art in params:
            w_lo = copy.deepcopy(st.session_state["weights"])
            w_hi = copy.deepcopy(st.session_state["weights"])
            s_lo = st.session_state["scalars"].copy()
            s_hi = st.session_state["scalars"].copy()

            if ptype == "SCALAR":
                s_lo[crit] = float(low_bound)
                s_hi[crit] = float(high_bound)
            else:  # WEIGHT
                w_lo[art][crit] = float(low_bound)
                w_hi[art][crit] = float(high_bound)

            v_lo = compute_outcome(w_lo, s_lo, scope, species_name, outcome_key)
            v_hi = compute_outcome(w_hi, s_hi, scope, species_name, outcome_key)
            d_lo = float(v_lo) - float(base_val)
            d_hi = float(v_hi) - float(base_val)
            records.append({
                "param": f"{ptype}: {crit}" if ptype=="SCALAR" else f"{ptype}: {crit}→{art.split('_')[1]}",
                "low_delta": d_lo, "high_delta": d_hi,
                "abs_max": max(abs(d_lo), abs(d_hi))
            })

        tdf = pd.DataFrame(records).sort_values("abs_max", ascending=False).head(topn)

        st.markdown("#### Tornado chart")
        if tdf.empty:
            st.info("Nothing to show. Enable scalars/weights and ensure data is uploaded.")
        else:
            fig_t, ax_t = plt.subplots(figsize=(8, max(4, 0.35*len(tdf))))
            y = np.arange(len(tdf))
            ax_t.barh(y, tdf["low_delta"], label="Low bound")
            ax_t.barh(y, tdf["high_delta"], left=0, label="High bound")
            ax_t.axvline(0, linestyle="--")
            ax_t.set_yticks(y)
            ax_t.set_yticklabels(tdf["param"])
            ax_t.set_xlabel("Change in outcome (vs baseline)")
            ax_t.invert_yaxis()
            ax_t.legend()
            st.pyplot(fig_t, clear_figure=True)

            st.dataframe(tdf[["param","low_delta","high_delta","abs_max"]])
            st.download_button("Download tornado data (CSV)",
                               data=tdf.to_csv(index=False),
                               file_name="tornado_oat.csv",
                               mime="text/csv")

with tab4:
    st.subheader("Compare profiles (A vs B)")
    if uploaded_file is None or work_df is None:
        st.info("Upload a dataset first to compare profiles.")
    else:
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Profile A**")
            use_current_A = st.checkbox("Use current profile as A", value=True)
            upA = st.file_uploader("Or upload profile A (JSON)", type="json", key="profA")
        with colB:
            st.markdown("**Profile B**")
            use_current_B = st.checkbox("Use current profile as B", value=False)
            upB = st.file_uploader("Or upload profile B (JSON)", type="json", key="profB")

        def _load_profile(use_current, uploaded):
            if use_current:
                return (
                    st.session_state["weights"],
                    st.session_state["scalars"],
                    st.session_state["iucn_category_map"],
                    st.session_state.get("combiner_mode","Evidence-union")
                )
            if uploaded is not None:
                try:
                    prof = json.load(uploaded)
                    w = prof["weights"]
                    s = prof["scalars"]
                    m = prof.get("iucn_category_map", st.session_state["iucn_category_map"])
                    comb = prof.get("combiner_mode", st.session_state.get("combiner_mode","Evidence-union"))
                    return (w, s, m, comb)
                except Exception as e:
                    st.warning(f"Could not load profile: {e}")
            return None

        profA = _load_profile(use_current_A, upA)
        profB = _load_profile(use_current_B, upB)

        if not profA or not profB:
            st.info("Select two profiles to compare.")
        else:
            wA, sA, mA, combA = profA
            wB, sB, mB, combB = profB
            dfA = compute_results_cached(work_df, wA, sA, mA, rli_threshold, combA)
            dfB = compute_results_cached(work_df, wB, sB, mB, rli_threshold, combB)
            def _summary(df):
                return {
                    "Mean_SRRS": float(df["SRRS"].mean()),
                    "Mean_RLI": float(df["RLI"].mean()),
                    **{f"Mean_{a}": float(df[a].mean()) for a in ARTICLES}
                }
            sumA, sumB = _summary(dfA), _summary(dfB)
            st.markdown("#### Summary (means)")
            st.write(pd.DataFrame([sumA, sumB], index=["Profile A","Profile B"]))
            st.markdown("#### Δ (Profile B − Profile A)")
            delta = {k: sumB[k]-sumA[k] for k in sumA.keys()}
            st.write(pd.DataFrame([delta], index=["Δ"]))
            st.markdown("#### Species-level Δ (B − A) — SRRS and Articles")
            show = (
                dfB[["Species","SRRS"] + ARTICLES].set_index("Species") -
                dfA[["Species","SRRS"] + ARTICLES].set_index("Species")
            ).reset_index()
            st.dataframe(show.head(30))
            st.download_button("Download species-level deltas (CSV)", data=show.to_csv(index=False), file_name="profile_deltas.csv", mime="text/csv")
