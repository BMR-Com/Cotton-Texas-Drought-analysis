"""
Texas Cotton Abandonment — Drought Predictor + Production Estimator
====================================================================
Run:  python process_data.py
"""

import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
CI_LEVEL      = 0.90
MIN_TRAIN_YRS = 15
DROUGHT_VARS  = ["D1-D4", "D2-D4", "D3-D4", "D4"]
SEASON_MONTHS = [4, 5, 6, 7, 8, 9, 10]
COTTON_CSV     = Path("data/cotton_texas.csv")
DROUGHT_US_CSV = Path("data/drought_US.csv")
DROUGHT_CSV   = Path("data/drought_texas.csv")
OUTPUT_HTML   = Path("docs/index.html")
MONTH_NAMES   = {4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct"}
US_GEO        = "US"
ANALOG_TOP_N  = 5
PERIODS       = [1, 5, 10, 15, 20]
PERIOD_LABELS = {1:"1yr", 5:"5yr", 10:"10yr", 15:"15yr", 20:"20yr"}

def week_label(iw):
    ref = pd.Timestamp("2019-01-01") + pd.to_timedelta(int(iw)*7-4, unit="D")
    return f"{MONTH_NAMES.get(ref.month,'M'+str(ref.month))} W{(ref.day-1)//7+1}"

def safe(v, n=4):
    if v is None or (isinstance(v, float) and np.isnan(v)): return None
    return round(float(v), n)

def jd(obj):
    def h(x):
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return None if np.isnan(x) else float(x)
        if isinstance(x, float) and np.isnan(x): return None
        raise TypeError(f"Not serializable: {type(x)}")
    return json.dumps(obj, separators=(",",":"), default=h)

# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
def load_cotton(path):
    """Load all cotton data — TX for regression, all states for production."""
    print(f"  Loading cotton data from {path}...")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    df["period"] = pd.to_numeric(df["period"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["period", "value"]).rename(columns={"period": "mkt_year"})
    CATS = ["upland_cotton_planted_acreage", "upland_cotton_harvested_acreage",
            "upland_cotton_lint_yield", "upland_cotton_production"]
    df = df[df["category"].isin(CATS)].copy()
    print(f"    Total rows: {len(df)}")
    print(f"    Geographies: {sorted(df['geography'].unique())}")
    print(f"    Years: {int(df['mkt_year'].min())}-{int(df['mkt_year'].max())}")
    
    pivot = df.pivot_table(
        index=["geography", "mkt_year"], columns="category",
        values="value", aggfunc="first").reset_index()
    pivot.columns.name = None
    
    # Calculate abandonment
    has_plt = "upland_cotton_planted_acreage" in pivot.columns
    has_hvs = "upland_cotton_harvested_acreage" in pivot.columns
    if has_plt and has_hvs:
        mask = pivot["upland_cotton_planted_acreage"] > 0
        pivot.loc[mask, "abandonment"] = (
            1 - pivot.loc[mask, "upland_cotton_harvested_acreage"] / 
            pivot.loc[mask, "upland_cotton_planted_acreage"]).clip(0, 1)
        print(f"    Calculated abandonment for {mask.sum()} rows")
    
    # Check each state
    for geo in sorted(pivot["geography"].unique()):
        sub = pivot[pivot["geography"] == geo]
        ab_cnt = sub["abandonment"].notna().sum()
        print(f"      {geo}: {len(sub)} rows, {ab_cnt} with abandonment")
    
    return pivot

def load_drought(path, geo_label="TX"):
    print(f"  Loading drought data from {path} for {geo_label}...")
    if not path.exists():
        print(f"    WARNING: {path} not found!")
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if df.columns[0] != "Week":
        df = df.rename(columns={df.columns[0]: "Week"})
    df["Week"] = pd.to_datetime(df["Week"], errors="coerce")
    df = df.dropna(subset=["Week"])
    for v in DROUGHT_VARS:
        if v in df.columns:
            df[v] = pd.to_numeric(df[v], errors="coerce")
    df = df.sort_values("Week").reset_index(drop=True)
    df["iso_week"] = df["Week"].dt.isocalendar().week.astype(int)
    df["cal_year"] = df["Week"].dt.year
    df["month"] = df["Week"].dt.month
    df["mkt_year"] = df.apply(
        lambda r: r["cal_year"] if r["Week"].month >= 4 else r["cal_year"] - 1, axis=1)
    df = df[df["month"].isin(SEASON_MONTHS)].copy()
    print(f"    Loaded {len(df)} rows, years: {sorted(df['mkt_year'].unique())}")
    return df

def get_abandonment_series(cotton, geo):
    sub = cotton[cotton["geography"] == geo].copy()
    if sub.empty:
        print(f"    WARNING: No data for '{geo}'")
        return None
    sub = sub.set_index("mkt_year")
    if "abandonment" not in sub.columns:
        print(f"    WARNING: No abandonment column for '{geo}'")
        return None
    ab = sub["abandonment"].dropna()
    if len(ab) == 0:
        print(f"    WARNING: No abandonment data for '{geo}'")
        return None
    print(f"    {geo}: {len(ab)} years ({int(ab.index.min())}-{int(ab.index.max())})")
    return ab

# ═══════════════════════════════════════════════════════════════════════════
# 2. OLS & REGRESSION MODELS
# ═══════════════════════════════════════════════════════════════════════════
def ols_fit(x_tr, y_tr, x_pred, years_list, ci=CI_LEVEL):
    mask = np.isfinite(x_tr) & np.isfinite(y_tr)
    xc, yc = x_tr[mask], y_tr[mask]
    yrs = [y for y, m in zip(years_list, mask) if m]
    n = len(xc)
    if n < MIN_TRAIN_YRS or not np.isfinite(x_pred): 
        return None
    sl, ic, r, p, _ = stats.linregress(xc, yc)
    r2 = r**2
    s = np.sqrt(np.sum((yc - (ic + sl * xc))**2) / (n - 2))
    tc = stats.t.ppf((1 + ci) / 2, df=n - 2)
    xm = xc.mean()
    ssx = max(np.sum((xc - xm)**2), 1e-9)
    se = s * np.sqrt(1 + 1/n + (x_pred - xm)**2 / ssx)
    yp = float(np.clip(ic + sl * x_pred, 0, 1))
    lo = float(np.clip(yp - tc * se, 0, 1))
    hi = float(np.clip(yp + tc * se, 0, 1))
    xlo, xhi = float(xc.min()), float(xc.max())
    return {
        "r2": round(float(r2), 4), "pvalue": round(float(p), 4),
        "point": round(yp, 4), "lo": round(lo, 4), "hi": round(hi, 4),
        "curr_x": round(float(x_pred), 2),
        "scatter_x": [round(float(v), 2) for v in xc],
        "scatter_y": [round(float(v), 4) for v in yc],
        "scatter_years": [int(y) for y in yrs],
        "reg_x": [round(xlo, 2), round(xhi, 2)],
        "reg_y": [round(float(np.clip(ic + sl * xlo, 0, 1)), 4),
                  round(float(np.clip(ic + sl * xhi, 0, 1)), 4)],
    }

def get_curr_yr(ab, drought):
    ly = int(drought["mkt_year"].max())
    return ly if ly not in ab.index else ly

def get_hist(ab, drought, curr_yr):
    return drought[(drought["mkt_year"] != curr_yr) & (drought["mkt_year"].isin(ab.index))]

def build_arr(hist_dict, ab):
    yrs = sorted(set(hist_dict.keys()) & set(ab.index))
    if len(yrs) == 0:
        return np.array([]), np.array([]), []
    x = np.array([hist_dict[y] for y in yrs], dtype=float)
    y = np.array([ab[y] for y in yrs], dtype=float)
    return x, y, yrs

def build_weekly(ab, drought):
    curr_yr = get_curr_yr(ab, drought)
    hist = get_hist(ab, drought, curr_yr)
    curr_df = drought[drought["mkt_year"] == curr_yr].sort_values("Week")
    rows = []
    for _, cr in curr_df.iterrows():
        iw = int(cr["iso_week"])
        row = {"iso_week": iw, "label": week_label(iw), "date": str(cr["Week"].date()), 
               "curr_yr": curr_yr, "vars": {}}
        for v in DROUGHT_VARS:
            if pd.isna(cr.get(v)): 
                row["vars"][v] = None
                continue
            hd = {}
            for hy, hg in hist.groupby("mkt_year"):
                sub = hg[hg["iso_week"] == iw]
                if sub.empty: 
                    sub = hg[(hg["iso_week"] - iw).abs() <= 1]
                if not sub.empty and pd.notna(sub[v].iloc[0]):
                    hd[hy] = float(sub[v].iloc[0])
            x, y, yrs = build_arr(hd, ab)
            row["vars"][v] = ols_fit(x, y, float(cr[v]), yrs) if len(x) > 0 else None
        rows.append(row)
    return rows, curr_yr

def build_monthly(ab, drought):
    curr_yr = get_curr_yr(ab, drought)
    hist = get_hist(ab, drought, curr_yr)
    curr_df = drought[drought["mkt_year"] == curr_yr]
    rows = []
    for mo in sorted(curr_df["month"].unique()):
        row = {"month": mo, "label": MONTH_NAMES.get(mo, f"M{mo}"), 
               "curr_yr": curr_yr, "vars": {}}
        for v in DROUGHT_VARS:
            mc = curr_df[curr_df["month"] == mo][v].dropna()
            if mc.empty: 
                row["vars"][v] = None
                continue
            hd = {}
            for hy, hg in hist.groupby("mkt_year"):
                sub = hg[hg["month"] == mo][v].dropna()
                if not sub.empty: 
                    hd[hy] = float(sub.mean())
            x, y, yrs = build_arr(hd, ab)
            row["vars"][v] = ols_fit(x, y, float(mc.mean()), yrs) if len(x) > 0 else None
        rows.append(row)
    return rows

def build_cumulative(ab, drought):
    curr_yr = get_curr_yr(ab, drought)
    hist = get_hist(ab, drought, curr_yr)
    curr_df = drought[drought["mkt_year"] == curr_yr].sort_values("Week")
    rows = []
    for _, cr in curr_df.iterrows():
        iw = int(cr["iso_week"])
        row = {"iso_week": iw, "label": week_label(iw), "date": str(cr["Week"].date()),
               "curr_yr": curr_yr, "vars": {}}
        cs = curr_df[curr_df["iso_week"] <= iw]
        for v in DROUGHT_VARS:
            cv = cs[v].dropna()
            if cv.empty: 
                row["vars"][v] = None
                continue
            hd = {}
            for hy, hg in hist.groupby("mkt_year"):
                sub = hg[hg["iso_week"] <= iw][v].dropna()
                if not sub.empty: 
                    hd[hy] = float(sub.mean())
            x, y, yrs = build_arr(hd, ab)
            row["vars"][v] = ols_fit(x, y, float(cv.mean()), yrs) if len(x) > 0 else None
        rows.append(row)
    return rows

def best_prediction(wk, mo, cu):
    best = {"r2": -1, "point": None, "lo": None, "hi": None,
            "variable": None, "model": None, "label": None, "curr_yr": None}
    for rows, mname in [(wk, "Weekly"), (mo, "Monthly"), (cu, "Cumulative")]:
        if not rows: 
            continue
        last = rows[-1]
        for v in DROUGHT_VARS:
            d = last.get("vars", {}).get(v)
            if d and d.get("r2") is not None and d["r2"] > best["r2"]:
                best = {"r2": d["r2"], "point": d["point"], "lo": d["lo"], "hi": d["hi"],
                        "variable": v, "model": mname,
                        "label": last.get("label", ""), "curr_yr": last.get("curr_yr")}
    return best

# ═══════════════════════════════════════════════════════════════════════════
# 3. PRODUCTION DATA (WITH DEBUG OUTPUT)
# ═══════════════════════════════════════════════════════════════════════════
def build_production(cotton, best_tx):
    max_yr = int(cotton["mkt_year"].max())
    print(f"\n    Building production for year {max_yr}...")
    
    state_data = {}
    all_states = [g for g in cotton["geography"].unique() if g != US_GEO]
    print(f"    Found {len(all_states)} states: {all_states}")
    
    for state in all_states:
        sdf = cotton[cotton["geography"] == state].copy()
        sdf = sdf[sdf["mkt_year"] <= max_yr].sort_values("mkt_year")
        if sdf.empty: 
            print(f"      {state}: SKIPPED (empty)")
            continue
        
        print(f"      {state}: {len(sdf)} rows")
        sd = {"periods": {}}
        
        for P in PERIODS:
            rec = sdf[sdf["mkt_year"] >= max_yr - P + 1]
            if rec.empty: 
                sd["periods"][P] = None
                print(f"        {P}yr: No data")
                continue
            
            def g(col):
                if col not in rec.columns: 
                    return None
                v = rec[col].dropna()
                return safe(float(v.mean()), 2) if len(v) else None
            
            sd["periods"][P] = {
                "ab": g("abandonment"), 
                "yld": g("upland_cotton_lint_yield"), 
                "plt": g("upland_cotton_planted_acreage")
            }
            print(f"        {P}yr: ab={sd['periods'][P]['ab']}, yld={sd['periods'][P]['yld']}, plt={sd['periods'][P]['plt']}")
        
        lr = sdf.iloc[-1]
        sd["last_yr_actual"] = {
            "year": int(lr["mkt_year"]), 
            "plt": safe(lr.get("upland_cotton_planted_acreage"), 0)
        }
        state_data[state] = sd
    
    # Build matrices
    matA, matB = {}, {}
    model_ab = best_tx.get("point")
    print(f"\n    Building matrices (model_ab={model_ab})...")
    
    for P_ab in PERIODS:
        matA[P_ab], matB[P_ab] = {}, {}
        for P_yld in PERIODS:
            total_a, total_b, ok = 0.0, 0.0, True
            failed_states = []
            
            for st, sd in state_data.items():
                pa, py = sd["periods"].get(P_ab), sd["periods"].get(P_yld)
                
                if not pa or not py: 
                    ok = False
                    failed_states.append(f"{st}:no_period_data")
                    break
                
                ab_a = pa["ab"]
                ab_b = model_ab if st == "TX" else pa["ab"]
                yld = py["yld"]
                plt = pa["plt"]
                
                if ab_a is None:
                    ok = False
                    failed_states.append(f"{st}:ab_a=None")
                    break
                if yld is None:
                    ok = False
                    failed_states.append(f"{st}:yld=None")
                    break
                if plt is None:
                    ok = False
                    failed_states.append(f"{st}:plt=None")
                    break
                
                prod_a = plt * 1000 * (1 - ab_a) * yld / 480_000_000
                prod_b = plt * 1000 * (1 - ab_b) * yld / 480_000_000
                total_a += prod_a
                total_b += prod_b
            
            matA[P_ab][P_yld] = round(total_a, 3) if ok else None
            matB[P_ab][P_yld] = round(total_b, 3) if ok else None
            
            if not ok:
                print(f"      FAIL {PERIOD_LABELS[P_ab]}x{PERIOD_LABELS[P_yld]}: {', '.join(failed_states)}")
            else:
                print(f"      OK   {PERIOD_LABELS[P_ab]}x{PERIOD_LABELS[P_yld]}: A={matA[P_ab][P_yld]:.2f}, B={matB[P_ab][P_yld]:.2f}")
    
    # TX abandonment range
    tx_hist = cotton[(cotton["geography"] == "TX") & (cotton["mkt_year"] >= max_yr - 9)]
    if not tx_hist.empty and "abandonment" in tx_hist.columns:
        ab_vals = tx_hist["abandonment"].dropna()
        ab_lo = int(float(ab_vals.min()) * 20) / 20
        ab_hi = (int(float(ab_vals.max()) * 20) + 1) / 20
        tx_ab_range = [round(ab_lo + i * 0.05, 2) for i in range(int(round((ab_hi - ab_lo) / 0.05)) + 1)]
    else:
        tx_ab_range = [round(0.10 + i * 0.05, 2) for i in range(13)]
    
    return {
        "state_data": state_data, 
        "matA": {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in matA.items()},
        "matB": {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in matB.items()},
        "tx_ab_range": tx_ab_range, 
        "periods": PERIODS, 
        "period_labels": PERIOD_LABELS, 
        "max_yr": max_yr, 
        "model_ab": model_ab
    }

def build_analyst_summary(best_tx, wk_rows, mo_rows, cu_rows, prod, ab, drought):
    curr_yr = best_tx.get("curr_yr", "?")
    model_ab = best_tx.get("point")
    return {
        "seasonality": f"Current season ({curr_yr}): {len(wk_rows)} weeks available.",
        "weekly": f"Weekly: Best {best_tx.get('variable','—')} R²={best_tx.get('r2',0)*100:.1f}%",
        "monthly": "Monthly: Averages weekly readings for stability.",
        "cumulative": "Cumulative: Running average from Apr W1.",
        "production": f"TX abandonment: {(model_ab or 0)*100:.1f}% via {best_tx.get('model','—')}"
    }

# ═══════════════════════════════════════════════════════════════════════════
# 4. HTML GENERATION
# ═══════════════════════════════════════════════════════════════════════════
def make_html(wk_rows, mo_rows, cu_rows, prod, best_tx, summary, ci_pct):
    curr_yr = wk_rows[0]["curr_yr"] if wk_rows else "?"
    j_wk, j_mo, j_cu = jd(wk_rows), jd(mo_rows), jd(cu_rows)
    j_prod, j_btx = jd(prod), jd(best_tx)
    
    # Check if matrices have data
    has_data = any(v for p in prod.get('matA', {}).values() for v in p.values() if v is not None)
    
    return f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>TX Cotton Drought Predictor</title>
<style>
body{{font-family:system-ui;background:#0d1117;color:#e2e8f0;margin:0}}
.hdr{{background:linear-gradient(135deg,#0f2a1a,#1a3050);padding:16px;border-bottom:2px solid #2d6a4f}}
h1{{margin:0;font-size:1.3rem;color:#f0fff4}} 
.tabs{{display:flex;background:#111820;border-bottom:1px solid #2d6a4f}}
.tab{{padding:10px 16px;cursor:pointer;color:#90a4ae}}
.tab.active{{color:#68d391;border-bottom:2px solid #68d391}}
.panel{{display:none;padding:16px}}
.panel.active{{display:block}}
.st{{color:#68d391;font-weight:600;margin:12px 0 6px}}
table{{width:100%;border-collapse:collapse;font-size:.8rem}}
th,td{{padding:6px;border:1px solid #1e2a3a;text-align:center}}
th{{background:#0e1d2e;color:#7fb3d3}}
td{{background:#0d1117}}
.mat{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
@media(max-width:700px){{.mat{{grid-template-columns:1fr}}}}
.error{{color:#fc8181;padding:20px}}
.success{{color:#68d391;padding:20px}}
</style></head>
<body>
<div class="hdr"><h1>🌾 TX Cotton Drought Predictor · MY {curr_yr}</h1>
<p>Best: {best_tx.get('variable','—')} R²={best_tx.get('r2',0)*100:.1f}% → TX abandonment {(best_tx.get('point') or 0)*100:.1f}%</p>
<p>Matrices: {'✓ DATA PRESENT' if has_data else '✗ EMPTY - check debug output'}</p></div>
<div class="tabs">
<div class="tab active" onclick="showTab('t1')">Weekly</div>
<div class="tab" onclick="showTab('t2')">Monthly</div>
<div class="tab" onclick="showTab('t3')">Cumulative</div>
<div class="tab" onclick="showTab('t4')">Production</div>
</div>
<div id="t1" class="panel active"><div class="st">Weekly Model</div><div id="tbl-w"></div></div>
<div id="t2" class="panel"><div class="st">Monthly Model</div><div id="tbl-m"></div></div>
<div id="t3" class="panel"><div class="st">Cumulative Model</div><div id="tbl-c"></div></div>
<div id="t4" class="panel">
<div class="st">Production Matrices (mn 480-lb bales)</div>
<div class="mat"><div><h3>Matrix A - All Historical</h3><div id="mA"></div></div>
<div><h3>Matrix B - TX from Model ★</h3><div id="mB"></div></div></div>
</div>
<script>
var WK={j_wk}, MO={j_mo}, CU={j_cu}, PROD={j_prod}, BTX={j_btx};
function showTab(id){{document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));document.getElementById(id).classList.add('active');document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));event.target.classList.add('active');if(id==='t4')renderProd();}}
function renderProd(){{var mA=document.getElementById('mA'),mB=document.getElementById('mB');if(!PROD.matA){{mA.innerHTML='<div class="error">No matrix data</div>';return;}}var h='<table><tr><th>Ab↓ Yld→</th>'+PROD.periods.map(p=>'<th>'+PROD.period_labels[p]+'</th>').join('')+'</tr>';PROD.periods.forEach(pAb=>{{h+='<tr><th>'+PROD.period_labels[pAb]+'</th>'+PROD.periods.map(pYld=>'<td>'+(PROD.matA[pAb]&&PROD.matA[pAb][pYld]!==null?PROD.matA[pAb][pYld].toFixed(2):'—')+'</td>').join('')+'</tr>';}});h+='</table>';mA.innerHTML=h;if(!PROD.matB){{mB.innerHTML='<div class="error">No matrix B data</div>';return;}}var h='<table><tr><th>Ab↓ Yld→</th>'+PROD.periods.map(p=>'<th>'+PROD.period_labels[p]+'</th>').join('')+'</tr>';PROD.periods.forEach(pAb=>{{h+='<tr><th>'+PROD.period_labels[pAb]+'</th>'+PROD.periods.map(pYld=>'<td>'+(PROD.matB[pAb]&&PROD.matB[pAb][pYld]!==null?PROD.matB[pAb][pYld].toFixed(2):'—')+'</td>').join('')+'</tr>';}});h+='</table>';mB.innerHTML=h;}}
function drawTable(id,data){{var el=document.getElementById(id);if(!el||!data)return;var h='<table><tr><th>Period</th><th>D1-D4</th><th>D2-D4</th><th>D3-D4</th><th>D4</th></tr>';data.forEach(r=>{{h+='<tr><td>'+r.label+'</td>';['D1-D4','D2-D4','D3-D4','D4'].forEach(v=>{{var d=r.vars[v];h+='<td>'+(d&&d.point!==null?(d.point*100).toFixed(1)+'%':'—')+'</td>';}});h+='</tr>';}});h+='</table>';el.innerHTML=h;}}
window.onload=function(){{drawTable('tbl-w',WK);drawTable('tbl-m',MO);drawTable('tbl-c',CU);renderProd();}};
</script></body></html>'''

# ═══════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("TX + US Cotton — Drought Predictor")
    print("=" * 70)
    
    for p in [COTTON_CSV, DROUGHT_CSV]:
        if not p.exists(): 
            print(f"ERROR: {p} not found")
            sys.exit(1)
    
    has_us = DROUGHT_US_CSV.exists()
    print(f"US drought file: {'FOUND' if has_us else 'NOT FOUND'}")
    
    print("\nLoading cotton data...")
    cotton = load_cotton(COTTON_CSV)
    
    print("\nLoading drought data...")
    drought_tx = load_drought(DROUGHT_CSV, "TX")
    
    print("\nGetting abandonment series...")
    ab_tx = get_abandonment_series(cotton, "TX")
    if ab_tx is None:
        print("ERROR: No TX abandonment data")
        sys.exit(1)
    
    print("\nBuilding TX regression models...")
    wk_rows, _ = build_weekly(ab_tx, drought_tx)
    mo_rows = build_monthly(ab_tx, drought_tx)
    cu_rows = build_cumulative(ab_tx, drought_tx)
    best_tx = best_prediction(wk_rows, mo_rows, cu_rows)
    print(f"TX: {len(wk_rows)} weekly rows")
    print(f"Best: {best_tx.get('variable','—')} ({best_tx.get('model','—')}) R²={best_tx.get('r2',0)*100:.1f}%")
    
    print("\nBuilding production data...")
    prod = build_production(cotton, best_tx)
    
    # Check what we got
    print(f"\nProduction data summary:")
    print(f"  States: {len(prod['state_data'])}")
    print(f"  Matrix A keys: {list(prod['matA'].keys()) if prod['matA'] else 'None'}")
    print(f"  Sample A[5yr][5yr]: {prod['matA'].get('5', {}).get('5', 'N/A') if prod['matA'] else 'N/A'}")
    
    summary = build_analyst_summary(best_tx, wk_rows, mo_rows, cu_rows, prod, ab_tx, drought_tx)
    
    print("\nGenerating HTML...")
    OUTPUT_HTML.parent.mkdir(exist_ok=True)
    html = make_html(wk_rows, mo_rows, cu_rows, prod, best_tx, summary, "90")
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"\n✓ Done: {OUTPUT_HTML} ({OUTPUT_HTML.stat().st_size//1024} KB)")

if __name__ == "__main__":
    main()
