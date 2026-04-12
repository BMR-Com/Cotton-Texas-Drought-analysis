"""
Texas Cotton Abandonment — Drought Predictor + Production Estimator
====================================================================
Run:  python process_data.py

Reads:
    data/cotton_texas.csv  — USDA NASS cotton data (all states once full file uploaded)
    data/drought_texas.csv — USDA Drought Monitor weekly TX cotton area

Writes:
    docs/index.html — self-contained dashboard (open in any browser)

Install:  pip install numpy pandas scipy
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
EXCLUDE_STATES = {"KY", "NV"}  # Kentucky and Nevada have zero cotton data
US_GEO        = "US"
ANALOG_TOP_N  = 5
PERIODS       = [1, 5, 10, 15, 20]
PERIOD_LABELS = {1:"1yr", 5:"5yr", 10:"10yr", 15:"15yr", 20:"20yr"}

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════
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
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["geography"] = df["geography"].str.strip()

    # FILTER OUT INVALID STATES (KY, NV with zero data)
    df = df[~df["geography"].isin(EXCLUDE_STATES)].copy()
    df["period"]    = pd.to_numeric(df["period"], errors="coerce")
    df["value"]     = pd.to_numeric(df["value"],  errors="coerce")
    df = df.dropna(subset=["period","value"]).rename(columns={"period":"mkt_year"})

    CATS = ["upland_cotton_planted_acreage", "upland_cotton_harvested_acreage",
            "upland_cotton_lint_yield",       "upland_cotton_production"]
    df = df[df["category"].isin(CATS)]

    pivot = df.pivot_table(
        index=["geography","mkt_year"], columns="category",
        values="value", aggfunc="first"
    ).reset_index()
    pivot.columns.name = None

    # Compute abandonment for non-US rows
    has_plt = "upland_cotton_planted_acreage"  in pivot.columns
    has_hvs = "upland_cotton_harvested_acreage" in pivot.columns
    if has_plt and has_hvs:
        non_us = pivot["geography"] != US_GEO
        # Ensure planted > 0 to avoid division by zero
        valid_plt = (pivot.loc[non_us, "upland_cotton_planted_acreage"] > 0) & \
                    (pivot.loc[non_us, "upland_cotton_harvested_acreage"].notna())

        pivot.loc[non_us & valid_plt, "abandonment"] = (
            1 - pivot.loc[non_us & valid_plt, "upland_cotton_harvested_acreage"]
              / pivot.loc[non_us & valid_plt, "upland_cotton_planted_acreage"]
        ).clip(0, 1)

        # Mark invalid abandonment as NaN
        pivot.loc[non_us & ~valid_plt, "abandonment"] = np.nan

    geos = sorted(pivot["geography"].unique())
    print(f"  Cotton: {len(geos)} geographies, "
          f"years {int(pivot.mkt_year.min())}–{int(pivot.mkt_year.max())}")
    return pivot


def build_us_ab(cotton):
    """US abandonment from the US-total row in cotton CSV."""
    us = cotton[cotton["geography"]==US_GEO].copy()
    if us.empty: return None
    us = us.set_index("mkt_year")
    if "abandonment" not in us.columns:
        p,h = "upland_cotton_planted_acreage","upland_cotton_harvested_acreage"
        if p in us.columns and h in us.columns:
            us["abandonment"] = (1 - us[h]/us[p]).clip(0,1)
        else:
            return None
    ab = us["abandonment"].dropna()
    if len(ab) < 5: return None
    print(f"  US abandonment: {len(ab)} years ({int(ab.index.min())}–{int(ab.index.max())})")
    return ab

def load_drought(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={df.columns[0]:"Week"})
    df["Week"] = pd.to_datetime(df["Week"], errors="coerce")
    df = df.dropna(subset=["Week"])
    for v in DROUGHT_VARS:
        if v in df.columns:
            df[v] = pd.to_numeric(df[v], errors="coerce")
    df = df.sort_values("Week").reset_index(drop=True)
    df["iso_week"] = df["Week"].dt.isocalendar().week.astype(int)
    df["cal_year"] = df["Week"].dt.year
    df["month"]    = df["Week"].dt.month
    df["mkt_year"] = df.apply(
        lambda r: r["cal_year"] if r["Week"].month>=4 else r["cal_year"]-1, axis=1)
    df = df[df["month"].isin(SEASON_MONTHS)].copy()
    print(f"  Drought: {len(df)} rows "
          f"({df['Week'].min().date()} – {df['Week'].max().date()})")
    return df

# ═══════════════════════════════════════════════════════════════════════════
# MONTE CARLO SIMULATION FOR PRODUCTION
# ═══════════════════════════════════════════════════════════════════════════
def run_monte_carlo_simulation(cotton, state_data, n_simulations=10000):
    """
    Run Monte Carlo simulation for TX and US production.
    Returns confidence intervals for production estimates.
    """
    np.random.seed(42)  # For reproducibility
    
    results = {
        "tx": {"production": [], "abandonment": [], "yield": []},
        "us": {"production": [], "abandonment": [], "yield": []}
    }
    
    # Get TX data for last 10 years
    tx_data = cotton[(cotton["geography"]=="TX") & (cotton["mkt_year"]>=cotton["mkt_year"].max()-9)]
    us_data = cotton[(cotton["geography"]==US_GEO) & (cotton["mkt_year"]>=cotton["mkt_year"].max()-9)]
    
    if tx_data.empty or us_data.empty:
        return None
    
    # Extract historical distributions
    tx_ab = tx_data["abandonment"].dropna()
    tx_yld = tx_data["upland_cotton_lint_yield"].dropna()
    tx_plt = tx_data["upland_cotton_planted_acreage"].dropna()
    
    us_ab = us_data["abandonment"].dropna() if "abandonment" in us_data.columns else tx_ab
    us_yld = us_data["upland_cotton_lint_yield"].dropna()
    us_plt = us_data["upland_cotton_planted_acreage"].dropna()
    
    # Calculate means and stds for simulation
    def get_dist_params(series):
        if len(series) < 2:
            return series.iloc[0] if len(series) == 1 else 0, 0.1
        return series.mean(), series.std()
    
    tx_ab_mean, tx_ab_std = get_dist_params(tx_ab)
    tx_yld_mean, tx_yld_std = get_dist_params(tx_yld)
    tx_plt_mean = tx_plt.iloc[-1] if not tx_plt.empty else 5000
    
    us_ab_mean, us_ab_std = get_dist_params(us_ab)
    us_yld_mean, us_yld_std = get_dist_params(us_yld)
    us_plt_mean = us_plt.iloc[-1] if not us_plt.empty else 7000
        # Run simulations
    for _ in range(n_simulations):
        # TX simulation
        tx_ab_sim = np.clip(np.random.normal(tx_ab_mean, tx_ab_std), 0, 1)
        tx_yld_sim = max(0, np.random.normal(tx_yld_mean, tx_yld_std))
        tx_prod_sim = tx_plt_mean * 1000 * (1 - tx_ab_sim) * tx_yld_sim / 480_000_000
        
        results["tx"]["production"].append(tx_prod_sim)
        results["tx"]["abandonment"].append(tx_ab_sim)
        results["tx"]["yield"].append(tx_yld_sim)
        
        # US simulation
        us_ab_sim = np.clip(np.random.normal(us_ab_mean, us_ab_std), 0, 1)
        us_yld_sim = max(0, np.random.normal(us_yld_mean, us_yld_std))
        us_prod_sim = us_plt_mean * 1000 * (1 - us_ab_sim) * us_yld_sim / 480_000_000
        
        results["us"]["production"].append(us_prod_sim)
        results["us"]["abandonment"].append(us_ab_sim)
        results["us"]["yield"].append(us_yld_sim)
    
    # Calculate statistics
    def calc_stats(arr):
        arr = np.array(arr)
        return {
            "mean": round(float(np.mean(arr)), 3),
            "median": round(float(np.median(arr)), 3),
            "std": round(float(np.std(arr)), 3),
            "ci_5": round(float(np.percentile(arr, 5)), 3),
            "ci_95": round(float(np.percentile(arr, 95)), 3),
            "min": round(float(np.min(arr)), 3),
            "max": round(float(np.max(arr)), 3)
        }
    
    mc_results = {
        "tx": {k: calc_stats(v) for k, v in results["tx"].items()},
        "us": {k: calc_stats(v) for k, v in results["us"].items()},
        "n_simulations": n_simulations,
        "tx_params": {
            "abandonment_mean": round(tx_ab_mean, 4),
            "abandonment_std": round(tx_ab_std, 4),
            "yield_mean": round(tx_yld_mean, 1),
            "yield_std": round(tx_yld_std, 1),
            "planted": round(tx_plt_mean, 0)
        },
        "us_params": {
            "abandonment_mean": round(us_ab_mean, 4),
            "abandonment_std": round(us_ab_std, 4),
            "yield_mean": round(us_yld_mean, 1),
            "yield_std": round(us_yld_std, 1),
            "planted": round(us_plt_mean, 0)
        }
    }
    
    return mc_results

# ═══════════════════════════════════════════════════════════════════════════
# 2. OLS HELPER
# ═══════════════════════════════════════════════════════════════════════════
def ols_fit(x_tr, y_tr, x_pred, years_list, ci=CI_LEVEL):
    mask = np.isfinite(x_tr) & np.isfinite(y_tr)
    xc, yc = x_tr[mask], y_tr[mask]
    yrs = [y for y,m in zip(years_list, mask) if m]
    n = len(xc)
    if n < MIN_TRAIN_YRS or not np.isfinite(x_pred): 
        return None

    # Check for constant x (no variance)
    if np.var(xc) < 1e-10:
        return None

    sl, ic, r, p, _ = stats.linregress(xc, yc)
    r2 = r**2

    # Handle numerical issues
    if not np.isfinite(r2) or r2 < 0:
        r2 = 0

    s  = np.sqrt(np.sum((yc-(ic+sl*xc))**2)/(n-2))
    tc = stats.t.ppf((1+ci)/2, df=n-2)
    xm = xc.mean(); ssx = max(np.sum((xc-xm)**2), 1e-9)
    se = s*np.sqrt(1+1/n+(x_pred-xm)**2/ssx)
    yp = float(np.clip(ic+sl*x_pred, 0, 1))
    lo = float(np.clip(yp-tc*se, 0, 1))
    hi = float(np.clip(yp+tc*se, 0, 1))
    xlo, xhi = float(xc.min()), float(xc.max())

    # Ensure confidence interval is valid
    if not np.isfinite(lo) or lo < 0: lo = 0
    if not np.isfinite(hi) or hi > 1: hi = 1
    if lo > hi: lo, hi = hi, lo

    return {
        "r2": round(float(r2),4), "pvalue": round(float(p),4),
        "point": round(yp,4), "lo": round(lo,4), "hi": round(hi,4),
        "curr_x": round(float(x_pred),2),
        "scatter_x":     [round(float(v),2) for v in xc],
        "scatter_y":     [round(float(v),4) for v in yc],
        "scatter_years": [int(y) for y in yrs],
        "reg_x": [round(xlo,2), round(xhi,2)],
        "reg_y": [round(float(np.clip(ic+sl*xlo,0,1)),4),
                  round(float(np.clip(ic+sl*xhi,0,1)),4)],
    }

# ═══════════════════════════════════════════════════════════════════════════
# 3. REGRESSION MODELS
# ═══════════════════════════════════════════════════════════════════════════
def get_tx(cotton):
    """Extract TX abandonment series."""
    tx = cotton[cotton["geography"]=="TX"].copy()
    if tx.empty:
        return pd.Series(dtype=float)
    ab = tx.set_index("mkt_year")["abandonment"]
    return ab.dropna()

def get_curr_yr(ab, drought):
    ly = int(drought["mkt_year"].max())
    return ly if ly not in ab.index else ly

def get_hist(ab, drought, curr_yr):
    return drought[
        (drought["mkt_year"] != curr_yr) &
        (drought["mkt_year"].isin(ab.index))
    ]

def build_arr(hist_dict, ab):
    yrs = sorted(set(hist_dict.keys()) & set(ab.index))
    x = np.array([hist_dict[y] for y in yrs], dtype=float)
    y = np.array([ab[y] for y in yrs], dtype=float)
    return x, y, yrs

def build_weekly(ab, drought):
    curr_yr = get_curr_yr(ab, drought)
    hist    = get_hist(ab, drought, curr_yr)
    curr_df = drought[drought["mkt_year"]==curr_yr].sort_values("Week")
    rows = []
    for _, cr in curr_df.iterrows():
        iw  = int(cr["iso_week"])
        row = {"iso_week":iw, "label":week_label(iw),
               "date":str(cr["Week"].date()), "curr_yr":curr_yr, "vars":{}}
        for v in DROUGHT_VARS:
            if pd.isna(cr.get(v)): row["vars"][v]=None; continue
            hd={}
            for hy,hg in hist.groupby("mkt_year"):
                sub=hg[hg["iso_week"]==iw]
                if sub.empty: sub=hg[(hg["iso_week"]-iw).abs()<=1]
                if not sub.empty and pd.notna(sub[v].iloc[0]):
                    hd[hy]=float(sub[v].iloc[0])
            x,y,yrs=build_arr(hd,ab)
            row["vars"][v]=ols_fit(x,y,float(cr[v]),yrs)
        rows.append(row)
    return rows, curr_yr

def build_monthly(ab, drought):
    curr_yr = get_curr_yr(ab, drought)
    hist    = get_hist(ab, drought, curr_yr)
    curr_df = drought[drought["mkt_year"]==curr_yr]
    rows=[]
    for mo in sorted(curr_df["month"].unique()):
        row={"month":mo,"label":MONTH_NAMES.get(mo,f"M{mo}"),"curr_yr":curr_yr,"vars":{}}
        for v in DROUGHT_VARS:
            mc=curr_df[curr_df["month"]==mo][v].dropna()
            if mc.empty: row["vars"][v]=None; continue
            hd={}
            for hy,hg in hist.groupby("mkt_year"):
                sub=hg[hg["month"]==mo][v].dropna()
                if not sub.empty: hd[hy]=float(sub.mean())
            x,y,yrs=build_arr(hd,ab)
            row["vars"][v]=ols_fit(x,y,float(mc.mean()),yrs)
        rows.append(row)
    return rows

def build_cumulative(ab, drought):
    curr_yr = get_curr_yr(ab, drought)
    hist    = get_hist(ab, drought, curr_yr)
    curr_df = drought[drought["mkt_year"]==curr_yr].sort_values("Week")
    rows=[]
    for _,cr in curr_df.iterrows():
        iw=int(cr["iso_week"])
        row={"iso_week":iw,"label":week_label(iw),
             "date":str(cr["Week"].date()),"curr_yr":curr_yr,"vars":{}}
        cs=curr_df[curr_df["iso_week"]<=iw]
        for v in DROUGHT_VARS:
            cv=cs[v].dropna()
            if cv.empty: row["vars"][v]=None; continue
            hd={}
            for hy,hg in hist.groupby("mkt_year"):
                sub=hg[hg["iso_week"]<=iw][v].dropna()
                if not sub.empty: hd[hy]=float(sub.mean())
            x,y,yrs=build_arr(hd,ab)
            row["vars"][v]=ols_fit(x,y,float(cv.mean()),yrs)
        rows.append(row)
    return rows

def best_prediction(wk, mo, cu):
    best={"r2":-1,"point":None,"lo":None,"hi":None,
          "variable":None,"model":None,"label":None,"curr_yr":None}
    for rows,mname in [(wk,"Weekly"),(mo,"Monthly"),(cu,"Cumulative")]:
        if not rows: continue
        last=rows[-1]
        for v in DROUGHT_VARS:
            d=last.get("vars",{}).get(v)
            if d and d.get("r2") is not None and d["r2"]>best["r2"]:
                best={"r2":d["r2"],"point":d["point"],"lo":d["lo"],"hi":d["hi"],
                      "variable":v,"model":mname,
                      "label":last.get("label",""),"curr_yr":last.get("curr_yr")}
    return best
    # ═══════════════════════════════════════════════════════════════════════════
# 4. SEASONALITY LINE DATA - FIXED VERSION
# ═══════════════════════════════════════════════════════════════════════════
def build_season_lines(drought, geo_label='TX', n_years=20):
    """
    Build seasonality line chart data.
    Shows: Current year (bold), Last 2 years (distinct colors), Top 5 analogs (other colors)
    All other years available for toggle selection.
    """
    max_yr   = int(drought["cal_year"].max())
    curr_yr  = int(drought["mkt_year"].max())
    min_yr   = max_yr - n_years + 1
    
    s_weeks  = sorted(w for w in drought["iso_week"].unique() if 14<=w<=43)
    wlabels  = [week_label(w) for w in s_weeks]
    
    curr_df  = drought[drought["mkt_year"]==curr_yr].sort_values("Week")
    lat_iw   = int(curr_df["iso_week"].max()) if not curr_df.empty else s_weeks[-1]
    
    # All historical years (excluding current)
    all_hist_yrs = sorted(y for y in drought["cal_year"].unique()
                          if min_yr<=y<=max_yr and y!=curr_yr)
    
    # Last 2 years
    last_yr_1 = curr_yr - 1
    last_yr_2 = curr_yr - 2
    
    out = {
        "geo": geo_label, 
        "weeks": wlabels, 
        "iso_weeks": [int(w) for w in s_weeks],
        "curr_yr": int(curr_yr), 
        "latest_iw": int(lat_iw),
        "last_yr_1": int(last_yr_1),
        "last_yr_2": int(last_yr_2),
        "all_hist_years": [int(y) for y in all_hist_yrs],
        "variables": {},
        "last6_weeks": [], 
        "last6_comparisons": []
    }

    # Last 6 current-year weeks
    for _,row in curr_df.tail(6).iterrows():
        entry = {
            "date": str(row["Week"].date()),
            "label": week_label(int(row["iso_week"])),
            "iso_week": int(row["iso_week"]),
            "year": int(curr_yr),
            "is_current": True
        }
        for v in DROUGHT_VARS: 
            entry[v] = safe(row.get(v), 1)
        out["last6_weeks"].append(entry)

    # Latest week for comparison
    latest_row = curr_df.iloc[-1] if not curr_df.empty else None
    latest_iw_for_comp = int(latest_row["iso_week"]) if latest_row is not None else lat_iw
    out["latest_week_iw"] = latest_iw_for_comp
    out["latest_week_date"] = str(latest_row["Week"].date()) if latest_row is not None else ""
    out["last6_iws"] = [int(r["iso_week"]) for _, r in curr_df.tail(6).iterrows()]

    for v in DROUGHT_VARS:
        # Build year series for all years
        series = {}
        for yr in all_hist_yrs + [curr_yr]:
            yr_df = drought[drought["cal_year"] == yr]
            line = []
            for iw in s_weeks:
                sub = yr_df[yr_df["iso_week"] == iw]
                val = float(sub[v].iloc[0]) if (not sub.empty and pd.notna(sub[v].iloc[0])) else None
                line.append(val)
            if any(x is not None for x in line):
                series[int(yr)] = line

        # Analog computation: RMSE through latest available week
        lat_idx = s_weeks.index(lat_iw) if lat_iw in s_weeks else len(s_weeks) - 1
        c_line = series.get(curr_yr, [])
        c_vals = [c_line[i] for i in range(lat_idx + 1)
                  if i < len(c_line) and c_line[i] is not None]
        
        scores = []
        for yr in all_hist_yrs:
            yl = series.get(yr, [])
            yv = [yl[i] for i in range(lat_idx + 1) if i < len(yl) and yl[i] is not None]
            n = min(len(c_vals), len(yv))
            if n < 2: continue
            rmse = float(np.sqrt(np.mean((np.array(c_vals[:n]) - np.array(yv[:n]))**2)))
            scores.append((yr, rmse))
        
        scores.sort(key=lambda x: x[1])
        top5 = [int(y) for y, _ in scores[:ANALOG_TOP_N]]

        # Comparison data for last 6 weeks table
        comparisons = []
        
        # Add last 2 years + top 5 analogs for comparison
        comp_years = [last_yr_1, last_yr_2] + top5
        
        for comp_yr in comp_years:
            if comp_yr not in series:
                continue
            comp_df = drought[drought["cal_year"] == comp_yr]
            
            # Get values for the last 6 weeks of current year
            comp_entries = []
            for iw in out["last6_iws"]:
                sub = comp_df[comp_df["iso_week"] == iw]
                entry_week = {
                    "iso_week": iw,
                    "label": week_label(iw),
                    "year": int(comp_yr)
                }
                for dv in DROUGHT_VARS:
                    entry_week[dv] = safe(float(sub[dv].iloc[0]), 1) if (not sub.empty and dv in sub.columns and pd.notna(sub[dv].iloc[0])) else None
                comp_entries.append(entry_week)
            
            # RMSE score
            rmse = next((round(s, 2) for y, s in scores if y == comp_yr), None)
            
            comparisons.append({
                "year": int(comp_yr),
                "is_last_yr": comp_yr == last_yr_1,
                "is_last_yr_2": comp_yr == last_yr_2,
                "is_analog": comp_yr in top5,
                "rmse": rmse,
                "entries": comp_entries
            })

        out["variables"][v] = {
            "series": {str(k): v2 for k, v2 in series.items()},
            "analogs": top5,
            "analog_scores": {int(y): round(s, 2) for y, s in scores[:ANALOG_TOP_N]},
            "all_hist_years": [int(y) for y in all_hist_yrs],
            "last_yr_1": int(last_yr_1),
            "last_yr_2": int(last_yr_2),
            "last6_comparisons": comparisons,
        }

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 5. PRODUCTION DATA - WITH MONTE CARLO
# ═══════════════════════════════════════════════════════════════════════════
def build_production(cotton, best_tx):
    """
    For each period P:
      Each state contributes: planted_P * (1-abandon_P) * yield_P
      Sum across all non-US states = US production estimate
    Returns everything the JS needs for all 4 matrices/grids.
    Includes Monte Carlo simulation results.
    """
    max_yr = int(cotton["mkt_year"].max())
    prev_yr = max_yr - 1

    # Get per-state per-period averages
    state_data = {}
    all_states = [g for g in cotton["geography"].unique() if g != US_GEO]

    for state in all_states:
        sdf = cotton[cotton["geography"]==state].copy()
        sdf = sdf[sdf["mkt_year"]<=max_yr].sort_values("mkt_year")
        if sdf.empty: continue
        sd = {"periods": {}}
        for P in PERIODS:
            rec = sdf[sdf["mkt_year"]>=max_yr-P+1]
            if rec.empty: 
                sd["periods"][P] = None
                continue
            def g(col):
                if col not in rec.columns: 
                    return None
                v = rec[col].dropna()
                return safe(float(v.mean()), 2) if len(v) else None
            sd["periods"][P] = {
                "ab":  g("abandonment"),
                "yld": g("upland_cotton_lint_yield"),
                "plt": g("upland_cotton_planted_acreage"),
                "prd": g("upland_cotton_production"),
            }
        # Last year actual
        lr = sdf.iloc[-1]
        sd["last_yr_actual"] = {
            "year": int(lr["mkt_year"]),
            "ab":  safe(lr.get("abandonment"), 4),
            "yld": safe(lr.get("upland_cotton_lint_yield"), 1),
            "plt": safe(lr.get("upland_cotton_planted_acreage"), 0),
            "prd": safe(lr.get("upland_cotton_production"), 3),
        }
        # Previous year actual
        prev = sdf[sdf["mkt_year"] == prev_yr]
        if not prev.empty:
            pr = prev.iloc[0]
            sd["prev_yr_actual"] = {
                "year": int(prev_yr),
                "ab":  safe(pr.get("abandonment"), 4),
                "yld": safe(pr.get("upland_cotton_lint_yield"), 1),
                "plt": safe(pr.get("upland_cotton_planted_acreage"), 0),
                "prd": safe(pr.get("upland_cotton_production"), 3),
            }
        else:
            sd["prev_yr_actual"] = None
        state_data[state] = sd

    def prod_mn_bales(plt_k, ab, yld):
        """plt_k = 1000 acres, yld = lb/acre → mn 480-lb bales"""
        if plt_k is None or ab is None or yld is None: 
            return None
        return round(plt_k * 1000 * (1-ab) * yld / 480_000_000, 3)

    # Calculate production ranges for each matrix
    matrix_ranges = {}
    
    # Matrix A: 5×5 — rows=abandon period, cols=yield period, all states historical
    # Matrix B: same but TX abandonment = model prediction
    matA, matB = {}, {}
    model_ab = best_tx.get("point")

    for P_ab in PERIODS:
        matA[P_ab] = {}
        matB[P_ab] = {}
        for P_yld in PERIODS:
            total_a = 0.0
            total_b = 0.0
            ok = True
            for st, sd in state_data.items():
                pa = sd["periods"].get(P_ab)
                py = sd["periods"].get(P_yld)
                if not pa or not py: 
                    ok = False
                    break
                ab_a = pa["ab"]
                ab_b = (model_ab if st=="TX" and model_ab is not None else pa["ab"])
                yld = py["yld"]
                plt = pa["plt"]
                if ab_a is None or yld is None or plt is None: 
                    ok = False
                    break
                val_a = plt * 1000 * (1-ab_a) * yld / 480_000_000
                val_b = plt * 1000 * (1-ab_b) * yld / 480_000_000
                total_a += val_a
                total_b += val_b
            matA[P_ab][P_yld] = round(total_a, 3) if ok else None
            matB[P_ab][P_yld] = round(total_b, 3) if ok else None
    
    # Calculate ranges
    matA_vals = [v for row in matA.values() for v in row.values() if v is not None]
    matB_vals = [v for row in matB.values() for v in row.values() if v is not None]
    
    matrix_ranges["matA"] = {
        "min": round(min(matA_vals), 2) if matA_vals else None,
        "max": round(max(matA_vals), 2) if matA_vals else None,
        "label": "All Historical"
    }
    matrix_ranges["matB"] = {
        "min": round(min(matB_vals), 2) if matB_vals else None,
        "max": round(max(matB_vals), 2) if matB_vals else None,
        "label": "TX from Model"
    }

    # TX abandonment range for Grid C (last 10 yrs, 5% steps)
    tx_hist = cotton[(cotton["geography"]=="TX") & (cotton["mkt_year"]>=max_yr-9)]
    if not tx_hist.empty and "abandonment" in tx_hist.columns:
        ab_vals = tx_hist["abandonment"].dropna()
        ab_lo = int(float(ab_vals.min())*20)/20
        ab_hi = (int(float(ab_vals.max())*20)+1)/20
        tx_ab_range = [round(ab_lo+i*0.05, 2) for i in range(int(round((ab_hi-ab_lo)/0.05))+1)
                        if round(ab_lo+i*0.05, 2) <= ab_hi+0.001]
    else:
        tx_ab_range = [round(0.10+i*0.05, 2) for i in range(13)]

    # TX yield range (50lb steps, last 10 yrs)
    if not tx_hist.empty and "upland_cotton_lint_yield" in tx_hist.columns:
        yv = tx_hist["upland_cotton_lint_yield"].dropna()
        tx_yld_range = list(range(int(float(yv.min())//50)*50,
                                  (int(float(yv.max())//50)+2)*50, 50))
    else:
        tx_yld_range = list(range(250, 851, 50))
            # Calculate Grid C range
    tx_plt_10 = state_data.get("TX", {}).get("periods", {}).get(10, {}).get("plt")
    grid_c_vals = []
    for ab in tx_ab_range:
        for yld in tx_yld_range:
            if tx_plt_10:
                val = tx_plt_10 * 1000 * (1-ab) * yld / 480_000_000
                grid_c_vals.append(val)
    
    matrix_ranges["gridC"] = {
        "min": round(min(grid_c_vals), 2) if grid_c_vals else None,
        "max": round(max(grid_c_vals), 2) if grid_c_vals else None,
        "label": "TX Only"
    }

    # Derive implied US abandonment for each TX abandonment value
    tx_sd = state_data.get("TX", {})
    tx_p10 = tx_sd.get("periods", {}).get(10, {}) if tx_sd else {}
    tx_plt_10 = tx_p10.get("plt") if tx_p10 else None
    tx_yld_10 = tx_p10.get("yld") if tx_p10 else None

    other_wtd_ab = 0.0
    other_plt = 0.0
    for st, sd in state_data.items():
        if st == "TX": 
            continue
        p10 = sd.get("periods", {}).get(10, {})
        if p10 and p10.get("ab") is not None and p10.get("plt") is not None:
            other_wtd_ab += p10["ab"] * p10["plt"]
            other_plt += p10["plt"]

    def us_ab_from_tx(tx_ab):
        if tx_plt_10 is None or other_plt == 0: 
            return tx_ab
        return (tx_ab * tx_plt_10 + other_wtd_ab) / (tx_plt_10 + other_plt)

    # US yield range (50lb steps)
    us_df = cotton[(cotton["geography"]==US_GEO) & (cotton["mkt_year"]>=max_yr-9)]
    if not us_df.empty and "upland_cotton_lint_yield" in us_df.columns:
        yv = us_df["upland_cotton_lint_yield"].dropna()
        us_yld_range = list(range(int(float(yv.min())//50)*50,
                                  (int(float(yv.max())//50)+2)*50, 50))
    else:
        us_yld_range = list(range(700, 1001, 50))

    # US planted for Grid D
    us_plt = None
    if not us_df.empty and "upland_cotton_planted_acreage" in us_df.columns:
        v = us_df["upland_cotton_planted_acreage"].dropna()
        if len(v): 
            us_plt = safe(float(v.iloc[-1]), 0)
    if us_plt is None:
        us_plt_sum = sum(sd["periods"].get(10, {}).get("plt") or 0
                       for sd in state_data.values()
                       if sd["periods"].get(10))
        us_plt = safe(us_plt_sum, 0) if us_plt_sum else None

    # US implied abandonment range rows (derived from TX range)
    us_ab_range = [round(us_ab_from_tx(tx_ab), 4) for tx_ab in tx_ab_range]

    # Calculate Grid D range
    grid_d_vals = []
    for ab in us_ab_range:
        for yld in us_yld_range:
            if us_plt:
                val = us_plt * 1000 * (1-ab) * yld / 480_000_000
                grid_d_vals.append(val)
    
    matrix_ranges["gridD"] = {
        "min": round(min(grid_d_vals), 2) if grid_d_vals else None,
        "max": round(max(grid_d_vals), 2) if grid_d_vals else None,
        "label": "US Total"
    }

    # Get previous year production for comparison
    prev_year_prod = {"tx": None, "us": None}
    tx_data = state_data.get("TX", {})
    if tx_data.get("prev_yr_actual") and tx_data["prev_yr_actual"].get("prd"):
        prev_year_prod["tx"] = tx_data["prev_yr_actual"]["prd"]
    
    # US previous year from US row or sum of states
    us_data = cotton[cotton["geography"]==US_GEO]
    if not us_data.empty:
        us_prev = us_data[us_data["mkt_year"]==prev_yr]
        if not us_prev.empty and "upland_cotton_production" in us_prev.columns:
            prev_year_prod["us"] = safe(us_prev["upland_cotton_production"].iloc[0], 3)
    
    if prev_year_prod["us"] is None:
        # Sum from states
        prev_sum = sum(
            sd.get("prev_yr_actual", {}).get("prd", 0) or 0 
            for sd in state_data.values()
        )
        if prev_sum > 0:
            prev_year_prod["us"] = round(prev_sum / 1000, 3)

    # Build last-year planted defaults for all states
    state_defaults = {}
    for st, sd in state_data.items():
        ly = sd.get("last_yr_actual", {})
        state_defaults[st] = {
            "plt": ly.get("plt"),
            "year": ly.get("year"),
        }

    # Run Monte Carlo simulation
    mc_results = run_monte_carlo_simulation(cotton, state_data)

    return {
        "state_data": state_data,
        "state_defaults": state_defaults,
        "matA": {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in matA.items()},
        "matB": {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in matB.items()},
        "tx_ab_range": tx_ab_range,
        "tx_yld_range": tx_yld_range,
        "us_ab_range": us_ab_range,
        "us_yld_range": us_yld_range,
        "us_plt": us_plt,
        "tx_plt_10yr": tx_plt_10,
        "model_ab": model_ab,
        "periods": PERIODS,
        "period_labels": PERIOD_LABELS,
        "max_yr": max_yr,
        "prev_yr": prev_yr,
        "has_multi_state": len(state_data) > 1,
        "matrix_ranges": matrix_ranges,
        "prev_year_prod": prev_year_prod,
        "monte_carlo": mc_results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. ANALYST SUMMARY - WITH PRODUCTION RANGES AND PREVIOUS YEAR COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
def build_analyst_summary(best_tx, wk_rows, mo_rows, cu_rows, prod, ab, drought):
    """Auto-generate summary text for each section with production ranges."""
    curr_yr = best_tx.get("curr_yr", "?")
    model_ab = best_tx.get("point")
    model_ab_pct = f"{model_ab*100:.1f}%" if model_ab else "N/A"
    lo_pct = f"{(best_tx.get('lo') or 0)*100:.1f}%"
    hi_pct = f"{(best_tx.get('hi') or 0)*100:.1f}%"
    r2_pct = f"{best_tx.get('r2',0)*100:.1f}%"
    n_wk = len(wk_rows)
    latest = wk_rows[-1]["date"] if wk_rows else "N/A"
    latest_lbl = wk_rows[-1]["label"] if wk_rows else "N/A"

    # Find weeks with significant models
    sig_weeks = []
    for row in wk_rows:
        for v in DROUGHT_VARS:
            d = row["vars"].get(v)
            if d and d.get("pvalue") and d["pvalue"] < 0.10:
                sig_weeks.append(f"{row['label']} ({v}, R²={d['r2']*100:.0f}%)")
    sig_txt = ", ".join(sig_weeks[:3]) if sig_weeks else "none yet at p<0.10"

    # Historical TX abandonment stats
    ab_mean = f"{ab.mean()*100:.0f}%"
    ab_std = f"{ab.std()*100:.0f}pp"
    ab_min = f"{ab.min()*100:.0f}%"
    ab_max = f"{ab.max()*100:.0f}%"

    # Production ranges from matrices
    mr = prod.get("matrix_ranges", {})
    matA_range = mr.get("matA", {})
    matB_range = mr.get("matB", {})
    gridC_range = mr.get("gridC", {})
    gridD_range = mr.get("gridD", {})
    
    # Previous year production
    prev_prod = prod.get("prev_year_prod", {})
    prev_tx = prev_prod.get("tx")
    prev_us = prev_prod.get("us")
    
    # Format range text
    def fmt_range(r):
        if r.get("min") is None or r.get("max") is None:
            return "N/A"
        return f"{r['min']:.2f}–{r['max']:.2f} mn bales"
    
    # Production context with ranges
    range_text = f"""Production Scenario Ranges (mn 480-lb bales):
• Matrix A ({matA_range.get('label', 'All Historical')}): {fmt_range(matA_range)}
• Matrix B ({matB_range.get('label', 'TX from Model')}): {fmt_range(matB_range)} 
• Grid C ({gridC_range.get('label', 'TX Only')}): {fmt_range(gridC_range)}
• Grid D ({gridD_range.get('label', 'US Total')}): {fmt_range(gridD_range)}"""

    # Previous year comparison
    prev_yr = prod.get("prev_yr", curr_yr - 1)
    prev_comparison = ""
    if prev_tx is not None:
        prev_comparison += f"\n• TX {prev_yr} Actual: {prev_tx:.2f} mn bales"
    if prev_us is not None:
        prev_comparison += f"\n• US {prev_yr} Actual: {prev_us:.2f} mn bales"
    
    if not prev_comparison:
        prev_comparison = f"\n• {prev_yr} actual production data not available in current dataset."

    # Monte Carlo results
    mc = prod.get("monte_carlo")
    mc_text = ""
    if mc:
        mc_text = f"""\n\nMonte Carlo Simulation Results ({mc.get('n_simulations', 10000)} runs):
• TX Production: mean={mc['tx']['production']['mean']:.2f}, median={mc['tx']['production']['median']:.2f}, 90% CI [{mc['tx']['production']['ci_5']:.2f}–{mc['tx']['production']['ci_95']:.2f}] mn bales
• US Production: mean={mc['us']['production']['mean']:.2f}, median={mc['us']['production']['median']:.2f}, 90% CI [{mc['us']['production']['ci_5']:.2f}–{mc['us']['production']['ci_95']:.2f}] mn bales"""

    sections = {
        "seasonality": (
            f"Current season ({curr_yr}): {n_wk} weeks of drought data available through {latest}. "
            f"Analog years most similar to current season drought pattern shown. "
            f"Compare current trajectory against analogs to assess seasonal outlook."
        ),
        "weekly": (
            f"Weekly model uses single-week drought snapshots as predictor. "
            f"Significant weeks (p<0.10): {sig_txt}. "
            f"Best overall: {best_tx.get('variable','—')} at {latest_lbl} (R²={r2_pct}). "
            f"Wide confidence intervals are expected with a single predictor."
        ),
        "monthly": (
            f"Monthly model averages weekly drought readings within each calendar month, "
            f"smoothing week-to-week volatility. Compare against weekly model to assess stability."
        ),
        "cumulative": (
            f"Cumulative model uses running average drought from Apr W1 through each week. "
            f"Becomes progressively more stable as the season accumulates. "
            f"At season end, reflects full-season drought stress."
        ),
        "production": (
            f"Best regression prediction: TX abandonment = {model_ab_pct} "
            f"({CI_LEVEL*100:.0f}% CI: {lo_pct}–{hi_pct}) via {best_tx.get('variable','—')} "
            f"{best_tx.get('model','—')} model at {latest_lbl}. "
            f"Historical TX abandonment: mean={ab_mean}, std={ab_std}, range={ab_min}–{ab_max}.\n\n"
            f"{range_text}"
            f"{prev_comparison}"
            f"{mc_text}"
        ),
    }
    return sections
    # ═══════════════════════════════════════════════════════════════════════════
# 7. HTML - WITH FIXED SEASONALITY CHARTS AND TABLES
# ═══════════════════════════════════════════════════════════════════════════
def make_html(wk_rows, mo_rows, cu_rows, slines, prod, best_tx, summary, cotton_ab, drought,
              us_wk, us_mo, us_cu, us_slines, us_best, ci_pct):

    curr_yr = wk_rows[0]["curr_yr"] if wk_rows else "?"
    ab = cotton_ab
    run_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    n_wk = len(wk_rows)
    latest_wk = wk_rows[-1]["date"] if wk_rows else "—"
    model_ab_pct = f"{best_tx.get('point',0)*100:.1f}%" if best_tx.get("point") else "N/A"

    j_us_wk = jd(us_wk)
    j_us_mo = jd(us_mo)
    j_us_cu = jd(us_cu)
    j_us_sl = jd(us_slines) if us_slines is not None else 'null'
    j_us_btx = jd(us_best)
    
    # Monte Carlo results JSON
    mc_json = jd(prod.get("monte_carlo")) if prod.get("monte_carlo") else 'null'
    
    # Previous year production
    prev_prod_json = jd(prod.get("prev_year_prod"))
    
    # Matrix ranges
    matrix_ranges_json = jd(prod.get("matrix_ranges", {}))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>TX Cotton – Drought Predictor</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,Arial,sans-serif;background:#0d1117;color:#e2e8f0;min-height:100vh}}
code{{background:#1e2a3a;padding:2px 5px;border-radius:3px;font-size:.84em}}
.hdr{{background:linear-gradient(135deg,#0f2a1a 0%,#1a3050 60%,#0f2a1a 100%);
  padding:13px 24px;border-bottom:2px solid #2d6a4f;
  display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:8px}}
.hdr h1{{font-size:1.2rem;font-weight:700;color:#f0fff4}}
.hdr p{{font-size:.73rem;color:#90a4ae;margin-top:3px}}
.badge{{background:#1a3050;border:1px solid #2d6a4f;border-radius:20px;padding:4px 11px;font-size:.7rem;color:#68d391;white-space:nowrap;align-self:center}}
.meta{{background:#111820;padding:5px 24px;font-size:.7rem;color:#607080;border-bottom:1px solid #1e2a3a;display:flex;gap:14px;flex-wrap:wrap}}
.meta b{{color:#68d391}}
.tabs{{display:flex;background:#0d1117;border-bottom:2px solid #1e2a3a;padding:0 14px;overflow-x:auto}}
.tab{{padding:8px 13px;cursor:pointer;font-size:.79rem;color:#607080;border-bottom:3px solid transparent;margin-bottom:-2px;white-space:nowrap;transition:color .2s}}
.tab:hover{{color:#c0d0e0}}
.tab.active{{color:#68d391;border-bottom-color:#68d391;font-weight:600}}
.panel{{display:none;padding:13px 16px 32px}}
.panel.active{{display:block}}
.st{{font-size:.87rem;font-weight:600;color:#68d391;margin:13px 0 3px}}
.st:first-child{{margin-top:0}}
.sn{{font-size:.71rem;color:#607080;line-height:1.5;margin-bottom:7px}}
.ctrls{{display:flex;gap:9px;flex-wrap:wrap;margin-bottom:9px;align-items:flex-end}}
.ctrl{{display:flex;flex-direction:column;gap:2px;font-size:.71rem;color:#90a4ae}}
.ctrl span{{font-size:.65rem;color:#607080}}
select,input[type=number],input[type=text]{{background:#1a2535;border:1px solid #2d3e50;color:#e2e8f0;padding:4px 7px;border-radius:5px;font-size:.71rem;cursor:pointer}}
input[type=number]{{width:110px}}
.btn{{background:#1a4030;border:1px solid #2d6a4f;color:#68d391;padding:4px 11px;border-radius:5px;font-size:.71rem;cursor:pointer}}
.btn:hover{{background:#1e4a38}}
.btn-pdf{{background:#1a3050;border:1px solid #4a6080;color:#90c4ff;padding:5px 14px;border-radius:5px;font-size:.76rem;cursor:pointer;display:flex;align-items:center;gap:6px}}
.btn-pdf:hover{{background:#1e3a60}}
/* Charts */
.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px}}
@media(max-width:680px){{.chart-grid{{grid-template-columns:1fr}}}}
.chart-card{{background:#111820;border:1px solid #1e2a3a;border-radius:8px;padding:9px}}
.chart-card h3{{font-size:.75rem;font-weight:600;margin-bottom:3px}}
/* Scatter */
.sc-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px}}
@media(max-width:620px){{.sc-grid{{grid-template-columns:1fr}}}}
.sc-card{{background:#111820;border:1px solid #1e2a3a;border-radius:8px;padding:9px}}
.sc-card h3{{font-size:.75rem;font-weight:600;margin-bottom:3px}}
/* Tables */
.tw{{overflow-x:auto;border-radius:7px;border:1px solid #1e2a3a;margin-bottom:7px}}
table{{width:100%;border-collapse:collapse;font-size:.7rem;white-space:nowrap}}
/* FIXED: Header styling to prevent overlap */
.tw thead{{position:sticky;top:0;z-index:10}}
.tw thead tr:first-child th{{background:#0e1d2e;color:#7fb3d3;font-weight:600;padding:8px 7px;border-bottom:2px solid #2d4060;position:sticky;top:0;z-index:11}}
.tw thead tr:last-child th{{background:#0a1520;color:#607080;font-weight:400;font-size:.65rem;padding:6px 7px;border-bottom:2px solid #2d4060;position:sticky;top:32px;z-index:10}}
th.lft,td.lft{{text-align:left;padding-left:8px}}
th.dvh{{background:#0a1f14!important}}
th.dvs{{background:#061410!important}}
th.bvh{{background:#150d25!important}}
th.bvs{{background:#0d0820!important}}
tbody tr:nth-child(even){{background:#0e1420}}
tbody tr:nth-child(odd){{background:#0d1117}}
tbody tr:hover{{background:#132030}}
td{{padding:4px 7px;text-align:center;vertical-align:middle;border-bottom:1px solid #141e28}}
td.lft{{font-weight:600;color:#c8d8e8}}
td.pt{{color:#f6e05e;font-weight:600}}
td.ci{{color:#607080;font-size:.65rem}}
td.bv{{font-weight:600;font-size:.67rem}}
td.bp{{color:#c4a0f0;font-weight:700}}
td.bc{{color:#9070c0;font-size:.65rem}}
.r2w{{display:inline-flex;align-items:center;gap:3px}}
.r2bg{{width:26px;height:4px;background:#1e2a3a;border-radius:2px}}
.r2f{{height:4px;border-radius:2px}}
/* Matrices - FIXED header overlap */
.mat-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px}}
@media(max-width:700px){{.mat-grid{{grid-template-columns:1fr}}}}
.mat-card{{background:#111820;border:1px solid #1e2a3a;border-radius:8px;padding:10px;overflow-x:auto;max-height:500px;overflow-y:auto}}
.mat-card h3{{font-size:.77rem;font-weight:600;margin-bottom:3px}}
.mat-card .sub{{font-size:.67rem;color:#607080;margin-bottom:7px}}
.mtbl{{border-collapse:collapse;font-size:.68rem;white-space:nowrap;width:100%}}
.mtbl th,.mtbl td{{padding:4px 8px;text-align:right;border:1px solid #1e2a3a}}
/* FIXED: Sticky headers for matrices */
.mtbl thead th{{background:#0e1d2e;color:#90a4ae;font-weight:600;position:sticky;top:0;z-index:5}}
.mtbl thead th:first-child{{text-align:left;position:sticky;left:0;z-index:6;background:#0e1d2e}}
.mtbl tbody th{{background:#111820;color:#90a4ae;font-weight:600;text-align:left;position:sticky;left:0;z-index:4}}
.mtbl tbody tr:hover td,.mtbl tbody tr:hover th{{background:#132030}}
.cell-model{{background:#0a1f14!important;color:#68d391!important;font-weight:700}}
.cell-hi{{background:#1a2a14!important;color:#9ae89a!important}}
.row-model th{{color:#68d391!important;background:#061410!important}}
/* Best banner */
.best-banner{{background:#0a1f14;border:1px solid #2d6a4f;border-radius:7px;padding:7px 13px;margin-bottom:11px;font-size:.76rem;line-height:1.6}}
/* Planted area input */
.plant-box{{background:#111820;border:1px solid #2d6a4f;border-radius:8px;padding:12px;max-width:420px;margin-bottom:14px}}
.plant-box label{{font-size:.77rem;color:#90a4ae;display:block;margin-bottom:6px;font-weight:600}}
.plant-box .hint{{font-size:.67rem;color:#607080;margin-top:5px;line-height:1.5}}
.plant-box .saved{{font-size:.67rem;color:#68d391;margin-top:4px}}
/* Analyst summary */
.summary-box{{background:#111820;border:1px solid #1e2a3a;border-radius:8px;padding:12px;margin-bottom:14px}}
.summary-box h3{{font-size:.8rem;color:#68d391;font-weight:600;margin-bottom:6px}}
.summary-box textarea{{width:100%;background:#0d1117;border:1px solid #2d3e50;color:#e2e8f0;
  padding:8px;border-radius:5px;font-size:.76rem;line-height:1.6;resize:vertical;min-height:80px}}
/* Monte Carlo section */
.mc-box{{background:#0a1520;border:1px solid #1e3a5f;border-radius:8px;padding:12px;margin-bottom:14px}}
.mc-box h3{{font-size:.8rem;color:#90c4ff;font-weight:600;margin-bottom:8px}}
.mc-stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;font-size:.72rem}}
.mc-stat{{background:#111820;padding:8px;border-radius:5px;border:1px solid #1e2a3a}}
.mc-stat .label{{color:#607080;font-size:.65rem}}
.mc-stat .value{{color:#68d391;font-weight:600;font-size:.8rem}}
/* Year selector for seasonality */
.year-selector{{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0;max-height:120px;overflow-y:auto;padding:4px;background:#0a1520;border-radius:5px;border:1px solid #1e2a3a}}
.year-chip{{padding:3px 8px;border-radius:12px;font-size:.65rem;cursor:pointer;border:1px solid #2d3e50;background:#1a2535;color:#90a4ae;transition:all .2s}}
.year-chip:hover{{background:#1e3a50}}
.year-chip.active{{background:#2d6a4f;border-color:#68d391;color:#68d391}}
.year-chip.current{{background:#0f4a2a;border-color:#68d391;color:#f0fff4;font-weight:600}}
.year-chip.last{{background:#1a3050;border-color:#90c4ff;color:#90c4ff}}
.year-chip.last2{{background:#1a2040;border-color:#c4a0f0;color:#c4a0f0}}
.year-chip.analog{{background:#1a2a14;border-color:#f6e05e;color:#f6e05e}}
/* Diag */
.dg-grid{{display:grid;grid-template-columns:1fr 1fr;gap:11px}}
@media(max-width:620px){{.dg-grid{{grid-template-columns:1fr}}}}
.dg-card{{background:#111820;border:1px solid #1e2a3a;border-radius:8px;padding:9px}}
.dg-card h3{{font-size:.75rem;font-weight:600;margin-bottom:5px}}
/* About */
.about{{max-width:620px;font-size:.8rem;line-height:1.75;color:#b0c0d0}}
.about h3{{color:#68d391;font-size:.84rem;margin:12px 0 3px}}
.about p{{margin-bottom:6px}}
.nbox{{background:#1a2535;border-left:3px solid #68d391;padding:7px 10px;border-radius:0 5px 5px 0;font-size:.73rem;color:#90c4ae;margin:7px 0;line-height:1.6}}
/* Print styles */
@media print{{
  body{{background:white;color:black;font-size:10pt}}
  .tabs,.hdr .badge,.btn,.btn-pdf,.ctrls,.plant-box .hint,.year-selector{{display:none!important}}
  .panel{{display:block!important;page-break-after:always;padding:8px}}
  .hdr{{background:none;border-bottom:2px solid #2d6a4f;padding:6px 0}}
  .hdr h1{{color:black;font-size:14pt}}
  .hdr p,.meta{{color:#444;font-size:8pt}}
  .st{{color:#2d6a4f}}
  table,svg{{break-inside:avoid}}
  .mat-grid,.chart-grid,.sc-grid,.dg-grid{{grid-template-columns:1fr 1fr}}
  .mtbl th,.mtbl td{{border-color:#ccc;color:black}}
  .mtbl thead th{{background:#e8f4e8!important;color:black!important}}
  .cell-model{{background:#d4edda!important;color:darkgreen!important}}
  tbody tr:nth-child(even){{background:#f9f9f9!important}}
}}
</style>
</head>
<body>
<div class="hdr">
  <div>
    <h1>🌾 TX Cotton · Drought Abandonment Predictor + Production Estimator · MY {curr_yr}/{str(int(curr_yr)+1)[-2:]}</h1>
    <p>{n_wk} weeks available through {latest_wk} · Best model: {best_tx.get('variable','—')} ({best_tx.get('model','—')}) R²={best_tx.get('r2',0)*100:.1f}% → TX abandonment {model_ab_pct}</p>
  </div>
  <div style="display:flex;gap:8px;align-items:center">
    <button class="btn-pdf" onclick="printPDF()">🖨 Print / PDF</button>
    <div class="badge">Run: {run_time}</div>
  </div>
</div>
<div class="meta">
  <span>Cotton: <b>{int(cotton_ab.index.min())}–{int(cotton_ab.index.max())}</b></span>
  <span>Drought: <b>{drought['Week'].min().date()} – {drought['Week'].max().date()}</b></span>
  <span>CI: <b>{ci_pct}%</b></span>
  <span>Min train: <b>{MIN_TRAIN_YRS} yrs</b></span>
</div>

<div class="tabs">
  <div class="tab active" onclick="showTab('t1')">🌡 TX Drought</div>
  <div class="tab"        onclick="showTab('t2')">🌍 US Drought</div>
  <div class="tab"        onclick="showTab('t3')">📊 Weekly</div>
  <div class="tab"        onclick="showTab('t4')">📅 Monthly</div>
  <div class="tab"        onclick="showTab('t5')">📈 Cumulative</div>
  <div class="tab"        onclick="showTab('t6')">🌽 Production</div>
  <div class="tab"        onclick="showTab('t7')">📋 Summary</div>
  <div class="tab"        onclick="showTab('t8')">🔬 Diagnostics</div>
</div>

<!-- ═══ TAB 1: TX SEASONALITY ════════════════════════════════════════════════ -->
<div id="t1" class="panel active">
  <div class="st">Drought Seasonality — Line Charts ({curr_yr} vs Historical)</div>
  <div class="sn">Each variable shown as a line chart. Grey = all historical years. 
    <b style="color:#68d391">Green</b> = current year {curr_yr}. 
    <b style="color:#90c4ff">Blue</b> = {curr_yr-1} (last year). 
    <b style="color:#c4a0f0">Purple</b> = {curr_yr-2} (2 years ago). 
    <b style="color:#f6e05e">Yellow dashed</b> = top 5 analog years. 
    Click years below to toggle visibility.</div>
  
  <div class="ctrls">
    <div class="ctrl"><span>Geography</span>
      <div style="display:flex;border:1px solid #2d3e50;border-radius:5px;overflow:hidden">
        <button class="btn" id="sl-tx-btn" style="border-radius:0;background:#1a4030;color:#68d391" onclick="setSlGeo('TX')">TX</button>
        <button class="btn" id="sl-us-btn" style="border-radius:0;background:#1a2535;color:#90a4ae" onclick="setSlGeo('US')">US</button>
      </div>
    </div>
    <div class="ctrl"><span>Show</span>
      <select id="sl-var" onchange="drawSeasonCharts()">
        <option value="ALL">All 4 variables (2×2)</option>
        <option value="D1-D4">D1-D4 only</option>
        <option value="D2-D4">D2-D4 only</option>
        <option value="D3-D4">D3-D4 only</option>
        <option value="D4">D4 only</option>
      </select>
    </div>
  </div>
  
  <!-- Year selector chips -->
  <div class="ctrl"><span>Toggle Years (click to show/hide):</span></div>
  <div id="year-chips" class="year-selector"></div>
  
  <div id="sg" class="chart-grid"></div>
  
  <div class="st">Last 6 Weeks — Current vs Analogs & Previous Years</div>
  <div class="sn">Current season last 6 weeks, plus same 6-week period from previous 2 years and top 5 analog years.</div>
  <div class="ctrl" style="margin-bottom:8px"><span>Variable</span>
    <select id="l6-var" onchange="drawLast6()">
      <option value="D1-D4">D1-D4</option>
      <option value="D2-D4">D2-D4</option>
      <option value="D3-D4">D3-D4</option>
      <option value="D4">D4</option>
    </select>
  </div>
  <div class="tw"><div id="l6tbl"></div></div>
  <div class="summary-box" style="margin-top:12px">
    <h3>📝 Analyst Notes — Seasonality</h3>
    <textarea id="sum-t1">{summary['seasonality']}</textarea>
  </div>
</div>
<!-- ═══ TAB 2: US DROUGHT SEASONALITY ══════════════════════════════════ -->
<div id="t2" class="panel">
  <div class="st">US Cotton Drought Seasonality — Line Charts</div>
  <div class="sn">% of US cotton area in each drought category. Same analog year analysis as TX tab. Upload <code>data/drought_US.csv</code> to enable.</div>
  <div class="ctrls">
    <div class="ctrl"><span>Show</span>
      <select id="sl-var-us" onchange="drawSeasonChartsUS()">
        <option value="ALL">All 4 variables (2×2)</option>
        <option value="D1-D4">D1-D4 only</option>
        <option value="D2-D4">D2-D4 only</option>
        <option value="D3-D4">D3-D4 only</option>
        <option value="D4">D4 only</option>
      </select>
    </div>
  </div>
  <div id="year-chips-us" class="year-selector"></div>
  <div id="sg-us" class="chart-grid"></div>
  <div class="st">Last 6 Weeks — US Drought (Latest Week Comparison)</div>
  <div class="ctrls">
    <div class="ctrl"><span>Variable</span>
      <select id="l6-var-us" onchange="drawLast6US()">
        <option value="D1-D4">D1-D4</option>
        <option value="D2-D4">D2-D4</option>
        <option value="D3-D4">D3-D4</option>
        <option value="D4">D4</option>
      </select>
    </div>
  </div>
  <div class="tw"><div id="l6tbl-us"></div></div>
  <div class="summary-box" style="margin-top:12px">
    <h3>📝 Analyst Notes — US Drought Seasonality</h3>
    <textarea id="sum-us-seas">US drought seasonality analysis. Compare with TX patterns to assess relative drought conditions across the cotton belt.</textarea>
  </div>
</div>

<!-- ═══ TAB 3: WEEKLY ═════════════════════════════════════════════════════ -->
<div id="t3" class="panel">
  <div class="ctrls">
    <div class="ctrl"><span>Geography</span>
      <div style="display:flex;border:1px solid #2d3e50;border-radius:5px;overflow:hidden">
        <button id="wk-tx-btn" class="btn" style="border-radius:0;border:none;background:#1a4030;color:#68d391" onclick="setGeo('wk','TX')">TX</button>
        <button id="wk-us-btn" class="btn" style="border-radius:0;border:none;background:#1a2535;color:#90a4ae" onclick="setGeo('wk','US')">US</button>
      </div>
    </div>
  </div>
  <div id="wk-st" class="st">Weekly Model — TX</div>
  <div class="sn">Predictor = that exact week's drought %. Latest available week shown. ★ = current season ({ci_pct}% CI).</div>
  <div class="sc-grid" id="sc-w"></div>
  <div class="st">Weekly Model — Prediction Table</div>
  <div class="tw"><div id="tbl-w"></div></div>
  <div style="font-size:.65rem;color:#405060;margin-top:4px">R²: <span style="color:#68d391">green p&lt;0.05</span> · <span style="color:#f6e05e">yellow p&lt;0.10</span> · <span style="color:#607080">grey n.s.</span></div>
  <div class="summary-box" style="margin-top:12px">
    <h3>📝 Analyst Notes — Weekly Model</h3>
    <textarea id="sum-t2">{summary['weekly']}</textarea>
  </div>
</div>

<!-- ═══ TAB 4: MONTHLY ════════════════════════════════════════════════════ -->
<div id="t4" class="panel">
  <div class="ctrls">
    <div class="ctrl"><span>Geography</span>
      <div style="display:flex;border:1px solid #2d3e50;border-radius:5px;overflow:hidden">
        <button id="mo-tx-btn" class="btn" style="border-radius:0;border:none;background:#1a4030;color:#68d391" onclick="setGeo('mo','TX')">TX</button>
        <button id="mo-us-btn" class="btn" style="border-radius:0;border:none;background:#1a2535;color:#90a4ae" onclick="setGeo('mo','US')">US</button>
      </div>
    </div>
  </div>
  <div id="mo-st" class="st">Monthly Model — TX</div>
  <div class="sn">Predictor = average drought for that calendar month. Latest available month shown.</div>
  <div class="sc-grid" id="sc-m"></div>
  <div class="st">Monthly Model — Prediction Table</div>
  <div class="tw"><div id="tbl-m"></div></div>
  <div class="summary-box" style="margin-top:12px">
    <h3>📝 Analyst Notes — Monthly Model</h3>
    <textarea id="sum-t3">{summary['monthly']}</textarea>
  </div>
</div>

<!-- ═══ TAB 5: CUMULATIVE ═════════════════════════════════════════════════ -->
<div id="t5" class="panel">
  <div class="ctrls">
    <div class="ctrl"><span>Geography</span>
      <div style="display:flex;border:1px solid #2d3e50;border-radius:5px;overflow:hidden">
        <button id="cu-tx-btn" class="btn" style="border-radius:0;border:none;background:#1a4030;color:#68d391" onclick="setGeo('cu','TX')">TX</button>
        <button id="cu-us-btn" class="btn" style="border-radius:0;border:none;background:#1a2535;color:#90a4ae" onclick="setGeo('cu','US')">US</button>
      </div>
    </div>
  </div>
  <div id="cu-st" class="st">Cumulative Model — TX</div>
  <div class="sn">Predictor = running average drought from Apr W1 through that week.</div>
  <div class="sc-grid" id="sc-c"></div>
  <div class="st">Cumulative Model — Prediction Table</div>
  <div class="tw"><div id="tbl-c"></div></div>
  <div class="summary-box" style="margin-top:12px">
    <h3>📝 Analyst Notes — Cumulative Model</h3>
    <textarea id="sum-t4">{summary['cumulative']}</textarea>
  </div>
</div>

<!-- ═══ TAB 6: PRODUCTION ════════════════════════════════════════════════ -->
<div id="t6" class="panel">
  <div id="best-banner" class="best-banner"></div>

  <!-- Monte Carlo Results -->
  <div id="mc-results" class="mc-box" style="display:none">
    <h3>🎲 Monte Carlo Simulation Results</h3>
    <div id="mc-stats" class="mc-stats"></div>
  </div>

  <!-- Planted Area Inputs — all states, pre-filled from last year -->
  <div class="plant-box" style="max-width:100%">
    <label>🌱 Planted Area by State (1,000 acres) — Pre-filled with last year actuals. Edit any value then click Save.</label>
    <div id="planted-inputs" style="display:flex;flex-wrap:wrap;gap:8px;margin:8px 0;max-height:220px;overflow-y:auto;padding:4px"></div>
    <div style="display:flex;gap:8px;align-items:center;margin-top:6px">
      <button class="btn" onclick="savePlanted()">💾 Save &amp; Update All Grids</button>
      <button class="btn" style="background:#1a2535;border-color:#4a5060;color:#90a4ae" onclick="resetPlanted()">↺ Reset to Last Year Actuals</button>
    </div>
    <div class="saved" id="saved-note"></div>
    <div class="hint">Values persist in your browser. Reset returns to last year USDA actuals.</div>
  </div>

  <!-- Matrix A & B -->
  <div class="st">Production Scenario Matrices — US Total (mn 480-lb bales)</div>
  <div class="sn">
    Rows = historical avg abandonment period · Cols = historical avg yield period · Cell = sum of all states' production.<br>
    <b>Matrix A</b>: all states use their own historical avgs &nbsp;·&nbsp;
    <b>Matrix B</b>: TX abandonment fixed at model prediction <span style="color:#68d391">★</span>, other states unchanged.
  </div>
  <div class="mat-grid">
    <div class="mat-card">
      <h3>Matrix A — All Historical</h3>
      <div class="sub" id="mA-sub">Rows = abandonment period · Cols = yield period</div>
      <div id="mA"></div>
    </div>
    <div class="mat-card">
      <h3>Matrix B — TX from Model ★</h3>
      <div class="sub" id="mB-sub">TX abandonment = regression prediction</div>
      <div id="mB"></div>
    </div>
  </div>

  <!-- Grid C & D -->
  <div class="st">Sensitivity Grids — Abandonment × Yield (last 10-yr actual range)</div>
  <div class="sn">
    Rows = abandonment in 5% increments · Cols = yield in 50 lb/acre steps · Values = mn 480-lb bales.<br>
    <span style="color:#68d391">★ Highlighted row</span> = current model prediction.
  </div>
  <div class="mat-grid">
    <div class="mat-card">
      <h3>Grid C — Texas Only</h3>
      <div class="sub">Uses TX planted area entered above</div>
      <div id="gC"></div>
    </div>
    <div class="mat-card">
      <h3>Grid D — US Total</h3>
      <div class="sub">TX abandonment from model; other states 10-yr avg; US planted area from data</div>
      <div id="gD"></div>
    </div>
  </div>
  <div class="summary-box" style="margin-top:12px">
    <h3>📝 Analyst Notes — Production</h3>
    <textarea id="sum-t5">{summary['production']}</textarea>
  </div>
</div>

<!-- ═══ TAB 7: SUMMARY ════════════════════════════════════════════════════ -->
<div id="t7" class="panel">
  <div class="st">Analyst Summary — Full Report</div>
  <div class="sn">Auto-generated from current data. Edit any section before printing.</div>
  <div style="margin-bottom:10px;display:flex;gap:8px">
    <button class="btn" onclick="refreshSummary()">↺ Refresh from Data</button>
    <button class="btn-pdf" onclick="printPDF()">🖨 Print 5-Page PDF</button>
  </div>
  <div class="summary-box">
    <h3>1. Drought Seasonality</h3>
    <textarea id="ps1" rows="4"></textarea>
  </div>
  <div class="summary-box">
    <h3>2. Abandonment Model Predictions (Weekly / Monthly / Cumulative)</h3>
    <textarea id="ps2" rows="5"></textarea>
  </div>
  <div class="summary-box">
    <h3>3. Best Model & Current Prediction</h3>
    <textarea id="ps3" rows="4"></textarea>
  </div>
  <div class="summary-box">
    <h3>4. Production Scenarios</h3>
    <textarea id="ps4" rows="5"></textarea>
  </div>
  <div class="summary-box">
    <h3>5. Key Risks & Uncertainties</h3>
    <textarea id="ps5" rows="4"></textarea>
  </div>
</div>

<!-- ═══ TAB 8: DIAGNOSTICS ═══════════════════════════════════════════════ -->
<div id="t8" class="panel">
  <div class="st">R² Comparison — All Three Models</div>
  <div class="sn"><span style="color:#68d391">■ Weekly</span> · <span style="color:#f6e05e">■ Monthly</span> · <span style="color:#90a0ff">■ Cumulative</span></div>
  <div class="dg-grid" id="dg"></div>
</div>

<script>
// ── Data ─────────────────────────────────────────────────────────────────
var WK={jd(wk_rows)};var MO={jd(mo_rows)};var CU={jd(cu_rows)};
var WK_US={j_us_wk};var MO_US={j_us_mo};var CU_US={j_us_cu};
var SL={jd(slines)};var SL_US={j_us_sl};
var BTX_US={j_us_btx};
var PROD={jd(prod)};var BTX={jd(best_tx)};
var SUMMARY={jd(summary)};
var CI_PCT="{ci_pct}%";
var VARS=["D1-D4","D2-D4","D3-D4","D4"];
var VC={{"D1-D4":"#68d391","D2-D4":"#f6e05e","D3-D4":"#fc8181","D4":"#d6bcfa"}};
var VBG={{"D1-D4":"#0a1f14","D2-D4":"#1f1a08","D3-D4":"#1f0a0a","D4":"#130a1f"}};
var MC={{"weekly":"#68d391","monthly":"#f6e05e","cumulative":"#90a0ff"}};

// FIXED: Color scheme for seasonality charts
var COLORS = {{
  curr: "#68d391",      // Current year - green
  last1: "#90c4ff",    // Last year - blue  
  last2: "#c4a0f0",    // 2 years ago - purple
  analog: ["#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff", "#ff9f43"], // Analog years
  hist: "#2a3848"      // Other historical - grey
}};

var ANALOGS={{}};var ANALOGS_US={{}};var GEO_STATE={{"wk":"TX","mo":"TX","cu":"TX"}};

// Year visibility state
var VISIBLE_YEARS = {{}};
var VISIBLE_YEARS_US = {{}};

// ── Tabs ──────────────────────────────────────────────────────────────────
function showTab(id){{
  var ids=["t1","t2","t3","t4","t5","t6","t7","t8"];
  ids.forEach(function(t,i){{
    document.querySelectorAll(".tab")[i].classList.toggle("active",t===id);
    document.getElementById(t).classList.toggle("active",t===id);
  }});
  if(id==="t6")renderProd();
  if(id==="t7")refreshSummary();
  if(id==="t8")drawDiag();
}}

// ── SVG helpers ───────────────────────────────────────────────────────────
var NS="http://www.w3.org/2000/svg";
function mel(t,a){{var e=document.createElementNS(NS,t);Object.keys(a).forEach(function(k){{e.setAttribute(k,a[k]);}});return e;}}
function msvg(w,h){{var s=document.createElementNS(NS,"svg");s.setAttribute("width","100%");s.setAttribute("viewBox","0 0 "+w+" "+h);s.setAttribute("style","font-family:inherit;display:block;overflow:visible");return s;}}
function tt(el,txt){{var t=document.createElementNS(NS,"title");t.textContent=txt;el.appendChild(t);return el;}}
function mtx(txt,a,fill){{var e=mel("text",a);e.textContent=txt;if(fill)e.setAttribute("fill",fill);return e;}}
function sc(d0,d1,r0,r1){{return function(v){{return d1===d0?r0:(r0+(v-d0)/(d1-d0)*(r1-r0));}}}}
function cl(v,lo,hi){{return Math.max(lo,Math.min(hi,v));}}
function lc(c1,c2,t){{
  var r1=parseInt(c1.slice(1,3),16),g1=parseInt(c1.slice(3,5),16),b1=parseInt(c1.slice(5,7),16);
  var r2=parseInt(c2.slice(1,3),16),g2=parseInt(c2.slice(3,5),16),b2=parseInt(c2.slice(5,7),16);
  return "rgb("+Math.round(r1+(r2-r1)*t)+","+Math.round(g1+(g2-g1)*t)+","+Math.round(b1+(b2-b1)*t)+")";
}}
function sp(cx,cy,or_,ir,pts){{
  var d="";for(var i=0;i<pts*2;i++){{var r=i%2===0?or_:ir,a=(Math.PI/pts)*i-Math.PI/2;d+=(i===0?"M":"L")+(cx+r*Math.cos(a)).toFixed(2)+","+(cy+r*Math.sin(a)).toFixed(2);}}return d+"Z";
}}

// ── Geography state ────────────────────────────────────────────────────────
var SL_GEO = "TX";
function getSlData(){{ return SL_GEO==="US"?SL_US:SL; }}
function getWkData(){{ return GEO_STATE.wk==="US"?WK_US:WK; }}
function getMoData(){{ return GEO_STATE.mo==="US"?MO_US:MO; }}
function getCuData(){{ return GEO_STATE.cu==="US"?CU_US:CU; }}

function setSlGeo(geo){{
  SL_GEO=geo;
  setGeoBtn("sl-tx-btn","sl-us-btn",geo);
  initYearChips();
  drawSeasonCharts();
  drawLast6();
}}

function setGeo(tab,geo){{
  GEO_STATE[tab]=geo;
  var on="#1a4030",off="#1a2535",onT="#68d391",offT="#90a4ae";
  var txBtn=document.getElementById(tab+"-tx-btn");
  var usBtn=document.getElementById(tab+"-us-btn");
  if(txBtn){{txBtn.style.background=geo==="TX"?on:off;txBtn.style.color=geo==="TX"?onT:offT;}}
  if(usBtn){{usBtn.style.background=geo==="US"?on:off;usBtn.style.color=geo==="US"?onT:offT;}}
  var rowsTX={{"wk":WK,"mo":MO,"cu":CU}};
  var rowsUS={{"wk":WK_US,"mo":MO_US,"cu":CU_US}};
  var rows=geo==="US"?rowsUS[tab]:rowsTX[tab];
  var st=document.getElementById(tab+"-st");
  if(st)st.textContent={{"wk":"Weekly","mo":"Monthly","cu":"Cumulative"}}[tab]+" Model — "+geo;
  var sc_ids={{"wk":"sc-w","mo":"sc-m","cu":"sc-c"}};
  var tbl_ids={{"wk":"tbl-w","mo":"tbl-m","cu":"tbl-c"}};
  drawScSet(sc_ids[tab],rows||[]);
  drawTblSet(tbl_ids[tab],rows||[],"label");
}}

function setGeoBtn(txId,usId,geo){{
  var tx=document.getElementById(txId),us=document.getElementById(usId);
  if(tx){{tx.style.background=geo==="TX"?"#1a4030":"#1a2535";tx.style.color=geo==="TX"?"#68d391":"#90a4ae";}}
  if(us){{us.style.background=geo==="US"?"#1a4030":"#1a2535";us.style.color=geo==="US"?"#68d391":"#90a4ae";}}
}}
// ── Year Chips for Seasonality ────────────────────────────────────────────
function initYearChips(){{
  var slData = getSlData();
  if(!slData || !slData.variables) return;
  
  var container = document.getElementById("year-chips");
  var containerUS = document.getElementById("year-chips-us");
  
  // Build chips for TX
  if(container){{
    container.innerHTML = '';
    var allYears = slData.all_hist_years || [];
    var currYr = slData.curr_yr;
    var last1 = slData.last_yr_1;
    var last2 = slData.last_yr_2;
    
    // Initialize visibility
    if(Object.keys(VISIBLE_YEARS).length === 0){{
      // Default: show current, last 2 years, and analogs
      VISIBLE_YEARS[currYr] = true;
      VISIBLE_YEARS[last1] = true;
      VISIBLE_YEARS[last2] = true;
    }}
    
    // Current year chip
    var chip = createYearChip(currYr, 'current', true, true);
    container.appendChild(chip);
    
    // Last 2 years
    chip = createYearChip(last1, 'last', VISIBLE_YEARS[last1] !== false, false);
    chip.onclick = function() {{ toggleYear(last1, this); }};
    container.appendChild(chip);
    
    chip = createYearChip(last2, 'last2', VISIBLE_YEARS[last2] !== false, false);
    chip.onclick = function() {{ toggleYear(last2, this); }};
    container.appendChild(chip);
    
    // All other years
    allYears.forEach(function(yr){{
      if(yr !== currYr && yr !== last1 && yr !== last2){{
        var isAnalog = false;
        VARS.forEach(function(v){{
          var vd = slData.variables[v];
          if(vd && vd.analogs && vd.analogs.indexOf(yr) >= 0) isAnalog = true;
        }});
        var cls = isAnalog ? 'analog' : '';
        var visible = VISIBLE_YEARS[yr] === true || (isAnalog && VISIBLE_YEARS[yr] !== false);
        chip = createYearChip(yr, cls, visible, false);
        chip.onclick = function() {{ toggleYear(yr, this); }};
        container.appendChild(chip);
      }}
    }});
  }}
  
  // Build chips for US
  if(containerUS && SL_US){{
    containerUS.innerHTML = '';
    // Similar logic for US...
  }}
}}

function createYearChip(year, cls, active, disabled){{
  var chip = document.createElement('span');
  chip.className = 'year-chip ' + cls + (active ? ' active' : '');
  chip.textContent = year;
  chip.dataset.year = year;
  if(disabled) chip.style.cursor = 'default';
  return chip;
}}

function toggleYear(year, chip){{
  VISIBLE_YEARS[year] = !chip.classList.contains('active');
  chip.classList.toggle('active');
  drawSeasonCharts();
}}

// ── FIXED: Seasonality Charts ───────────────────────────────────────────
function drawSeasonCharts(){{
  var grid=document.getElementById("sg");if(!grid)return;grid.innerHTML="";
  var sel=document.getElementById("sl-var")&&document.getElementById("sl-var").value;
  var show=(!sel||sel==="ALL")?VARS:[sel];
  grid.style.gridTemplateColumns=show.length===1?"1fr":"1fr 1fr";
  
  var slData = getSlData();
  if(!slData || !slData.variables){{
    grid.innerHTML='<div style="color:#fc8181;padding:16px;font-size:.78rem">⚠ No drought data available.</div>';
    return;
  }}
  
  show.forEach(function(v){{
    var card=document.createElement("div");card.className="chart-card";
    var cid="sl_"+v.replace(/[^a-z0-9]/gi,"_");
    card.innerHTML='<h3 style="color:'+VC[v]+'">'+(slData.geo||"TX")+' — '+v+'</h3><div id="'+cid+'"></div>';
    grid.appendChild(card);
    drawOneLineFixed(cid, v, slData);
  }});
}}

function drawOneLineFixed(cid, vn, slData){{
  var c=document.getElementById(cid);if(!c)return;c.innerHTML="";
  var d=slData.variables[vn];if(!d)return;
  
  var weeks=slData.iso_weeks, wl=slData.weeks, cYr=slData.curr_yr;
  var last1=slData.last_yr_1, last2=slData.last_yr_2;
  var analogs=d.analogs||[];
  var ser=d.series, aYrs=d.all_hist_years;
  
  var cw=c.clientWidth||340, ML=36, MR=60, MT=14, MB=44, svgH=185;
  var pw=cw-ML-MR, ph=svgH-MT-MB;
  
  var allV=[];
  Object.keys(ser).forEach(function(y){{(ser[y]||[]).forEach(function(v){{if(v!==null)allV.push(v);}});}});
  var yMax=allV.length?Math.min(100,Math.max.apply(null,allV)*1.1):100;
  
  var xS=sc(0,weeks.length-1,0,pw), yS=sc(0,yMax,ph,0);
  var svg=msvg(cw,svgH), g=mel("g",{{transform:"translate("+ML+","+MT+")"}});
  
  // Grid lines
  [0,25,50,75,100].filter(function(t){{return t<=yMax+1;}}).forEach(function(t){{
    var yy=yS(t);
    g.appendChild(mel("line",{{x1:0,y1:yy,x2:pw,y2:yy,stroke:"#1a2535","stroke-width":"1"}}));
    g.appendChild(mtx(t+"%",{{x:-4,y:yy+3,"text-anchor":"end","font-size":"7"}},"#607080"));
  }});
  
  // X-axis labels
  wl.forEach(function(lbl,i){{
    if(i%4===0){{
      var xx=xS(i);
      var tx=mel("text",{{x:xx,y:ph+12,"text-anchor":"middle",transform:"rotate(-35,"+xx+","+(ph+12)+")","font-size":"7","fill":"#607080"}});
      tx.textContent=lbl;g.appendChild(tx);
    }}
  }});
  
  // Draw line function
  function mkLine(yr, col, sw, op, dash){{
    var line=ser[String(yr)];if(!line)return;
    var pts=[];line.forEach(function(v,i){{if(v!==null)pts.push([xS(i),yS(v)]);}});
    if(pts.length<2)return;
    var pd="M"+pts[0][0].toFixed(1)+","+pts[0][1].toFixed(1);
    for(var i=1;i<pts.length;i++)pd+="L"+pts[i][0].toFixed(1)+","+pts[i][1].toFixed(1);
    var a={{d:pd,stroke:col,"stroke-width":String(sw),fill:"none",opacity:String(op)}};
    if(dash)a["stroke-dasharray"]=dash;
    tt(g.appendChild(mel("path",a)),String(yr));
  }}
  
  // Draw historical years (grey, thin)
  aYrs.forEach(function(yr){{
    if(yr!==cYr && yr!==last1 && yr!==last2 && analogs.indexOf(yr)<0){{
      if(VISIBLE_YEARS[yr] !== false){{
        mkLine(yr, COLORS.hist, 0.7, 0.4, null);
      }}
    }}
  }});
  
  // Draw 2 years ago (purple)
  if(ser[String(last2)] && VISIBLE_YEARS[last2] !== false){{
    mkLine(last2, COLORS.last2, 1.5, 0.8, null);
  }}
  
  // Draw last year (blue)
  if(ser[String(last1)] && VISIBLE_YEARS[last1] !== false){{
    mkLine(last1, COLORS.last1, 2, 0.9, null);
  }}
  
  // Draw analogs (dashed, distinct colors)
  analogs.forEach(function(yr,i){{
    if(VISIBLE_YEARS[yr] !== false){{
      var col=COLORS.analog[i%5];
      mkLine(yr, col, 1.8, 0.85, "5,3");
    }}
  }});
  
  // Draw current year (green, bold)
  mkLine(cYr, COLORS.curr, 3, 1, null);
  
  // Highlight points
  var cl2=ser[String(cYr)];
  if(cl2){{
    var lv=null, li=-1;cl2.forEach(function(v,j){{if(v!==null){{lv=v;li=j;}}}});
    if(lv!==null){{
      g.appendChild(mel("circle",{{cx:xS(li),cy:yS(lv),r:"5",fill:COLORS.curr,stroke:"#fff","stroke-width":"1"}}));
      g.appendChild(mtx(cYr+": "+lv.toFixed(1)+"%",{{x:xS(li)+8,y:yS(lv)-8,"font-size":"9","font-weight":"700"}},COLORS.curr));
    }}
  }}
  
  // Vertical line at latest week
  var latIdx=weeks.indexOf(slData.latest_iw);
  if(latIdx>=0){{var xx=xS(latIdx);g.appendChild(mel("line",{{x1:xx,y1:0,x2:xx,y2:ph,stroke:"#607080","stroke-width":"1","stroke-dasharray":"3,3"}}));}}
  
  svg.appendChild(g);c.appendChild(svg);
  
  // Legend
  var leg=document.createElement("div");leg.style.cssText="display:flex;flex-wrap:wrap;gap:8px;margin-top:6px;font-size:.65rem";
  var items=[
    [String(cYr), COLORS.curr, "700", "solid", "●"],
    [String(last1), COLORS.last1, "600", "solid", "—"],
    [String(last2), COLORS.last2, "500", "solid", "—"]
  ];
  analogs.forEach(function(yr,i){{items.push([String(yr), COLORS.analog[i%5], "500", "dashed", "–"]);}});
  items.push(["Other", COLORS.hist, "400", "solid", "—"]);
  
  items.forEach(function(item){{
    var ls=item[3]==="dashed"?"border-top:2px dashed "+item[1]:"border-top:2px solid "+item[1];
    var marker=item[4]==="●"?('<span style="color:'+item[1]+';font-size:10px;margin-right:2px">●</span>'):('<span style="width:12px;'+ls+';display:inline-block;margin-right:3px"></span>');
    leg.innerHTML+='<span style="display:flex;align-items:center;gap:2px">'+marker+'<span style="color:'+item[1]+';font-weight:'+item[2]+'">'+item[0]+'</span></span>';
  }});
  c.appendChild(leg);
}}

// ── FIXED: Last 6 Weeks Table ───────────────────────────────────────────
function drawLast6(){{
  var c=document.getElementById("l6tbl");if(!c)return;
  var vn=(document.getElementById("l6-var")||{{}}).value||"D1-D4";
  drawLast6Fixed(c, vn, SL);
}}

function drawLast6Fixed(c, vn, slData){{
  if(!slData||!slData.variables){{c.innerHTML='<div style="color:#607080;padding:10px">No data.</div>';return;}}
  var vd=slData.variables[vn];if(!vd)return;
  
  var rows6=slData.last6_weeks||[];
  var comps=vd.last6_comparisons||[];
  var latestIW=slData.latest_week_iw;
  var latestDate=slData.latest_week_date||"";
  
  // Build header
  var hdr='<tr>'+
    '<th class="lft" style="background:#0e1d2e;min-width:100px">Year / Type</th>'+
    '<th style="background:#0e1d2e;color:#90a4ae">Week</th>'+
    '<th style="background:#0e1d2e;color:#607080">Date</th>';
  VARS.forEach(function(v){{
    hdr+='<th style="background:#0e1d2e;color:'+VC[v]+'">'+v+'</th>';
  }});
  hdr+='</tr>';
  
  // Current year rows
  var body='';
  rows6.forEach(function(r, idx){{
    var isMostRecent=(r.iso_week===latestIW);
    var isLast = idx === rows6.length - 1;
    body+='<tr style="'+(isMostRecent?'background:#0a1f14':'')+'">'+
      '<td class="lft" style="color:#68d391">'+(isMostRecent?'★ ':'')+slData.curr_yr+'</td>'+
      '<td>'+(r.label||'')+'</td>'+
      '<td style="color:#607080;font-size:.64rem">'+(r.date||'')+'</td>';
    VARS.forEach(function(v){{
      var val=r[v];
      body+='<td style="'+(v===vn?'color:#f6e05e;font-weight:600':'')+'">'+(val!=null?val.toFixed(1)+'%':'—')+'</td>';
    }});
    body+='</tr>';
  }});
  
  // Comparison years header
  body+='<tr><td colspan="'+(3+VARS.length)+'" style="background:#1a2535;padding:4px 8px;font-size:.64rem;color:#607080;font-weight:600">Comparison Years — Same Weeks</td></tr>';
  
  // Comparison rows
  comps.forEach(function(comp){{
    var isLY=comp.is_last_yr;
    var isLY2=comp.is_last_yr_2;
    var isAnalog=comp.is_analog;
    var color=isLY?'#90c4ff':(isLY2?'#c4a0f0':'#90a4ae');
    var marker=isLY?'⬅ Last Yr':(isLY2?'← 2 Yrs Ago':(isAnalog?'● Analog':'  '));
    
    if(comp.entries){{
      comp.entries.forEach(function(entry, idx){{
        body+='<tr style="'+(isLY?'background:#0a1a2a':(isLY2?'background:#0a0a2a':''))+'">'+
          '<td class="lft" style="color:'+color+'">'+(idx===0?marker+' '+comp.year:'')+'</td>'+
          '<td style="color:#607080">'+(entry.label||'')+'</td>'+
          '<td style="color:#607080;font-size:.64rem">—</td>';
        VARS.forEach(function(v){{
          var val=entry[v];
          body+='<td style="'+(v===vn?'font-weight:600':'')+'">'+(val!=null?val.toFixed(1)+'%':'—')+'</td>';
        }});
        body+='</tr>';
      }});
    }}
  }});
  
  c.innerHTML='<table><thead>'+hdr+'</thead><tbody>'+body+'</tbody></table>';
}}

// ── US Seasonality (similar fixes) ──────────────────────────────────────
function drawSeasonChartsUS(){{
  var grid=document.getElementById("sg-us");if(!grid)return;grid.innerHTML="";
  if(!SL_US||!SL_US.variables){{
    grid.innerHTML='<div style="color:#fc8181;padding:16px;font-size:.78rem">⚠ No US drought data. Upload <code>data/drought_US.csv</code> and re-run process_data.py.</div>';
    return;
  }}
  var sel=document.getElementById("sl-var-us")&&document.getElementById("sl-var-us").value;
  var show=(!sel||sel==="ALL")?VARS:[sel];
  grid.style.gridTemplateColumns=show.length===1?"1fr":"1fr 1fr";
  show.forEach(function(v){{
    var card=document.createElement("div");card.className="chart-card";
    var cid="sl_us_"+v.replace(/[^a-z0-9]/gi,"_");
    card.innerHTML='<h3 style="color:'+VC[v]+'">'+(SL_US.geo||"US")+' — '+v+'</h3><div id="'+cid+'"></div>';
    grid.appendChild(card);
    drawOneLineFixed(cid, v, SL_US);
  }});
}}

function drawLast6US(){{
  var c=document.getElementById("l6tbl-us");if(!c)return;
  if(!SL_US||!SL_US.variables){{c.innerHTML='<div style="color:#607080;padding:10px">No US drought data.</div>';return;}}
  var vn=(document.getElementById("l6-var-us")||{{}}).value||"D1-D4";
  drawLast6Fixed(c, vn, SL_US);
}}
// ── Scatter charts ───────────────────────────────────────────────────────
function drawScSet(gid,rows){{
  var g=document.getElementById(gid);if(!g)return;g.innerHTML="";
  if(!rows||!rows.length){{g.innerHTML='<div style="color:#607080;padding:16px">No data.</div>';return;}}
  var last=rows[rows.length-1];
  VARS.forEach(function(v){{
    var card=document.createElement("div");card.className="sc-card";
    var sid=gid+"_"+v.replace(/[^a-z0-9]/gi,"_");
    var lbl=last.label||(last.month?["","","","","Apr","May","Jun","Jul","Aug","Sep","Oct"][last.month]:"?");
    card.innerHTML='<h3 style="color:'+VC[v]+'">'+v+' <span style="color:#607080;font-weight:400;font-size:.68rem">'+lbl+'</span></h3><div id="'+sid+'"></div>';
    g.appendChild(card);
    var d=last.vars&&last.vars[v];
    if(!d||d.point===null){{var el=document.getElementById(sid);if(el)el.innerHTML='<div style="color:#607080;padding:10px;font-size:.7rem">No data.</div>';return;}}
    drawSc(sid,v,d);
  }});
}}

function drawSc(cid,vn,d){{
  var c=document.getElementById(cid);if(!c)return;c.innerHTML="";
  var xs=d.scatter_x||[],ys=d.scatter_y||[],yrs=d.scatter_years||[];
  var cx=d.curr_x,cy=d.point,clo=d.lo,chi=d.hi;
  var rx=d.reg_x||[],ry=d.reg_y||[];
  if(!xs.length){{c.innerHTML='<div style="color:#607080;padding:8px;font-size:.7rem">No data</div>';return;}}
  var cw=c.clientWidth||300,ML=38,MR=12,MT=16,MB=40,svgW=cw,svgH=200,pw=svgW-ML-MR,ph=svgH-MT-MB;
  var allX=xs.slice();if(cx!=null)allX.push(cx);
  var allY=ys.slice();if(cy!=null)allY.push(cy);if(clo!=null)allY.push(clo);if(chi!=null)allY.push(chi);
  var xMn=Math.max(0,Math.min.apply(null,allX)),xMx=Math.min(100,Math.max.apply(null,allX));
  var yMn=Math.max(0,Math.min.apply(null,allY)),yMx=Math.min(1,Math.max.apply(null,allY));
  var xPad=(xMx-xMn)*0.14||5,yPad=(yMx-yMn)*0.18||0.05;
  xMn=Math.max(0,xMn-xPad);xMx=Math.min(100,xMx+xPad);yMn=Math.max(0,yMn-yPad);yMx=Math.min(1,yMx+yPad);
  var xS=sc(xMn,xMx,0,pw),yS=sc(yMn,yMx,ph,0),svg=msvg(svgW,svgH);
  var g=mel("g",{{transform:"translate("+ML+","+MT+")"}});
  [0,20,40,60,80,100].filter(function(t){{return t>=xMn&&t<=xMx;}}).forEach(function(t){{
    var xx=xS(t);g.appendChild(mel("line",{{x1:xx,y1:0,x2:xx,y2:ph,stroke:"#1a2535","stroke-width":"1"}}));
    g.appendChild(mtx(t+"%",{{x:xx,y:ph+12,"text-anchor":"middle","font-size":"7.5"}},"#607080"));
  }});
  [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1].filter(function(t){{return t>=yMn-0.01&&t<=yMx+0.01;}}).forEach(function(t){{
    var yy=yS(t);g.appendChild(mel("line",{{x1:0,y1:yy,x2:pw,y2:yy,stroke:"#1a2535","stroke-width":"1"}}));
    g.appendChild(mtx((t*100).toFixed(0)+"%",{{x:-4,y:yy+3,"text-anchor":"end","font-size":"7.5"}},"#607080"));
  }});
  g.appendChild(mtx(vn+" (% area)",{{x:pw/2,y:ph+30,"text-anchor":"middle","font-size":"8"}},"#90a4ae"));
  var ayl=mel("text",{{x:0,y:0,"text-anchor":"middle","font-size":"8",fill:"#90a4ae",transform:"rotate(-90) translate("+(-(ph/2))+",-30)"}});ayl.textContent="Abandonment ratio";g.appendChild(ayl);
  if(rx.length===2)g.appendChild(mel("line",{{x1:xS(rx[0]),y1:yS(ry[0]),x2:xS(rx[1]),y2:yS(ry[1]),stroke:VC[vn],"stroke-width":"1.5","stroke-dasharray":"5,3",opacity:"0.65"}}));
  if(cx!=null&&clo!=null&&chi!=null){{
    var xcx=xS(cx),yclo=yS(clo),ychi=yS(chi);
    g.appendChild(mel("line",{{x1:xcx,y1:ychi,x2:xcx,y2:yclo,stroke:"#fff","stroke-width":"2",opacity:"0.5"}}));
    g.appendChild(mel("line",{{x1:xcx-5,y1:yclo,x2:xcx+5,y2:yclo,stroke:"#fff","stroke-width":"2",opacity:"0.5"}}));
    g.appendChild(mel("line",{{x1:xcx-5,y1:ychi,x2:xcx+5,y2:ychi,stroke:"#fff","stroke-width":"2",opacity:"0.5"}}));
  }}
  var mxI=0,mnI=0;
  for(var i=0;i<xs.length;i++){{tt(g.appendChild(mel("circle",{{cx:xS(xs[i]),cy:yS(ys[i]),r:"4",fill:VC[vn],opacity:"0.7",stroke:"#0d1117","stroke-width":"0.5"}})),yrs[i]+": drought="+xs[i].toFixed(1)+"%, abandon="+(ys[i]*100).toFixed(1)+"%");if(ys[i]>ys[mxI])mxI=i;if(ys[i]<ys[mnI])mnI=i;}}
  if(xs.length>0)[mxI,mnI].forEach(function(i){{g.appendChild(mtx(String(yrs[i]),{{x:xS(xs[i])+5,y:yS(ys[i])-4,"font-size":"7.5"}},"#a0b0c0"));}});
  if(cx!=null&&cy!=null){{
    var sx=xS(cx),sy=yS(cy);
    g.appendChild(mel("path",{{d:sp(sx,sy,10,4,5),fill:"#fff",stroke:"#f0a020","stroke-width":"1.5"}}));
    var onR=sx<=pw*0.62,anch=onR?"start":"end",lx=sx+(onR?12:-12),onT=sy>ph*0.35,ly=sy+(onT?-14:16);
    g.appendChild(mtx((d.curr_yr||"Curr")+": "+(cy*100).toFixed(1)+"%",{{x:lx,y:ly,"text-anchor":anch,"font-size":"9","font-weight":"bold"}},"#ffffff"));
    if(clo!=null&&chi!=null)g.appendChild(mtx(CI_PCT+" CI: ["+(clo*100).toFixed(1)+"–"+(chi*100).toFixed(1)+"%]",{{x:lx,y:ly+12,"text-anchor":anch,"font-size":"7"}},"#c0c0c0"));
  }}
  var pv=d.pvalue,pc=pv!=null?(pv<0.05?"#68d391":pv<0.1?"#f6e05e":"#607080"):"#607080";
  g.appendChild(mtx("R²="+(d.r2!=null?(d.r2*100).toFixed(1)+"%":"—")+"  "+(pv!=null?(pv<0.05?"★p<0.05":pv<0.1?"▲p<0.10":"p="+pv.toFixed(2)):""),{{x:pw-2,y:11,"text-anchor":"end","font-size":"8","font-weight":"600"}},pc));
  svg.appendChild(g);c.appendChild(svg);
}}

function drawTblSet(tid,rows,lk){{
  var c=document.getElementById(tid);if(!c)return;
  if(!rows||!rows.length){{c.innerHTML='<div style="padding:14px;color:#607080;text-align:center">No data.</div>';return;}}
  var vH=VARS.map(function(v){{return '<th class="dvh" colspan="3" style="color:'+VC[v]+';background:'+VBG[v]+'">'+v+'</th>';}}).join("");
  var sH=VARS.map(function(){{return '<th class="dvs">Pred%</th><th class="dvs">'+CI_PCT+' CI</th><th class="dvs">R²</th>';}}).join("");
  var tb="";
  rows.forEach(function(row){{
    var lbl=row[lk]||row.label||"?";
    var sub=row.date?'<br><span style="font-size:.59rem;color:#405060">'+row.date+'</span>':'';
    var cells=VARS.map(function(v){{
      var d=row.vars&&row.vars[v];
      if(!d||d.point==null)return '<td class="pt">—</td><td class="ci">—</td><td>—</td>';
      var pt=(d.point*100).toFixed(1)+"%",lo=d.lo!=null?(d.lo*100).toFixed(1)+"%":"?",hi=d.hi!=null?(d.hi*100).toFixed(1)+"%":"?";
      var r2v=d.r2||0,pv=d.pvalue;
      var pc=pv==null?"#607080":pv<0.05?"#68d391":pv<0.1?"#f6e05e":"#607080";
      var pct=Math.round(r2v*100);
      return '<td class="pt">'+pt+'</td><td class="ci">['+lo+'–'+hi+']</td>'+
        '<td><div class="r2w"><div class="r2bg"><div class="r2f" style="width:'+pct+'%;background:'+pc+'"></div></div>'+
        '<span style="font-size:.64rem;color:'+pc+'">'+pct+'%</span></div></td>';
    }}).join("");
    var bv=null,br=-1;VARS.forEach(function(v){{var d=row.vars&&row.vars[v];if(d&&d.r2!=null&&d.r2>br){{br=d.r2;bv=v;}}}});
    var bd=bv?row.vars[bv]:null;
    var bpt=bd&&bd.point!=null?(bd.point*100).toFixed(1)+"%":"—";
    var blo=bd&&bd.lo!=null?(bd.lo*100).toFixed(1)+"%":"?",bhi=bd&&bd.hi!=null?(bd.hi*100).toFixed(1)+"%":"?";
    tb+='<tr><td class="lft">'+lbl+sub+'</td>'+cells+'<td class="bv" style="color:'+(VC[bv]||"#c4a0f0")+'">'+(bv||"—")+'</td><td class="bp">'+bpt+'</td><td class="bc">['+blo+'–'+bhi+']</td></tr>';
  }});
  c.innerHTML='<table><thead><tr><th class="lft" rowspan="2">Period</th>'+vH+'<th class="bvh" colspan="3" style="color:#c4a0f0">★ Best</th></tr><tr>'+sH+'<th class="bvs">Var</th><th class="bvs">Pred%</th><th class="bvs">'+CI_PCT+' CI</th></tr></thead><tbody>'+tb+'</tbody></table>';
}}

// ── Production ────────────────────────────────────────────────────────────
function renderProd(){{
  var txPlt=getTxPlanted();
  var modelAb=BTX&&BTX.point!=null?BTX.point:null;
  initBanner();
  initMCResults();
  buildMatAB();
  buildGridC(txPlt,modelAb);
  buildGridD(modelAb);
}}

function initBanner(){{
  var b=document.getElementById("best-banner");if(!b)return;
  if(BTX&&BTX.point!==null){{
    b.innerHTML='🎯 <b>Best TX Prediction:</b> '+BTX.variable+' ('+BTX.model+', '+BTX.label+') '+
      '· R²='+(BTX.r2*100).toFixed(1)+'% · Abandonment: <b style="color:#68d391">'+(BTX.point*100).toFixed(1)+
      '%</b> ['+((BTX.lo||0)*100).toFixed(1)+'%–'+((BTX.hi||0)*100).toFixed(1)+'%] ('+CI_PCT+' CI)';
  }}else{{b.innerHTML='⚠ No prediction available yet. Add drought data for current season.';}}
}}

function initMCResults(){{
  var box=document.getElementById("mc-results");
  var stats=document.getElementById("mc-stats");
  if(!box||!stats)return;
  
  var mc=PROD.monte_carlo;
  if(!mc){{box.style.display='none';return;}}
  
  box.style.display='block';
  
  // TX stats
  var txStats=mc.tx;
  var usStats=mc.us;
  
  stats.innerHTML=
    '<div class="mc-stat">'+
      '<div class="label">TX Production (mn bales)</div>'+
      '<div class="value">Mean: '+txStats.production.mean+' | Median: '+txStats.production.median+'</div>'+
      '<div style="color:#607080;font-size:.6rem;margin-top:2px">90% CI: ['+txStats.production.ci_5+' – '+txStats.production.ci_95+'] | σ='+txStats.production.std+'</div>'+
    '</div>'+
    '<div class="mc-stat">'+
      '<div class="label">TX Abandonment</div>'+
      '<div class="value">Mean: '+(txStats.abandonment.mean*100).toFixed(1)+'% | σ='+(txStats.abandonment.std*100).toFixed(1)+'%</div>'+
    '</div>'+
    '<div class="mc-stat">'+
      '<div class="label">US Production (mn bales)</div>'+
      '<div class="value">Mean: '+usStats.production.mean+' | Median: '+usStats.production.median+'</div>'+
      '<div style="color:#607080;font-size:.6rem;margin-top:2px">90% CI: ['+usStats.production.ci_5+' – '+usStats.production.ci_95+'] | σ='+usStats.production.std+'</div>'+
    '</div>'+
    '<div class="mc-stat">'+
      '<div class="label">US Abandonment</div>'+
      '<div class="value">Mean: '+(usStats.abandonment.mean*100).toFixed(1)+'% | σ='+(usStats.abandonment.std*100).toFixed(1)+'%</div>'+
    '</div>';
}}

// FIXED: Matrix B shows actual values only (no delta)
function buildMatAB(){{
  var ca=document.getElementById("mA"),cb=document.getElementById("mB");
  if(!ca||!cb)return;
  var PL=PROD.period_labels;
  var periods=[1,5,10,15,20];
  
  function computeCell(pAb, pYld, useTxModel){{
    var total=0; var ok=true;
    var states=PROD.state_data;
    var keys=Object.keys(states).filter(function(s){{return s!=="US"&&s!==PROD.us_geo;}});
    keys.forEach(function(st){{
      var sd=states[st]; if(!ok)return;
      var pa=pAb===1?(sd.last_yr_actual?{{ab:sd.last_yr_actual.ab}}:null):sd.periods&&sd.periods[pAb];
      var py=pYld===1?(sd.last_yr_actual?{{yld:sd.last_yr_actual.yld}}:null):sd.periods&&sd.periods[pYld];
      if(!pa||!py){{ok=false;return;}}
      var ab=(st==="TX"&&useTxModel&&BTX.point!=null)?BTX.point:pa.ab;
      var yld=py.yld;
      var plt=getPlanted(st);
      if(ab==null||yld==null||plt==null){{ok=false;return;}}
      total+=plt*1000*(1-ab)*yld/480000000;
    }});
    return ok?Math.round(total*1000)/1000:null;
  }}
  
  function hdr(){{
    return '<tr><th class="lft" style="min-width:60px;background:#0e1d2e;position:sticky;top:0;z-index:6">Ab↓ Yld→</th>'+
      periods.map(function(p){{return '<th style="background:#0e1d2e;color:#7fb3d3;position:sticky;top:0;z-index:5">'+PL[p]+'</th>';}}).join("")+'</tr>';
  }}
  
  function buildRows(useTxModel){{
    return periods.map(function(pAb){{
      var cells=periods.map(function(pYld){{
        var v=computeCell(pAb,pYld,useTxModel);
        return '<td>'+(v!=null?v.toFixed(2):"—")+'</td>';
      }}).join("");
      return '<tr><th class="lft" style="background:#111820;color:#90a4ae;position:sticky;left:0;z-index:4">'+PL[pAb]+'</th>'+cells+'</tr>';
    }}).join("");
  }}
  
  ca.innerHTML='<table class="mtbl"><thead>'+hdr()+'</thead><tbody>'+buildRows(false)+'</tbody></table>';
  
  // Matrix B - just actual values, no delta
  cb.innerHTML='<table class="mtbl"><thead>'+hdr()+'</thead><tbody>'+buildRows(true)+'</tbody></table>';
  
  document.getElementById("mB-sub").textContent='TX abandonment fixed at '+(BTX.point*100).toFixed(1)+'% (model prediction)';
}}

function buildGridC(txPlt,modelAb){{
  var c=document.getElementById("gC");if(!c)return;
  if(!txPlt){{c.innerHTML='<div style="color:#fc8181;font-size:.7rem;padding:8px">Enter TX planted area above.</div>';return;}}
  var abR=PROD.tx_ab_range,yldR=PROD.tx_yld_range;
  if(!abR||!abR.length){{c.innerHTML='<div style="color:#607080;padding:8px;font-size:.7rem">No TX data available.</div>';return;}}
  var mRow=null;
  if(modelAb!=null){{var diffs=abR.map(function(a){{return Math.abs(a-modelAb);}});mRow=diffs.indexOf(Math.min.apply(null,diffs));}}
  
  var hdr='<tr><th style="text-align:left;background:#0e1d2e;min-width:55px;position:sticky;left:0;top:0;z-index:6">Ab%↓ Yld→</th>'+
    yldR.map(function(y){{return '<th style="background:#0e1d2e;color:#7fb3d3;position:sticky;top:0;z-index:5">'+y+'</th>';}}).join("")+'</tr>';
  var body=abR.map(function(ab,ri){{
    var isMod=ri===mRow;
    var cells=yldR.map(function(yld){{
      var prod=txPlt*1000*(1-ab)*yld/480000000;
      return '<td style="'+(isMod?"color:#68d391;font-weight:600":"")+'">'+prod.toFixed(2)+'</td>';
    }}).join("");
    return '<tr style="'+(isMod?"background:#0a1f14":"")+'">'+
      '<th style="background:'+(isMod?"#061410":"#111820")+';color:'+(isMod?"#68d391":"#90a4ae")+';position:sticky;left:0;z-index:4;text-align:left">'+
      (ab*100).toFixed(0)+"%"+(isMod?" ★":"")+'</th>'+cells+'</tr>';
  }}).join("");
  c.innerHTML='<div style="font-size:.65rem;color:#607080;margin-bottom:4px">TX planted: '+Math.round(txPlt).toLocaleString()+'K ac · mn 480-lb bales'+(modelAb!=null?" · ★=model row":"")+
    '</div><div style="overflow-x:auto;max-height:400px;overflow-y:auto"><table class="mtbl"><thead>'+hdr+'</thead><tbody>'+body+'</tbody></table></div>';
}}

function buildGridD(modelAb){{
  var c=document.getElementById("gD");if(!c)return;
  var usAb=PROD.us_ab_range,usYld=PROD.us_yld_range;
  var usPlt=0,hasAny=false;
  var defs=PROD.state_defaults||{{}};
  Object.keys(defs).filter(function(s){{return s!=="US"&&s!==PROD.us_geo;}}).forEach(function(st){{
    var v=getPlanted(st);if(v){{usPlt+=v;hasAny=true;}}
  }});
  if(!hasAny) usPlt=PROD.us_plt||null;
  if(!usPlt||!usAb||!usAb.length){{
    c.innerHTML='<div style="color:#607080;padding:8px;font-size:.7rem">US data requires full all-states CSV.</div>';return;
  }}
  var mRow=null;
  if(modelAb!=null){{var diffs=usAb.map(function(a){{return Math.abs(a-modelAb);}});mRow=diffs.indexOf(Math.min.apply(null,diffs));}}
  
  var hdr='<tr><th style="text-align:left;background:#0e1d2e;min-width:55px;position:sticky;left:0;top:0;z-index:6">Ab%↓ Yld→</th>'+
    usYld.map(function(y){{return '<th style="background:#0e1d2e;color:#7fb3d3;position:sticky;top:0;z-index:5">'+y+'</th>';}}).join("")+'</tr>';
  var body=usAb.map(function(ab,ri){{
    var isMod=ri===mRow;
    var cells=usYld.map(function(yld){{
      var prod=usPlt*1000*(1-ab)*yld/480000000;
      return '<td style="'+(isMod?"color:#68d391;font-weight:600":"")+'">'+prod.toFixed(2)+'</td>';
    }}).join("");
    return '<tr style="'+(isMod?"background:#0a1f14":"")+'">'+
      '<th style="background:'+(isMod?"#061410":"#111820")+';color:'+(isMod?"#68d391":"#90a4ae")+';position:sticky;left:0;z-index:4;text-align:left">'+
      (ab*100).toFixed(2)+"%"+(isMod?" ★":"")+'</th>'+cells+'</tr>';
  }}).join("");
  c.innerHTML='<div style="font-size:.65rem;color:#607080;margin-bottom:4px">US planted: '+Math.round(usPlt).toLocaleString()+'K ac · Derived US ab from TX model · mn 480-lb bales</div>'+
    '<div style="overflow-x:auto;max-height:400px;overflow-y:auto"><table class="mtbl"><thead>'+hdr+'</thead><tbody>'+body+'</tbody></table></div>';
}}

// ── Planted area ───────────────────────────────────────────────────────────
function initPlantedInputs(){{
  var container=document.getElementById("planted-inputs");
  if(!container)return;
  container.innerHTML="";
  var defs=PROD.state_defaults||{{}};
  var states=Object.keys(defs).sort().filter(function(s){{return s!=="US";}});
  if(defs["US"]||defs[PROD.us_geo||"US"]) states.push("US");
  states.forEach(function(st){{
    var def=defs[st]||{{}};
    var saved=localStorage.getItem("planted_"+st);
    var val=saved?saved:(def.plt?Math.round(def.plt):"");
    var yr=def.year?"("+def.year+")":"";
    var box=document.createElement("div");
    box.style.cssText="display:flex;flex-direction:column;gap:2px;min-width:90px";
    box.innerHTML='<span style="font-size:.66rem;color:'+(st==="TX"?"#68d391":"#90a4ae")+'">'+st+' '+yr+'</span>'+
      '<input type="number" id="plt_'+st+'" value="'+val+'" min="0" max="20000" step="50" '+
      'style="width:90px;background:#1a2535;border:1px solid '+(st==="TX"?"#2d6a4f":"#2d3e50")+
      ';color:#e2e8f0;padding:3px 6px;border-radius:4px;font-size:.71rem"/>';
    container.appendChild(box);
  }});
}}

function savePlanted(){{
  var defs=PROD.state_defaults||{{}};
  var states=Object.keys(defs).sort();
  var saved=[];
  states.forEach(function(st){{
    var el=document.getElementById("plt_"+st);
    if(el&&el.value&&!isNaN(el.value)){{
      localStorage.setItem("planted_"+st,el.value);
      saved.push(st+":"+el.value+"K");
    }}
  }});
  document.getElementById("saved-note").textContent="✓ Saved: "+saved.join(", ");
  renderProd();
}}

function resetPlanted(){{
  var defs=PROD.state_defaults||{{}};
  Object.keys(defs).forEach(function(st){{
    localStorage.removeItem("planted_"+st);
    var el=document.getElementById("plt_"+st);
    if(el&&defs[st]&&defs[st].plt) el.value=Math.round(defs[st].plt);
  }});
  document.getElementById("saved-note").textContent="↺ Reset to last year actuals.";
  renderProd();
}}

function getPlanted(st){{
  var v=localStorage.getItem("planted_"+st);
  if(v&&!isNaN(v)) return parseFloat(v);
  var def=(PROD.state_defaults||{{}})[st];
  return def&&def.plt?def.plt:null;
}}
function getTxPlanted(){{ return getPlanted("TX"); }}

// ── Summary ────────────────────────────────────────────────────────────
function refreshSummary(){{
  var txPlt=getTxPlanted();
  var modelAb=BTX&&BTX.point!=null?BTX.point:null;
  var modelPct=modelAb!=null?(modelAb*100).toFixed(1)+"%":"N/A";
  var loPct=BTX.lo!=null?((BTX.lo||0)*100).toFixed(1)+"%":"?";
  var hiPct=BTX.hi!=null?((BTX.hi||0)*100).toFixed(1)+"%":"?";
  var r2=BTX.r2!=null?(BTX.r2*100).toFixed(1)+"%":"N/A";
  var nWk=WK.length;var latest=WK.length?WK[WK.length-1].date:"N/A";
  
  // Get production ranges
  var mr=PROD.matrix_ranges||{{}};
  var rangeText="";
  ['matA','matB','gridC','gridD'].forEach(function(k){{
    if(mr[k]){{
      rangeText+='\\n• '+mr[k].label+': '+(mr[k].min!=null?mr[k].min.toFixed(2):'N/A')+'–'+(mr[k].max!=null?mr[k].max.toFixed(2):'N/A')+' mn bales';
    }}
  }});
  
  // Previous year comparison
  var prevProd=PROD.prev_year_prod||{{}};
  var prevYr=PROD.prev_yr||'?';
  var prevText="";
  if(prevProd.tx!=null) prevText+='\\n• TX '+prevYr+' Actual: '+prevProd.tx.toFixed(2)+' mn bales';
  if(prevProd.us!=null) prevText+='\\n• US '+prevYr+' Actual: '+prevProd.us.toFixed(2)+' mn bales';
  
  // Monte Carlo
  var mc=PROD.monte_carlo;
  var mcText=mc?
    '\\n\\nMonte Carlo (10,000 runs):\\n• TX: mean='+mc.tx.production.mean+' mn bales, 90% CI ['+mc.tx.production.ci_5+'–'+mc.tx.production.ci_95+']\\n• US: mean='+mc.us.production.mean+' mn bales, 90% CI ['+mc.us.production.ci_5+'–'+mc.us.production.ci_95+']':'';
  
  document.getElementById("ps1").value=SUMMARY.seasonality;
  document.getElementById("ps2").value="Weekly model: "+nWk+" weeks available through "+latest+". See tables for details.";
  document.getElementById("ps3").value="Best model: "+BTX.variable+" ("+BTX.model+") at "+BTX.label+", R²="+r2+". TX abandonment prediction: "+modelPct+" ("+CI_PCT+" CI: "+loPct+"–"+hiPct+").";
  document.getElementById("ps4").value="Production Scenario Ranges (mn 480-lb bales):"+rangeText+prevText+mcText;
  document.getElementById("ps5").value="Key risks: (1) Low R² ("+r2+") means wide CI. (2) Model trained on historical data; extreme events may be underweighted. (3) Final USDA NASS figures may differ.";
}}

// ── Diagnostics ────────────────────────────────────────────────────────────
function drawDiag(){{
  var g=document.getElementById("dg");if(!g)return;g.innerHTML="";
  VARS.forEach(function(v){{
    var card=document.createElement("div");card.className="dg-card";
    var did="dg_"+v.replace(/[^a-z0-9]/gi,"_");
    card.innerHTML='<h3 style="color:'+VC[v]+'">'+v+' — R² by week</h3><div id="'+did+'"></div>';
    g.appendChild(card);
    var lbls=[],wr=[],mr=[],cr=[];
    WK.forEach(function(r){{lbls.push(r.label);var d=r.vars[v];wr.push(d?d.r2:null);}});
    var moM={{}};MO.forEach(function(r){{var d=r.vars[v];moM[r.label]=(d?d.r2:null);}});
    mr=lbls.map(function(l){{return moM[l.split(" ")[0]]||null;}});
    CU.forEach(function(r,i){{var d=r.vars[v];cr.push(d?d.r2:null);}});
    drawDiagB(did,lbls,wr,mr,cr,VC[v]);
  }});
}}

function drawDiagB(cid,lbls,wr,mr,cr,col){{
  var c=document.getElementById(cid);if(!c)return;c.innerHTML="";
  if(!lbls.length)return;
  var cw=c.clientWidth||330,ML=26,MR=8,MB=38,svgH=120;
  var n=lbls.length,bW=Math.max(3,Math.floor((cw-ML-MR)/n/3)-0.5),gW=bW*3+2;
  var allR=[].concat(wr,mr,cr).filter(function(v){{return v!=null;}});
  var yMx=allR.length?Math.max(Math.max.apply(null,allR),0.05):0.15;
  var yS=sc(0,yMx,svgH-MB,8),svg=msvg(cw,svgH);
  [0,0.05,0.10,0.15,0.20].filter(function(t){{return t<=yMx+0.01;}}).forEach(function(t){{
    var yy=yS(t);
    svg.appendChild(mel("line",{{x1:ML,y1:yy,x2:cw-MR,y2:yy,stroke:"#1a2535","stroke-width":"1"}}));
    svg.appendChild(mtx((t*100).toFixed(0)+"%",{{x:ML-2,y:yy+3,"text-anchor":"end","font-size":"7"}},"#607080"));
  }});
  if(0.15<=yMx){{var yy=yS(0.15);svg.appendChild(mel("line",{{x1:ML,y1:yy,x2:cw-MR,y2:yy,stroke:"#68d391","stroke-width":"1","stroke-dasharray":"3,3"}}));}}
  for(var i=0;i<n;i++){{var gx=ML+i*(gW+1);[[wr[i],MC.weekly],[mr[i],MC.monthly],[cr[i],MC.cumulative]].forEach(function(item,j){{var rv=item[0]||0,pcol=item[1],yy=yS(rv),bh=Math.max(1,(svgH-MB)-yy);tt(svg.appendChild(mel("rect",{{x:gx+j*(bW+0.5),y:yy,width:bW,height:bh,rx:1,fill:pcol,opacity:"0.85"}})),lbls[i]+": "+["​Wkly","Mo","Cumul"][j]+" R²="+(item[0]!=null?(item[0]*100).toFixed(1)+"%":"N/A"));}});if(i%3===0){{var tx=mel("text",{{x:gx+gW/2,y:svgH-MB+8,"text-anchor":"middle",transform:"rotate(-35,"+(gx+gW/2)+","+(svgH-MB+8)+")","font-size":"7","fill":"#607080"}});tx.textContent=lbls[i];svg.appendChild(tx);}};}}
  [["​Wkly",MC.weekly],["Mo",MC.monthly],["Cumul",MC.cumulative]].forEach(function(item,i){{var lx=ML+i*58;svg.appendChild(mel("rect",{{x:lx,y:svgH-11,width:8,height:6,rx:1,fill:item[1]}}));svg.appendChild(mtx(item[0],{{x:lx+10,y:svgH-6,"font-size":"7"}},item[1]));}});
  c.appendChild(svg);
}}

// ── Print ──────────────────────────────────────────────────────────────────
function printPDF(){{
  refreshSummary();
  document.querySelectorAll(".panel").forEach(function(p){{p.style.display="block";}});
  window.print();
  document.querySelectorAll(".panel").forEach(function(p){{p.style.display="";}});
  document.querySelectorAll(".panel.active").forEach(function(p){{p.style.display="block";}});
}}

// ── Init ───────────────────────────────────────────────────────────────────
window.onload=function(){{
  VARS.forEach(function(v){{var d=SL.variables[v];ANALOGS[v]=d?d.analogs.slice():[];}});
  initYearChips();
  if(SL_US){{
    VARS.forEach(function(v){{var d=SL_US.variables[v];ANALOGS_US[v]=d?d.analogs.slice():[];}});
  }}
  initPlantedInputs();
  drawSeasonCharts();
  drawLast6();
  drawSeasonChartsUS();
  drawLast6US();
  drawScSet("sc-w",WK);drawScSet("sc-m",MO);drawScSet("sc-c",CU);
  drawTblSet("tbl-w",WK,"label");drawTblSet("tbl-m",MO,"label");drawTblSet("tbl-c",CU,"label");
  initBanner();
}};
window.onresize=function(){{
  drawSeasonCharts();
  drawSeasonChartsUS();
  drawScSet("sc-w",GEO_STATE.wk==="US"?WK_US:WK);
  drawScSet("sc-m",GEO_STATE.mo==="US"?MO_US:MO);
  drawScSet("sc-c",GEO_STATE.cu==="US"?CU_US:CU);
}};
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("TX + US Cotton — Drought Predictor + Production Estimator")
    print("=" * 56)
    for p in [COTTON_CSV, DROUGHT_CSV]:
        if not p.exists():
            print(f"\nERROR: {p} not found. Place CSV files in data/")
            sys.exit(1)

    has_us_drought = DROUGHT_US_CSV.exists()
    if not has_us_drought:
        print(f"  NOTE: {DROUGHT_US_CSV} not found — US drought analysis will be empty")

    print("\nLoading data…")
    cotton = load_cotton(COTTON_CSV)
    drought_tx = load_drought(DROUGHT_CSV)
    drought_us = load_drought(DROUGHT_US_CSV) if has_us_drought else drought_tx

    # TX abandonment for TX drought models
    ab_tx = get_tx(cotton)
    if ab_tx.empty:
        print("ERROR: No TX abandonment data found")
        sys.exit(1)
    print(f"  TX abandonment: {len(ab_tx)} years")

    # US abandonment for US drought models
    ab_us = build_us_ab(cotton)
    if ab_us is None:
        ab_us = ab_tx  # fallback
        print("  US abandonment: using TX as proxy")

    # US abandonment for US drought models (from all-states CSV)
    us_rows = cotton[cotton["geography"]==US_GEO]
    if not us_rows.empty and "abandonment" in us_rows.columns:
        ab_us = us_rows.set_index("mkt_year")["abandonment"]
    else:
        # Derive US abandonment from state sum if US row missing
        non_us = cotton[~cotton["geography"].isin([US_GEO])].copy()
        if not non_us.empty and "upland_cotton_planted_acreage" in non_us.columns:
            grp = non_us.groupby("mkt_year").apply(
                lambda g: 1 - g["upland_cotton_harvested_acreage"].sum()
                            / g["upland_cotton_planted_acreage"].sum()
            ).clip(0,1)
            ab_us = grp
            print(f"  US abandonment derived from state sum: {len(ab_us)} years")
        else:
            ab_us = ab_tx  # fallback
            print("  US abandonment: using TX as proxy (upload full all-states CSV)")

    print("\nBuilding TX regression models…")
    wk_rows, curr_yr = build_weekly(ab_tx, drought_tx)
    mo_rows = build_monthly(ab_tx, drought_tx)
    cu_rows = build_cumulative(ab_tx, drought_tx)
    print(f"  TX: Weekly={len(wk_rows)}, Monthly={len(mo_rows)}, Cumulative={len(cu_rows)}")

    best_tx = best_prediction(wk_rows, mo_rows, cu_rows)
    print(f"  TX best: {best_tx.get('variable','—')} ({best_tx.get('model','—')}) "
          f"R²={best_tx.get('r2',0)*100:.1f}% → {(best_tx.get('point') or 0)*100:.1f}%")

    print("Building US regression models…")
    if has_us_drought and ab_us is not None:
        wk_us, _ = build_weekly(ab_us, drought_us)
        mo_us = build_monthly(ab_us, drought_us)
        cu_us = build_cumulative(ab_us, drought_us)
        best_us = best_prediction(wk_us, mo_us, cu_us)
        print(f"  US: Weekly={len(wk_us)}, Monthly={len(mo_us)}, Cumulative={len(cu_us)}")
        print(f"  US best: {best_us.get('variable','—')} R²={best_us.get('r2',0)*100:.1f}%")
    else:
        wk_us, mo_us, cu_us = [], [], []
        best_us = {"r2":-1,"point":None,"lo":None,"hi":None,
                   "variable":None,"model":None,"label":None,"curr_yr":None}
        print("  US models skipped — no drought_US.csv or US abandonment")

    print("Building seasonality line chart data…")
    slines = build_season_lines(drought_tx, geo_label="TX")
    slines_us = build_season_lines(drought_us, geo_label="US") if has_us_drought else slines
    for v in DROUGHT_VARS:
        d = slines["variables"].get(v,{})
        print(f"  TX {v}: analogs={d.get('analogs',[])}")

    print("Building production data…")
    prod = build_production(cotton, best_tx)
    print(f"  States={len(prod['state_data'])} · TX ab range: "
          f"{prod['tx_ab_range'][0]*100:.0f}%–{prod['tx_ab_range'][-1]*100:.0f}%")
    
    # Monte Carlo results
    if prod.get('monte_carlo'):
        mc = prod['monte_carlo']
        print(f"  Monte Carlo: TX production mean={mc['tx']['production']['mean']:.2f} mn bales, "
              f"US production mean={mc['us']['production']['mean']:.2f} mn bales")

    print("Building analyst summary…")
    summary = build_analyst_summary(best_tx, wk_rows, mo_rows, cu_rows, prod, ab_tx, drought_tx)

    ci_pct = str(int(CI_LEVEL*100))
    print(f"\nGenerating HTML ({ci_pct}% CI)…")
    OUTPUT_HTML.parent.mkdir(exist_ok=True)
    html = make_html(
        wk_rows, mo_rows, cu_rows, slines, prod, best_tx, summary, ab_tx, drought_tx,
        wk_us, mo_us, cu_us, slines_us, best_us, ci_pct
    )
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"\n✓ Done → {OUTPUT_HTML} ({OUTPUT_HTML.stat().st_size//1024} KB)")
    print("\nOpen docs/index.html in any browser.")


if __name__ == "__main__":
    main()
  
