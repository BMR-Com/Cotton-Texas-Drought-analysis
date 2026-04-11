"""
Texas Cotton Abandonment — Drought Predictor
=============================================
Run:  python process_data.py

Reads:
    data/cotton_texas.csv   — USDA NASS upland cotton planted/harvested TX
    data/drought_texas.csv  — USDA Drought Monitor weekly TX cotton area

Writes:
    docs/index.html  — self-contained dashboard, open in any browser

THREE SEPARATE TABLES:
  1. Weekly     (30 rows max) — one row per week, predictor = that week's reading
  2. Monthly    ( 7 rows max) — one row per month, predictor = month average so far
  3. Cumulative (30 rows max) — one row per week, predictor = avg Apr W1 → that week

Only rows with current-season data are shown. New rows added automatically on re-run.
At Apr W1: all three tables show identical predictions (only one week exists).

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
COTTON_CSV    = Path("data/cotton_texas.csv")
DROUGHT_CSV   = Path("data/drought_texas.csv")
OUTPUT_HTML    = Path("docs/index.html")
COTTON_ALL_CSV = Path("data/cotton_texas.csv")   # same file as TX — contains all states + US
CONFIG_JSON    = Path("data/config.json")
US_GEO         = "US"
ANALOG_TOP_N   = 5
PERIODS        = [1, 5, 10, 15, 20]
MONTH_NAMES   = {4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct"}

# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
def load_cotton(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df[df["geography"].str.strip() == "TX"]
    df = df[df["category"].isin([
        "upland_cotton_planted_acreage",
        "upland_cotton_harvested_acreage"
    ])]
    df["period"] = pd.to_numeric(df["period"], errors="coerce")
    df["value"]  = pd.to_numeric(df["value"],  errors="coerce")
    df = df.dropna(subset=["period","value"])
    pivot = df.pivot_table(
        index="period", columns="category", values="value", aggfunc="first"
    ).reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={"period":"mkt_year"})
    pivot = pivot.dropna(subset=[
        "upland_cotton_planted_acreage",
        "upland_cotton_harvested_acreage"
    ])
    pivot["abandonment"] = (
        1 - pivot["upland_cotton_harvested_acreage"]
          / pivot["upland_cotton_planted_acreage"]
    ).clip(0, 1)
    print(f"  Cotton: {len(pivot)} marketing years "
          f"({int(pivot.mkt_year.min())}–{int(pivot.mkt_year.max())})")
    return pivot[["mkt_year","abandonment",
                  "upland_cotton_planted_acreage",
                  "upland_cotton_harvested_acreage"]].set_index("mkt_year")


def load_drought(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={df.columns[0]: "Week"})
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
        lambda r: r["cal_year"] if r["Week"].month >= 4 else r["cal_year"] - 1,
        axis=1
    )
    df = df[df["month"].isin(SEASON_MONTHS)].copy()
    print(f"  Drought: {len(df)} weekly rows "
          f"({df['Week'].min().date()} – {df['Week'].max().date()})")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. OLS HELPER
# ═══════════════════════════════════════════════════════════════════════════
def week_label(iw):
    ref = pd.Timestamp("2019-01-01") + pd.to_timedelta(int(iw) * 7 - 4, unit="D")
    mo  = MONTH_NAMES.get(ref.month, f"M{ref.month}")
    wom = (ref.day - 1) // 7 + 1
    return f"{mo} W{wom}"


def safe(v, n=4):
    if v is None or (isinstance(v, float) and np.isnan(v)): return None
    return round(float(v), n)

def jdump(obj):
    def handler(x):
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return None if np.isnan(x) else float(x)
        if isinstance(x, float) and np.isnan(x): return None
        raise TypeError(f"Not serializable: {type(x)}")
    return json.dumps(obj, separators=(",", ":"), default=handler)


def ols_predict(x_tr, y_tr, x_pred, years, ci=CI_LEVEL):
    """
    Fit OLS on historical (x_tr, y_tr), predict at x_pred.
    Returns a result dict or None.
    """
    mask = np.isfinite(x_tr) & np.isfinite(y_tr)
    xc, yc = x_tr[mask], y_tr[mask]
    yrs_clean = [y for y, m in zip(years, mask) if m]
    n = len(xc)
    if n < MIN_TRAIN_YRS or not np.isfinite(x_pred):
        return None

    sl, ic, r, p, _ = stats.linregress(xc, yc)
    r2 = r ** 2
    s  = np.sqrt(np.sum((yc - (ic + sl * xc)) ** 2) / (n - 2))

    t_crit = stats.t.ppf((1 + ci) / 2, df=n - 2)
    xm     = xc.mean()
    ssx    = np.sum((xc - xm) ** 2)
    se     = s * np.sqrt(1 + 1/n + (x_pred - xm)**2 / max(ssx, 1e-9))
    y_pred = float(np.clip(ic + sl * x_pred, 0, 1))
    lo     = float(np.clip(y_pred - t_crit * se, 0, 1))
    hi     = float(np.clip(y_pred + t_crit * se, 0, 1))

    xlo, xhi = float(xc.min()), float(xc.max())
    return {
        "r2":      round(float(r2), 4),
        "pvalue":  round(float(p), 4),
        "point":   round(y_pred, 4),
        "lo":      round(lo, 4),
        "hi":      round(hi, 4),
        "curr_x":  round(float(x_pred), 2),
        "scatter_x":     [round(float(v), 2) for v in xc],
        "scatter_y":     [round(float(v), 4) for v in yc],
        "scatter_years": [int(y) for y in yrs_clean],
        "reg_x": [round(xlo, 2), round(xhi, 2)],
        "reg_y": [
            round(float(np.clip(ic + sl * xlo, 0, 1)), 4),
            round(float(np.clip(ic + sl * xhi, 0, 1)), 4),
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. BUILD MODELS
# ═══════════════════════════════════════════════════════════════════════════
def get_curr_yr(cotton, drought):
    """Return the current (in-progress) marketing year."""
    latest_drought_yr = int(drought["mkt_year"].max())
    if latest_drought_yr not in cotton.index:
        return latest_drought_yr
    # All drought years are also in cotton — use next year if available
    return latest_drought_yr


def get_hist(cotton, drought, curr_yr):
    """Historical drought — only years with cotton outcome data."""
    return drought[
        (drought["mkt_year"] != curr_yr) &
        (drought["mkt_year"].isin(cotton.index))
    ]


def build_arr(hist_dict, cotton):
    """Convert {mkt_year: x_val} dict to aligned x, y arrays and year list."""
    years = sorted(set(hist_dict.keys()) & set(cotton.index))
    x = np.array([hist_dict[y] for y in years], dtype=float)
    y = np.array([cotton.loc[y, "abandonment"] for y in years], dtype=float)
    return x, y, years


# ── Model A: Weekly ────────────────────────────────────────────────────────
def build_weekly(cotton, drought):
    """
    One row per week present in current season.
    Predictor = that week's exact drought value.
    """
    ab = cotton["abandonment"]
    curr_yr  = get_curr_yr(cotton, drought)
    hist     = get_hist(cotton, drought, curr_yr)
    curr_df  = drought[drought["mkt_year"] == curr_yr].sort_values("Week")

    rows = []
    for _, crow in curr_df.iterrows():
        iw    = int(crow["iso_week"])
        label = week_label(iw)
        date  = str(crow["Week"].date())
        row   = {"iso_week": iw, "label": label, "date": date,
                 "curr_yr": curr_yr, "vars": {}}

        for v in DROUGHT_VARS:
            if pd.isna(crow.get(v)):
                row["vars"][v] = None
                continue
            # Historical: one reading per year at this iso_week (±1 week tolerance)
            hd = {}
            for hy, hg in hist.groupby("mkt_year"):
                sub = hg[hg["iso_week"] == iw]
                if sub.empty:
                    sub = hg[(hg["iso_week"] - iw).abs() <= 1]
                if not sub.empty and pd.notna(sub[v].iloc[0]):
                    hd[hy] = float(sub[v].iloc[0])

            x, y, yrs = build_arr(hd, cotton)
            res = ols_predict(x, y, float(crow[v]), yrs)
            row["vars"][v] = res
        rows.append(row)
    return rows, curr_yr


# ── Model B: Monthly ───────────────────────────────────────────────────────
def build_monthly(cotton, drought):
    """
    One row per calendar month present in current season.
    Predictor = average drought for that month (all available weeks).
    """
    curr_yr  = get_curr_yr(cotton, drought)
    hist     = get_hist(cotton, drought, curr_yr)
    curr_df  = drought[drought["mkt_year"] == curr_yr]
    curr_months = sorted(curr_df["month"].unique())

    rows = []
    for mo in curr_months:
        label = MONTH_NAMES.get(mo, f"M{mo}")
        row   = {"month": mo, "label": label, "curr_yr": curr_yr, "vars": {}}

        for v in DROUGHT_VARS:
            # Current month avg (all weeks of this month available so far)
            mo_curr = curr_df[curr_df["month"] == mo][v].dropna()
            if mo_curr.empty:
                row["vars"][v] = None
                continue
            x_curr = float(mo_curr.mean())

            # Historical: avg of that month per year
            hd = {}
            for hy, hg in hist.groupby("mkt_year"):
                sub = hg[hg["month"] == mo][v].dropna()
                if not sub.empty:
                    hd[hy] = float(sub.mean())

            x, y, yrs = build_arr(hd, cotton)
            res = ols_predict(x, y, x_curr, yrs)
            row["vars"][v] = res
        rows.append(row)
    return rows


# ── Model C: Cumulative ────────────────────────────────────────────────────
def build_cumulative(cotton, drought):
    """
    One row per week present in current season.
    Predictor = average drought from Apr W1 through that week (running avg).
    At Apr W1: same as weekly. Gets more stable as season progresses.
    """
    curr_yr  = get_curr_yr(cotton, drought)
    hist     = get_hist(cotton, drought, curr_yr)
    curr_df  = drought[drought["mkt_year"] == curr_yr].sort_values("Week")

    rows = []
    for _, crow in curr_df.iterrows():
        iw    = int(crow["iso_week"])
        label = week_label(iw)
        date  = str(crow["Week"].date())
        row   = {"iso_week": iw, "label": label, "date": date,
                 "curr_yr": curr_yr, "vars": {}}

        # Current cumulative = mean of all weeks up to and including this one
        curr_sub = curr_df[curr_df["iso_week"] <= iw]

        for v in DROUGHT_VARS:
            curr_vals = curr_sub[v].dropna()
            if curr_vals.empty:
                row["vars"][v] = None
                continue
            x_curr = float(curr_vals.mean())

            # Historical: cumulative avg per year up to this iso_week
            hd = {}
            for hy, hg in hist.groupby("mkt_year"):
                sub = hg[hg["iso_week"] <= iw][v].dropna()
                if not sub.empty:
                    hd[hy] = float(sub.mean())

            x, y, yrs = build_arr(hd, cotton)
            res = ols_predict(x, y, x_curr, yrs)
            row["vars"][v] = res
        rows.append(row)
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# 4. SEASONALITY HEATMAP
# ═══════════════════════════════════════════════════════════════════════════
def build_seasonality(drought, n_years=20):
    max_yr = drought["cal_year"].max()
    df     = drought[drought["cal_year"] >= max_yr - n_years + 1].copy()
    all_weeks = sorted(w for w in df["iso_week"].unique() if 14 <= w <= 43)
    weeks_labels = [week_label(w) for w in all_weeks]
    years = sorted(df["cal_year"].unique().tolist())
    out = {"weeks": weeks_labels, "years": years, "variables": {}}
    for v in DROUGHT_VARS:
        matrix = []
        for yr in years:
            yr_df = df[df["cal_year"] == yr]
            row_v = []
            for w in all_weeks:
                sub = yr_df[yr_df["iso_week"] == w]
                val = (float(sub[v].iloc[0])
                       if not sub.empty and v in sub.columns
                       and pd.notna(sub[v].iloc[0]) else None)
                row_v.append(val)
            matrix.append(row_v)
        out["variables"][v] = matrix
    return out



def load_config():
    if CONFIG_JSON.exists():
        try:
            cfg = json.loads(CONFIG_JSON.read_text())
            print(f"  Config: TX planted area = {cfg.get('tx_planted_area','not set')}")
            return cfg
        except Exception:
            pass
    cfg = {"tx_planted_area": None,
           "note": "Set tx_planted_area to USDA planted acreage in 1000 acres"}
    CONFIG_JSON.parent.mkdir(exist_ok=True)
    CONFIG_JSON.write_text(json.dumps(cfg, indent=2))
    print("  Config: created default data/config.json")
    return cfg


def load_cotton_all(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["geography"] = df["geography"].str.strip()
    df["period"]    = pd.to_numeric(df["period"], errors="coerce")
    df["value"]     = pd.to_numeric(df["value"],  errors="coerce")
    df = df.dropna(subset=["period","value"]).rename(columns={"period":"mkt_year"})
    CATS = ["upland_cotton_planted_acreage","upland_cotton_harvested_acreage",
            "upland_cotton_lint_yield","upland_cotton_production"]
    df = df[df["category"].isin(CATS)]
    pivot = df.pivot_table(index=["geography","mkt_year"],columns="category",
                           values="value",aggfunc="first").reset_index()
    pivot.columns.name = None
    non_us = pivot["geography"] != US_GEO
    if all(c in pivot.columns for c in ["upland_cotton_planted_acreage",
                                         "upland_cotton_harvested_acreage"]):
        pivot.loc[non_us,"abandonment"] = (
            1 - pivot.loc[non_us,"upland_cotton_harvested_acreage"]
              / pivot.loc[non_us,"upland_cotton_planted_acreage"]
        ).clip(0,1)
    print(f"  All-states: {len(pivot['geography'].unique())} geographies")
    return pivot


def best_tx_prediction(wk_rows, mo_rows, cu_rows):
    best = {"r2":-1,"point":None,"lo":None,"hi":None,
            "variable":None,"model":None,"label":None,"curr_yr":None}
    for rows, mname in [(wk_rows,"Weekly"),(mo_rows,"Monthly"),(cu_rows,"Cumulative")]:
        if not rows: continue
        last = rows[-1]
        for v in DROUGHT_VARS:
            d = last.get("vars",{}).get(v)
            if d and d.get("r2") is not None and d["r2"] > best["r2"]:
                best = {"r2":d["r2"],"point":d["point"],"lo":d["lo"],"hi":d["hi"],
                        "variable":v,"model":mname,
                        "label":last.get("label",""),"curr_yr":last.get("curr_yr")}
    return best


def build_seasonality_lines(drought, n_years=20):
    max_yr   = drought["cal_year"].max()
    curr_yr  = int(drought["mkt_year"].max())
    min_yr   = max_yr - n_years + 1
    s_weeks  = sorted(w for w in drought["iso_week"].unique() if 14 <= w <= 43)
    wlabels  = [week_label(w) for w in s_weeks]
    curr_df  = drought[drought["mkt_year"]==curr_yr].sort_values("Week")
    lat_iw   = int(curr_df["iso_week"].max()) if not curr_df.empty else s_weeks[-1]
    hist_yrs = sorted(y for y in drought["cal_year"].unique()
                      if min_yr <= y <= max_yr and y != curr_yr)

    result = {"weeks":wlabels,"iso_weeks":[int(w) for w in s_weeks],
              "curr_yr":int(curr_yr),"latest_iw":int(lat_iw),
              "variables":{},"last6_weeks":[]}

    for _, row in curr_df.tail(6).iterrows():
        entry = {"date":str(row["Week"].date()),"label":week_label(int(row["iso_week"]))}
        for v in DROUGHT_VARS: entry[v] = safe(row.get(v),1)
        result["last6_weeks"].append(entry)

    for v in DROUGHT_VARS:
        series = {}
        for yr in hist_yrs + [curr_yr]:
            yr_df = drought[drought["cal_year"]==yr]
            line  = []
            for iw in s_weeks:
                sub = yr_df[yr_df["iso_week"]==iw]
                val = float(sub[v].iloc[0]) if (not sub.empty and pd.notna(sub[v].iloc[0])) else None
                line.append(val)
            if any(x is not None for x in line):
                series[int(yr)] = line

        lat_idx = s_weeks.index(lat_iw) if lat_iw in s_weeks else len(s_weeks)-1
        c_line  = series.get(curr_yr,[])
        c_vals  = [c_line[i] for i in range(lat_idx+1) if i<len(c_line) and c_line[i] is not None]
        scores  = []
        for yr in hist_yrs:
            yl = series.get(yr,[])
            yv = [yl[i] for i in range(lat_idx+1) if i<len(yl) and yl[i] is not None]
            n  = min(len(c_vals),len(yv))
            if n < 2: continue
            scores.append((yr,float(np.sqrt(np.mean(
                (np.array(c_vals[:n])-np.array(yv[:n]))**2)))))
        scores.sort(key=lambda x:x[1])
        result["variables"][v] = {
            "series":        {str(k):v2 for k,v2 in series.items()},
            "analogs":       [int(y) for y,_ in scores[:ANALOG_TOP_N]],
            "analog_scores": {int(y):round(s,2) for y,s in scores[:ANALOG_TOP_N]},
            "all_hist_years":[int(y) for y in hist_yrs],
        }
    return result


def build_production_data(cotton_all, max_yr):
    out = {}
    for state, sdf in cotton_all.groupby("geography"):
        sdf = sdf[sdf["mkt_year"]<=max_yr].sort_values("mkt_year")
        if sdf.empty: continue
        sd = {"periods":{}}
        for P in PERIODS:
            rec = sdf[sdf["mkt_year"] >= max_yr-P+1]
            if rec.empty: sd["periods"][P]=None; continue
            def gavg(col): return safe(rec[col].mean(),4 if "abandon" in col else 1 if "yield" in col else 0) if col in rec.columns else None
            sd["periods"][P] = {
                "ab_avg": gavg("abandonment"), "yld_avg": gavg("upland_cotton_lint_yield"),
                "plt_avg":gavg("upland_cotton_planted_acreage"),
                "prd_avg":gavg("upland_cotton_production"), "n_years":int(len(rec)),
            }
        lr = sdf.iloc[-1]
        sd["last_yr"] = {"year":int(lr["mkt_year"]),"ab":safe(lr.get("abandonment"),4),
                         "yld":safe(lr.get("upland_cotton_lint_yield"),1),
                         "plt":safe(lr.get("upland_cotton_planted_acreage"),0),
                         "prd":safe(lr.get("upland_cotton_production"),0)}
        out[state] = sd

    def ab_range(df):
        if df.empty or "abandonment" not in df.columns: return [round(0.10+i*0.05,2) for i in range(13)]
        v=df["abandonment"].dropna(); lo=int(float(v.min())*20)/20; hi=(int(float(v.max())*20)+1)/20
        return [round(lo+i*0.05,2) for i in range(int(round((hi-lo)/0.05))+1) if round(lo+i*0.05,2)<=hi+0.001]
    def yr(df,col,default):
        if df.empty or col not in df.columns: return default
        v=df[col].dropna(); return list(range(int(float(v.min())//50)*50,(int(float(v.max())//50)+2)*50,50))

    tx=cotton_all[(cotton_all["geography"]=="TX")&(cotton_all["mkt_year"]>=max_yr-9)]
    us=cotton_all[(cotton_all["geography"]==US_GEO)&(cotton_all["mkt_year"]>=max_yr-9)]
    return {"states":out,"tx_ab_range":ab_range(tx),
            "tx_yld_range":yr(tx,"upland_cotton_lint_yield",list(range(250,851,50))),
            "us_yld_range":yr(us,"upland_cotton_lint_yield",list(range(700,1001,50))),
            "max_yr":int(max_yr),"us_geo":US_GEO}


# ═══════════════════════════════════════════════════════════════════════════
# 5. HTML
# ═══════════════════════════════════════════════════════════════════════════
def make_html(wk_rows, mo_rows, cu_rows, season_lines, prod_data, best_tx, config, cotton, drought, ci_pct):
    def jd(o):
        def handler(x):
            if isinstance(x, (np.integer,)):  return int(x)
            if isinstance(x, (np.floating,)): return None if np.isnan(x) else float(x)
            if isinstance(x, float) and np.isnan(x): return None
            raise TypeError(f"Object of type {type(x)} not serializable")
        return json.dumps(o, separators=(",",":"), default=handler)

    curr_yr   = wk_rows[0]["curr_yr"] if wk_rows else "?"
    ab        = cotton["abandonment"]
    ab_stats  = (f"Actual: min={ab.min()*100:.0f}% "
                 f"mean={ab.mean()*100:.0f}% "
                 f"max={ab.max()*100:.0f}% "
                 f"std={ab.std()*100:.0f}pp")
    run_time  = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    latest_wk = wk_rows[-1]["date"] if wk_rows else "—"
    latest_mo     = mo_rows[-1]["label"] if mo_rows else "—"
    tx_planted    = config.get("tx_planted_area")
    tx_plant_str  = f"{tx_planted:,.0f}K acres" if tx_planted else "Not set — edit data/config.json"
    j_wk     = jd(wk_rows)
    j_mo     = jd(mo_rows)
    j_cu     = jd(cu_rows)
    j_slines = jd(season_lines)
    j_prod   = jd(prod_data)
    j_btx    = jd(best_tx)
    tx_plant_js = str(tx_planted) if tx_planted else "null"

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
  padding:14px 26px;border-bottom:2px solid #2d6a4f;
  display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:8px}}
.hdr h1{{font-size:1.3rem;font-weight:700;color:#f0fff4}}
.hdr p{{font-size:.76rem;color:#90a4ae;margin-top:3px}}
.badge{{background:#1a3050;border:1px solid #2d6a4f;border-radius:20px;
  padding:4px 12px;font-size:.72rem;color:#68d391;white-space:nowrap;align-self:center}}
.meta{{background:#111820;padding:6px 26px;font-size:.72rem;color:#607080;
  border-bottom:1px solid #1e2a3a;display:flex;gap:16px;flex-wrap:wrap}}
.meta b{{color:#68d391}}
.tabs{{display:flex;background:#0d1117;border-bottom:2px solid #1e2a3a;padding:0 18px;overflow-x:auto}}
.tab{{padding:9px 15px;cursor:pointer;font-size:.82rem;color:#607080;
  border-bottom:3px solid transparent;margin-bottom:-2px;white-space:nowrap;transition:color .2s}}
.tab:hover{{color:#c0d0e0}}
.tab.active{{color:#68d391;border-bottom-color:#68d391;font-weight:600}}
.panel{{display:none;padding:16px 20px 40px}}
.panel.active{{display:block}}
.st{{font-size:.9rem;font-weight:600;color:#68d391;margin:16px 0 3px}}
.st:first-child{{margin-top:0}}
.sn{{font-size:.73rem;color:#607080;line-height:1.5;margin-bottom:9px}}
.ctrls{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:11px;align-items:flex-end}}
.ctrl{{display:flex;flex-direction:column;gap:2px;font-size:.73rem;color:#90a4ae}}
.ctrl span{{font-size:.67rem;color:#607080}}
select{{background:#1a2535;border:1px solid #2d3e50;color:#e2e8f0;padding:4px 7px;border-radius:5px;font-size:.73rem;cursor:pointer}}
.card{{background:#111820;border:1px solid #1e2a3a;border-radius:8px;padding:11px;margin-bottom:14px;overflow-x:auto}}
.hm-leg{{display:flex;align-items:center;gap:7px;margin-top:8px;font-size:.7rem;color:#607080}}
.hm-grad{{width:120px;height:8px;border-radius:3px}}
/* Scatter */
.sc-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px}}
@media(max-width:620px){{.sc-grid{{grid-template-columns:1fr}}}}
.sc-card{{background:#111820;border:1px solid #1e2a3a;border-radius:8px;padding:10px}}
.sc-card h3{{font-size:.78rem;font-weight:600;margin-bottom:4px}}
/* Table */
.tw{{overflow-x:auto;border-radius:8px;border:1px solid #1e2a3a;margin-bottom:6px}}
table{{width:100%;border-collapse:collapse;font-size:.72rem;white-space:nowrap}}
thead th{{padding:7px 8px;text-align:center;position:sticky;z-index:5;border-bottom:1px solid #1e2a3a}}
thead tr:first-child th{{background:#0e1d2e;color:#7fb3d3;font-weight:600;top:0}}
thead tr:last-child th{{background:#0a1520;color:#607080;font-weight:400;font-size:.67rem;top:34px;border-bottom:2px solid #2d4060}}
th.wk,td.wk{{text-align:left;padding-left:9px;min-width:72px}}
th.vh{{background:#0a1f14!important}}
th.vs{{background:#061410!important}}
th.bh{{background:#150d25!important}}
th.bs{{background:#0d0820!important}}
tbody tr:nth-child(even){{background:#0e1420}}
tbody tr:nth-child(odd){{background:#0d1117}}
tbody tr:hover{{background:#132030}}
td{{padding:5px 7px;text-align:center;vertical-align:middle;border-bottom:1px solid #141e28}}
td.wk{{font-weight:600;color:#c8d8e8}}
td.pt{{color:#f6e05e;font-weight:600}}
td.ci{{color:#607080;font-size:.67rem}}
td.bv{{font-weight:600;font-size:.69rem}}
td.bp{{color:#c4a0f0;font-weight:700}}
td.bc{{color:#9070c0;font-size:.67rem}}
.r2w{{display:inline-flex;align-items:center;gap:3px}}
.r2bg{{width:28px;height:4px;background:#1e2a3a;border-radius:2px}}
.r2f{{height:4px;border-radius:2px}}
/* Model tabs */
.sect-hdr{{display:flex;align-items:center;gap:10px;margin:18px 0 4px}}
.sect-hdr .st{{margin:0}}
.model-badge{{padding:3px 10px;border-radius:12px;font-size:.7rem;font-weight:600;border:1px solid}}
.mb-w{{background:#0a1f14;border-color:#68d391;color:#68d391}}
.mb-m{{background:#1a1208;border-color:#f6e05e;color:#f6e05e}}
.mb-c{{background:#0a0a1f;border-color:#90a0ff;color:#90a0ff}}
/* Diag */
.dg-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
@media(max-width:620px){{.dg-grid{{grid-template-columns:1fr}}}}
.dg-card{{background:#111820;border:1px solid #1e2a3a;border-radius:8px;padding:10px}}
.dg-card h3{{font-size:.77rem;font-weight:600;margin-bottom:6px}}
.about{{max-width:640px;font-size:.82rem;line-height:1.75;color:#b0c0d0}}
.about h3{{color:#68d391;font-size:.86rem;margin:13px 0 4px}}
.about p{{margin-bottom:7px}}
.nbox{{background:#1a2535;border-left:3px solid #68d391;padding:8px 11px;
  border-radius:0 5px 5px 0;font-size:.75rem;color:#90c4ae;margin:8px 0;line-height:1.6}}
.about table{{border-collapse:collapse;width:100%;margin:7px 0;font-size:.77rem}}
.about table th{{text-align:left;padding:4px 9px;background:#111820;color:#90a4ae;border-bottom:1px solid #1e2a3a}}
.about table td{{padding:4px 9px;border-bottom:1px solid #141e28}}

.mtbl{{border-collapse:collapse;font-size:.69rem;white-space:nowrap}}
.mtbl th,.mtbl td{{padding:3px 7px;text-align:right;border:1px solid #1e2a3a}}
.mtbl thead th{{background:#0e1d2e;color:#90a4ae;font-weight:600}}
.mtbl tbody th{{text-align:left;font-weight:600}}
.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}}
@media(max-width:680px){{.chart-grid{{grid-template-columns:1fr}}}}
.chart-card{{background:#111820;border:1px solid #1e2a3a;border-radius:8px;padding:10px}}
.chart-card h3{{font-size:.77rem;font-weight:600;margin-bottom:4px}}
</style>
</head>
<body>
<div class="hdr">
  <div>
    <h1>🌾 Texas Upland Cotton · Drought Abandonment Predictor · MY {curr_yr}/{str(int(curr_yr)+1)[-2:]}</h1>
    <p>Weekly / Monthly / Cumulative OLS · {ci_pct}% prediction intervals · {ab_stats}</p>
  </div>
  <div class="badge">Run: {run_time}</div>
</div>
<div class="meta">
  <span>Cotton: <b>{int(cotton.index.min())}–{int(cotton.index.max())}</b></span>
  <span>Drought: <b>{drought['Week'].min().date()} – {drought['Week'].max().date()}</b></span>
  <span>Weekly rows: <b>{len(wk_rows)}</b> (through {latest_wk})</span>
  <span>Monthly rows: <b>{len(mo_rows)}</b> (through {latest_mo})</span>
  <span>CI: <b>{ci_pct}%</b> · Min train: <b>{MIN_TRAIN_YRS} yrs</b></span>
</div>

<div class="tabs">
  <div class="tab active" onclick="showTab('t1')">🌡 Seasonality</div>
  <div class="tab"        onclick="showTab('t2')">📊 Weekly</div>
  <div class="tab"        onclick="showTab('t3')">📅 Monthly</div>
  <div class="tab"        onclick="showTab('t4')">📈 Cumulative</div>
  <div class="tab"        onclick="showTab('t5')">🌽 Production</div>
  <div class="tab"        onclick="showTab('t6')">🔬 Diagnostics</div>
  <div class="tab"        onclick="showTab('t7')">ℹ About</div>
</div>

<!-- TAB 1: SEASONALITY -->
<div id="t1" class="panel active">
  <div class="st">Drought Seasonality — Line Charts (Last 20 Years)</div>
  <div class="sn">
    Grey lines = all historical years · <b style="color:#68d391">Bold colored</b> = current year ({curr_yr}) ·
    Dashed colored = top 5 analog years (most similar drought pattern through latest week) ·
    Vertical dashed line = latest available data.
  </div>
  <div class="ctrls">
    <div class="ctrl"><span>Variable</span>
      <select id="sl-var" onchange="drawSeasonCharts()">
        <option value="ALL">All 4 variables (2×2 grid)</option>
        <option value="D1-D4">D1-D4 — Moderate or worse</option>
        <option value="D2-D4">D2-D4 — Severe or worse</option>
        <option value="D3-D4">D3-D4 — Extreme or worse</option>
        <option value="D4">D4 — Exceptional only</option>
      </select>
    </div>
    <div class="ctrl"><span>Swap analog year</span>
      <select id="sl-slot">
        <option value="0">Analog 1</option><option value="1">Analog 2</option>
        <option value="2">Analog 3</option><option value="3">Analog 4</option>
        <option value="4">Analog 5</option>
      </select>
    </div>
    <div class="ctrl"><span>Replace with</span>
      <select id="sl-year"><option value="">— year —</option></select>
    </div>
    <button onclick="swapAnalog()" style="background:#1a4030;border:1px solid #2d6a4f;color:#68d391;padding:4px 11px;border-radius:5px;font-size:.72rem;cursor:pointer">Swap</button>
  </div>
  <div id="season-grid" class="chart-grid"></div>

  <div class="st">Last 6 Weeks — Drought Readings</div>
  <div class="sn">Most recent 6 weekly readings. All values = % of Texas cotton area.</div>
  <div class="tw"><div id="last6-tbl"></div></div>
</div>

<!-- TAB 2: WEEKLY -->
<div id="t2" class="panel">
  <div class="sect-hdr">
    <div class="st">Weekly Model — Scatter Plots</div>
    <span class="model-badge mb-w">Predictor = that week's drought %</span>
  </div>
  <div class="sn">Latest available week shown. Dots = historical years. ★ = current season prediction with {ci_pct}% CI.</div>
  <div class="sc-grid" id="sc-w"></div>
  <div class="st">Weekly Model — Prediction Table</div>
  <div class="sn">One row per week with current-season data. New rows added automatically when drought CSV is updated.</div>
  <div class="tw"><div id="tbl-w"></div></div>
  <div style="font-size:.68rem;color:#405060;margin-top:5px;line-height:1.6">
    Pred% = predicted abandonment ratio · {ci_pct}% CI = prediction interval ·
    R² bar: <span style="color:#68d391">green p&lt;0.05</span> · <span style="color:#f6e05e">yellow p&lt;0.10</span> · <span style="color:#607080">grey n.s.</span>
  </div>
</div>

<!-- TAB 3: MONTHLY -->
<div id="t3" class="panel">
  <div class="sect-hdr">
    <div class="st">Monthly Model — Scatter Plots</div>
    <span class="model-badge mb-m">Predictor = monthly avg drought %</span>
  </div>
  <div class="sn">Latest available month shown. Predictor = average of all available weeks in that month.</div>
  <div class="sc-grid" id="sc-m"></div>
  <div class="st">Monthly Model — Prediction Table</div>
  <div class="sn">One row per calendar month. New rows appear when the first week of a new month is available.</div>
  <div class="tw"><div id="tbl-m"></div></div>
  <div style="font-size:.68rem;color:#405060;margin-top:5px;line-height:1.6">
    Monthly avg = mean of all drought monitor readings in that month so far.
  </div>
</div>

<!-- TAB 4: CUMULATIVE -->
<div id="t4" class="panel">
  <div class="sect-hdr">
    <div class="st">Cumulative Model — Scatter Plots</div>
    <span class="model-badge mb-c">Predictor = running avg from Apr W1</span>
  </div>
  <div class="sn">Latest available week shown. At Apr W1 = same as weekly. Gets more stable as season progresses.</div>
  <div class="sc-grid" id="sc-c"></div>
  <div class="st">Cumulative Model — Prediction Table</div>
  <div class="sn">One row per week. Predictor grows more stable as more weeks are averaged in.</div>
  <div class="tw"><div id="tbl-c"></div></div>
  <div style="font-size:.68rem;color:#405060;margin-top:5px;line-height:1.6">
    Cumulative avg = mean drought from Apr W1 through that exact week.
  </div>
</div>

<!-- TAB 5: PRODUCTION -->
<div id="t5" class="panel">
  <div id="best-pred-banner" style="background:#0a1f14;border:1px solid #2d6a4f;border-radius:7px;padding:8px 14px;margin-bottom:12px;font-size:.77rem"></div>

  <div class="ctrls">
    <div class="ctrl"><span>Historical period for other states</span>
      <select id="period-sel" onchange="renderProduction()">
        <option value="1">Last 1 year</option>
        <option value="5" selected>Last 5 years</option>
        <option value="10">Last 10 years</option>
        <option value="15">Last 15 years</option>
        <option value="20">Last 20 years</option>
      </select>
    </div>
  </div>

  <div class="st">Matrix A — All States Historical Averages</div>
  <div class="sn">All states use their own avg abandonment + avg yield for the selected period. Values = mn 480-lb bales.</div>
  <div class="tw"><div id="matA"></div></div>

  <div class="st">Matrix B — TX Abandonment from Regression Model</div>
  <div class="sn">Same as Matrix A except TX abandonment is replaced by the best OLS regression prediction (latest available week).</div>
  <div class="tw"><div id="matB"></div></div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:4px">
    <div>
      <div class="st">Grid C — Texas Production Scenarios</div>
      <div class="sn">Rows = TX abandonment % in 5% steps (last 10 yr range) · Cols = yield in 50 lb/ac steps · ★ = model prediction row</div>
      <div id="gridC" style="overflow-x:auto"></div>
    </div>
    <div>
      <div class="st">Grid D — US Production Scenarios</div>
      <div class="sn">Same structure for US total. Uses US planted area from data.</div>
      <div id="gridD" style="overflow-x:auto"></div>
    </div>
  </div>
</div>

<!-- TAB 6: DIAGNOSTICS -->
<div id="t6" class="panel">
  <div class="st">R² Comparison — All Three Models</div>
  <div class="sn">
    <span style="color:#68d391">■ Weekly</span> ·
    <span style="color:#f6e05e">■ Monthly</span> ·
    <span style="color:#90a0ff">■ Cumulative</span> ·
    Bar = R² for that period. Green dashed line = p=0.05 significance threshold.
  </div>
  <div class="dg-grid" id="dg-grid"></div>
</div>

<!-- TAB 7: ABOUT -->
<div id="t7" class="panel">
  <div class="about">
    <h3>Three separate model types</h3>
    <div class="nbox">
      <b style="color:#68d391">Weekly (Tab 2):</b> predictor = that exact week's drought %.
      Each row is independent. Captures week-to-week variation.<br><br>
      <b style="color:#f6e05e">Monthly (Tab 3):</b> predictor = average drought % for that calendar month.
      One row per month. Smooths weekly noise.<br><br>
      <b style="color:#90a0ff">Cumulative (Tab 4):</b> predictor = average drought % from Apr W1 through that week.
      At Apr W1 = identical to weekly. Gets progressively more stable as the season builds.
    </div>
    <h3>How predictions update</h3>
    <p>1. Update <code>data/drought_texas.csv</code> with new weekly data<br>
       2. Run <code>python process_data.py</code><br>
       3. New rows appear automatically in the relevant tables<br>
       Cumulative rows update every week · Monthly rows update when a new month begins · Weekly rows update every week</p>
    <h3>Why R² is relatively low (3–13%)</h3>
    <p>TX cotton abandonment depends on many factors: prices, insurance, planting conditions, localized weather. With ~25 overlapping years and a single predictor, 3–13% variance explained is expected. The CI is wide but statistically honest.</p>
    <h3>Season alignment</h3>
    <p>Drought weeks Apr–Oct of year <b>Y</b> → marketing year <b>Y</b> (Apr–Oct 2026 → MY 2026/27).</p>
    <h3>Cotton data notes</h3>
    <p>Marketing years 2004 and 2009 are excluded from training if USDA reports abnormal methodology changes. The model uses {MIN_TRAIN_YRS} minimum training years before making predictions.</p>
  </div>
</div>

<script>
var WK_ROWS  = {j_wk};
var MO_ROWS  = {j_mo};
var CU_ROWS  = {j_cu};
var SLINES   = {j_slines};
var PROD     = {j_prod};
var BEST_TX  = {j_btx};
var TX_PLANT = {tx_plant_js};
var VARS    = ["D1-D4","D2-D4","D3-D4","D4"];
var VCOL    = {{"D1-D4":"#68d391","D2-D4":"#f6e05e","D3-D4":"#fc8181","D4":"#d6bcfa"}};
var VBG     = {{"D1-D4":"#0a1f14","D2-D4":"#1f1a08","D3-D4":"#1f0a0a","D4":"#130a1f"}};
var MCOL    = {{"weekly":"#68d391","monthly":"#f6e05e","cumulative":"#90a0ff"}};
var CI_PCT  = "{ci_pct}%";

// ── Tabs ────────────────────────────────────────────────────────────────
function showTab(id){{
  var ids=["t1","t2","t3","t4","t5","t6","t7"];
  ids.forEach(function(t,i){{
    document.querySelectorAll(".tab")[i].classList.toggle("active",t===id);
    document.getElementById(t).classList.toggle("active",t===id);
  }});
  if(id==="t5") renderProduction();
  if(id==="t6") drawDiag();
}}

// ── SVG helpers ──────────────────────────────────────────────────────────
var NS="http://www.w3.org/2000/svg";
function mel(tag,a){{var e=document.createElementNS(NS,tag);Object.keys(a).forEach(function(k){{e.setAttribute(k,a[k]);}});return e;}}
function msvg(w,h){{var s=document.createElementNS(NS,"svg");s.setAttribute("width","100%");s.setAttribute("viewBox","0 0 "+w+" "+h);s.setAttribute("style","font-family:inherit;display:block;overflow:visible");return s;}}
function tt(el,txt){{var t=document.createElementNS(NS,"title");t.textContent=txt;el.appendChild(t);return el;}}
function mtxt(txt,a,fill){{var e=mel("text",a);e.textContent=txt;if(fill)e.setAttribute("fill",fill);return e;}}
function scl(d0,d1,r0,r1){{return function(v){{return d1===d0?r0:(r0+(v-d0)/(d1-d0)*(r1-r0));}}}}
function clamp(v,lo,hi){{return Math.max(lo,Math.min(hi,v));}}
function lerpc(c1,c2,t){{
  var r1=parseInt(c1.slice(1,3),16),g1=parseInt(c1.slice(3,5),16),b1=parseInt(c1.slice(5,7),16);
  var r2=parseInt(c2.slice(1,3),16),g2=parseInt(c2.slice(3,5),16),b2=parseInt(c2.slice(5,7),16);
  return "rgb("+Math.round(r1+(r2-r1)*t)+","+Math.round(g1+(g2-g1)*t)+","+Math.round(b1+(b2-b1)*t)+")";
}}
function starPath(cx,cy,or_,ir,pts){{
  var d="";for(var i=0;i<pts*2;i++){{var r=i%2===0?or_:ir,a=(Math.PI/pts)*i-Math.PI/2;d+=(i===0?"M":"L")+(cx+r*Math.cos(a)).toFixed(2)+","+(cy+r*Math.sin(a)).toFixed(2);}}return d+"Z";
}}

// ── Heatmap ──────────────────────────────────────────────────────────────
function drawHeatmap(){{
  var c=document.getElementById("hm");if(!c)return;c.innerHTML="";
  var vn=document.getElementById("hv").value,sc=document.getElementById("hs").value;
  var mat=SLINES.variables[vn],weeks=SLINES.weeks,years=SLINES.years;if(!mat)return;
  var cw=c.clientWidth||800,ML=50,MT=68,cellH=17;
  var cellW=Math.max(11,Math.floor((cw-ML-6)/weeks.length));
  var svgW=ML+weeks.length*cellW,svgH=MT+years.length*cellH+6;
  var allV=[];for(var yi=0;yi<mat.length;yi++)for(var wi=0;wi<mat[yi].length;wi++)if(mat[yi][wi]!==null)allV.push(mat[yi][wi]);
  var maxV=sc==="rel"?(allV.length?Math.max.apply(null,allV):100):100;
  var svg=msvg(svgW,svgH),colTo=VCOL[vn]||"#68d391";
  for(var wi=0;wi<weeks.length;wi++){{var tx=mel("text",{{x:ML+wi*cellW+cellW/2,y:MT-5,transform:"rotate(-55,"+(ML+wi*cellW+cellW/2)+","+(MT-5)+")","text-anchor":"end","font-size":"9","fill":"#607080"}});tx.textContent=weeks[wi];svg.appendChild(tx);}}
  for(var yi=0;yi<years.length;yi++){{var tx=mel("text",{{x:ML-5,y:MT+yi*cellH+cellH/2+3,"text-anchor":"end","font-size":"9","fill":"#8090a0"}});tx.textContent=String(years[yi]);svg.appendChild(tx);}}
  for(var yi=0;yi<years.length;yi++)for(var wi=0;wi<mat[yi].length;wi++){{var val=mat[yi][wi];tt(svg.appendChild(mel("rect",{{x:ML+wi*cellW,y:MT+yi*cellH,width:cellW-1,height:cellH-1,rx:2,fill:val===null?"#1a2030":lerpc("#0d1520",colTo,clamp(val/maxV,0,1))}})),years[yi]+" "+weeks[wi]+": "+(val===null?"N/A":val.toFixed(1)+"%"));}}
  c.appendChild(svg);
  document.getElementById("hm-leg").style.background="linear-gradient(90deg,#0d1520,"+colTo+")";
}}

function drawAvgBar(){{
  var c=document.getElementById("avgbar");if(!c)return;c.innerHTML="";
  var cw=c.clientWidth||800,ML=42,MR=10,MB=46,MT=8,weeks=SLINES.weeks,nw=weeks.length;
  var avgs={{}};
  VARS.forEach(function(v){{var mat=SLINES.variables[v];avgs[v]=[];for(var wi=0;wi<nw;wi++){{var s=0,cnt=0;for(var yi=0;yi<mat.length;yi++)if(mat[yi][wi]!==null){{s+=mat[yi][wi];cnt++;}}avgs[v].push(cnt?s/cnt:0);}}}});
  var svgH=120,bW=Math.max(2,Math.floor((cw-ML-MR)/nw/VARS.length)-0.5),gW=bW*VARS.length+2;
  var allA=[];VARS.forEach(function(v){{avgs[v].forEach(function(x){{allA.push(x);}});}});
  var yMax=allA.length?Math.max.apply(null,allA):100,yS=scl(0,yMax,svgH-MB,MT),svg=msvg(cw,svgH);
  [0,25,50,75,100].forEach(function(t){{if(t>yMax+5)return;var yy=yS(t);svg.appendChild(mel("line",{{x1:ML,y1:yy,x2:cw-MR,y2:yy,stroke:"#1a2535","stroke-width":"1"}}));svg.appendChild(mtxt(t+"%",{{x:ML-3,y:yy+3,"text-anchor":"end","font-size":"8"}},"#607080"));}});
  for(var wi=0;wi<nw;wi++){{var gx=ML+wi*gW;for(var vi=0;vi<VARS.length;vi++){{var val=avgs[VARS[vi]][wi]||0,yy=yS(val),bh=Math.max(1,(svgH-MB)-yy);tt(svg.appendChild(mel("rect",{{x:gx+vi*bW,y:yy,width:Math.max(1,bW-0.5),height:bh,rx:1,fill:VCOL[VARS[vi]]}})),weeks[wi]+" "+VARS[vi]+": "+val.toFixed(1)+"%");}}if(wi%3===0){{var tx=mel("text",{{x:gx+gW/2,y:svgH-MB+10,"text-anchor":"middle",transform:"rotate(-40,"+(gx+gW/2)+","+(svgH-MB+10)+")","font-size":"8","fill":"#607080"}});tx.textContent=weeks[wi];svg.appendChild(tx);}}}};
  VARS.forEach(function(v,i){{var lx=ML+i*86;svg.appendChild(mel("rect",{{x:lx,y:svgH-12,width:9,height:7,rx:2,fill:VCOL[v]}}));svg.appendChild(mtxt(v,{{x:lx+12,y:svgH-6,"font-size":"8"}},VCOL[v]));}});
  c.appendChild(svg);
}}

// ── Scatter ───────────────────────────────────────────────────────────────
function drawScatterSet(gridId, rows){{
  var grid=document.getElementById(gridId);if(!grid)return;grid.innerHTML="";
  if(!rows||!rows.length){{grid.innerHTML='<div style="color:#607080;padding:20px">No data.</div>';return;}}
  var lastRow=rows[rows.length-1];
  VARS.forEach(function(v){{
    var card=document.createElement("div");card.className="sc-card";
    var sid=gridId+"_"+v.replace(/[^a-z0-9]/gi,"_");
    var d=lastRow.vars&&lastRow.vars[v];
    var wk=lastRow.label||(lastRow.month?["","","","","Apr","May","Jun","Jul","Aug","Sep","Oct"][lastRow.month]:"?");
    card.innerHTML='<h3 style="color:'+VCOL[v]+'">'+v+' <span style="color:#607080;font-weight:400;font-size:.69rem">as of '+wk+'</span></h3><div id="'+sid+'"></div>';
    grid.appendChild(card);
    if(!d||d.point===null){{var el=document.getElementById(sid);if(el)el.innerHTML='<div style="color:#607080;padding:12px;font-size:.72rem">No data for current season.</div>';return;}}
    drawOneScatter(sid,v,d);
  }});
}}

function drawOneScatter(cid,vn,d){{
  var c=document.getElementById(cid);if(!c)return;c.innerHTML="";
  var xs=d.scatter_x||[],ys=d.scatter_y||[],yrs=d.scatter_years||[];
  var cx=d.curr_x,cy=d.point,clo=d.lo,chi=d.hi,cyr=d.curr_yr;
  var rx=d.reg_x||[],ry=d.reg_y||[];
  if(!xs.length){{c.innerHTML='<div style="color:#607080;padding:10px;font-size:.72rem">No data</div>';return;}}
  var cw=c.clientWidth||300,ML=40,MR=14,MT=18,MB=44,svgW=cw,svgH=215,pw=svgW-ML-MR,ph=svgH-MT-MB;
  var allX=xs.slice();if(cx!==null&&cx!==undefined)allX.push(cx);
  var allY=ys.slice();if(cy!==null)allY.push(cy);if(clo!==null)allY.push(clo);if(chi!==null)allY.push(chi);
  var xMin=Math.max(0,Math.min.apply(null,allX)),xMax=Math.min(100,Math.max.apply(null,allX));
  var yMin=Math.max(0,Math.min.apply(null,allY)),yMax=Math.min(1,Math.max.apply(null,allY));
  var xPad=(xMax-xMin)*0.14||5,yPad=(yMax-yMin)*0.18||0.05;
  xMin=Math.max(0,xMin-xPad);xMax=Math.min(100,xMax+xPad);yMin=Math.max(0,yMin-yPad);yMax=Math.min(1,yMax+yPad);
  var xS=scl(xMin,xMax,0,pw),yS=scl(yMin,yMax,ph,0),svg=msvg(svgW,svgH);
  var g=mel("g",{{transform:"translate("+ML+","+MT+")"}});
  [0,10,20,30,40,50,60,70,80,90,100].filter(function(t){{return t>=xMin&&t<=xMax;}}).forEach(function(t){{var xx=xS(t);g.appendChild(mel("line",{{x1:xx,y1:0,x2:xx,y2:ph,stroke:"#1a2535","stroke-width":"1"}}));g.appendChild(mtxt(t+"%",{{x:xx,y:ph+13,"text-anchor":"middle","font-size":"8"}},"#607080"));}});
  [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0].filter(function(t){{return t>=yMin-0.01&&t<=yMax+0.01;}}).forEach(function(t){{var yy=yS(t);g.appendChild(mel("line",{{x1:0,y1:yy,x2:pw,y2:yy,stroke:"#1a2535","stroke-width":"1"}}));g.appendChild(mtxt((t*100).toFixed(0)+"%",{{x:-4,y:yy+3,"text-anchor":"end","font-size":"8"}},"#607080"));}});
  g.appendChild(mtxt(vn+" (% area in drought)",{{x:pw/2,y:ph+31,"text-anchor":"middle","font-size":"8.5"}},"#90a4ae"));
  var ayl=mel("text",{{x:0,y:0,"text-anchor":"middle","font-size":"8.5",fill:"#90a4ae",transform:"rotate(-90) translate("+(-(ph/2))+",-32)"}});ayl.textContent="Abandonment ratio";g.appendChild(ayl);
  if(rx.length===2)g.appendChild(mel("line",{{x1:xS(rx[0]),y1:yS(ry[0]),x2:xS(rx[1]),y2:yS(ry[1]),stroke:VCOL[vn],"stroke-width":"1.5","stroke-dasharray":"5,3",opacity:"0.65"}}));
  if(cx!==null&&clo!==null&&chi!==null){{var xcx=xS(cx),yclo=yS(clo),ychi=yS(chi);g.appendChild(mel("line",{{x1:xcx,y1:ychi,x2:xcx,y2:yclo,stroke:"#fff","stroke-width":"2",opacity:"0.55"}}));g.appendChild(mel("line",{{x1:xcx-6,y1:yclo,x2:xcx+6,y2:yclo,stroke:"#fff","stroke-width":"2",opacity:"0.55"}}));g.appendChild(mel("line",{{x1:xcx-6,y1:ychi,x2:xcx+6,y2:ychi,stroke:"#fff","stroke-width":"2",opacity:"0.55"}}));}}
  var maxI=0,minI=0;
  for(var i=0;i<xs.length;i++){{tt(g.appendChild(mel("circle",{{cx:xS(xs[i]),cy:yS(ys[i]),r:"4.5",fill:VCOL[vn],opacity:"0.7",stroke:"#0d1117","stroke-width":"0.5"}})),yrs[i]+": drought="+xs[i].toFixed(1)+"%, abandon="+(ys[i]*100).toFixed(1)+"%");if(ys[i]>ys[maxI])maxI=i;if(ys[i]<ys[minI])minI=i;}}
  if(xs.length>0){{[maxI,minI].forEach(function(i){{g.appendChild(mtxt(String(yrs[i]),{{x:xS(xs[i])+6,y:yS(ys[i])-4,"font-size":"8"}},"#a0b0c0"));}});}}
  if(cx!==null&&cy!==null){{var sx=xS(cx),sy=yS(cy);g.appendChild(mel("path",{{d:starPath(sx,sy,11,4.5,5),fill:"#fff",stroke:"#f0a020","stroke-width":"1.5"}}));var onR=sx<=pw*0.60,anch=onR?"start":"end",lx=sx+(onR?14:-14),onT=sy>ph*0.35,ly=sy+(onT?-16:17);g.appendChild(mtxt((cyr||"Curr")+": "+(cy*100).toFixed(1)+"%",{{x:lx,y:ly,"text-anchor":anch,"font-size":"9.5","font-weight":"bold"}},"#ffffff"));if(clo!==null&&chi!==null)g.appendChild(mtxt(CI_PCT+" CI: ["+(clo*100).toFixed(1)+"–"+(chi*100).toFixed(1)+"%]",{{x:lx,y:ly+13,"text-anchor":anch,"font-size":"7.5"}},"#c0c0c0"));}}
  var pv=d.pvalue,col2=pv!==null?(pv<0.05?"#68d391":pv<0.1?"#f6e05e":"#607080"):"#607080";
  g.appendChild(mtxt("R²="+(d.r2!==null?(d.r2*100).toFixed(1)+"%":"—")+"  "+(pv!==null?(pv<0.05?"★ p<0.05":pv<0.1?"▲ p<0.10":"p="+pv.toFixed(2)):""),{{x:pw-2,y:12,"text-anchor":"end","font-size":"8.5","font-weight":"600"}},col2));
  svg.appendChild(g);c.appendChild(svg);
}}

// ── Table ─────────────────────────────────────────────────────────────────
function drawTableSet(tblId, rows, rowLabelKey){{
  var c=document.getElementById(tblId);if(!c)return;
  if(!rows||!rows.length){{c.innerHTML='<div style="padding:18px;color:#607080;text-align:center">No data for current season yet.</div>';return;}}
  var vH=VARS.map(function(v){{return '<th class="vh" colspan="3" style="color:'+VCOL[v]+';background:'+VBG[v]+'">'+v+'</th>';}}).join("");
  var sH=VARS.map(function(){{return '<th class="vs">Pred%</th><th class="vs">'+CI_PCT+' CI</th><th class="vs">R²</th>';}}).join("");
  var tb="";
  rows.forEach(function(row){{
    var lbl=row[rowLabelKey]||row.label||"?";
    var sublbl=row.date?' <span style="font-size:.61rem;color:#405060;font-weight:400">'+row.date+'</span>':'';
    var cells=VARS.map(function(v){{
      var d=row.vars&&row.vars[v];
      if(!d||d.point===null||d.point===undefined)return '<td class="pt">—</td><td class="ci">—</td><td>—</td>';
      var pt=(d.point*100).toFixed(1)+"%";
      var lo=d.lo!==null?(d.lo*100).toFixed(1)+"%":"?";
      var hi=d.hi!==null?(d.hi*100).toFixed(1)+"%":"?";
      var r2v=d.r2!=null?d.r2:0,pv=d.pvalue;
      var col=pv===null?"#607080":pv<0.05?"#68d391":pv<0.1?"#f6e05e":"#607080";
      var pct=Math.round((r2v||0)*100);
      return '<td class="pt">'+pt+'</td><td class="ci">['+lo+'–'+hi+']</td>'+
        '<td><div class="r2w"><div class="r2bg"><div class="r2f" style="width:'+pct+'%;background:'+col+'"></div></div>'+
        '<span style="font-size:.66rem;color:'+col+'">'+pct+'%</span></div></td>';
    }}).join("");
    var bestV=null,bestR2=-1;
    VARS.forEach(function(v){{var d=row.vars&&row.vars[v];if(d&&d.r2!==null&&d.r2>bestR2){{bestR2=d.r2;bestV=v;}}}});
    var bd=bestV?row.vars[bestV]:null;
    var bpt=bd&&bd.point!==null?(bd.point*100).toFixed(1)+"%":"—";
    var blo=bd&&bd.lo!==null?(bd.lo*100).toFixed(1)+"%":"?";
    var bhi=bd&&bd.hi!==null?(bd.hi*100).toFixed(1)+"%":"?";
    tb+='<tr><td class="wk">'+lbl+sublbl+'</td>'+cells+
      '<td class="bv" style="color:'+(VCOL[bestV]||"#c4a0f0")+'">'+(bestV||"—")+'</td>'+
      '<td class="bp">'+bpt+'</td><td class="bc">['+blo+'–'+bhi+']</td></tr>';
  }});
  c.innerHTML='<table><thead><tr><th class="wk" rowspan="2">Period</th>'+vH+
    '<th class="bh" colspan="3" style="color:#c4a0f0">★ Best Variable</th></tr>'+
    '<tr>'+sH+'<th class="bs">Var</th><th class="bs">Pred%</th><th class="bs">'+CI_PCT+' CI</th></tr>'+
    '</thead><tbody>'+tb+'</tbody></table>';
}}

// ── Diagnostics ───────────────────────────────────────────────────────────
function drawDiag(){{
  var grid=document.getElementById("dg-grid");if(!grid)return;grid.innerHTML="";
  VARS.forEach(function(v){{
    var card=document.createElement("div");card.className="dg-card";
    var did="dg_"+v.replace(/[^a-z0-9]/gi,"_");
    card.innerHTML='<h3 style="color:'+VCOL[v]+'">'+v+' — R² by period</h3><div id="'+did+'"></div>';
    grid.appendChild(card);
    var labels=[],wkR2=[],moR2=[],cuR2=[];
    WK_ROWS.forEach(function(r){{labels.push(r.label);var d=r.vars[v];wkR2.push(d?d.r2:null);}});
    // monthly — align by label
    var moMap={{}};MO_ROWS.forEach(function(r){{var d=r.vars[v];moMap[r.label]=(d?d.r2:null);}});
    moR2=labels.map(function(l){{return moMap[l.split(" ")[0]]||null;}});
    CU_ROWS.forEach(function(r,i){{var d=r.vars[v];cuR2.push(d?d.r2:null);}});
    drawDiagBars(did,labels,wkR2,moR2,cuR2);
  }});
}}

function drawDiagBars(cid,labels,wkR2,moR2,cuR2){{
  var c=document.getElementById(cid);if(!c)return;c.innerHTML="";
  if(!labels.length){{c.innerHTML='<div style="color:#607080;padding:10px;font-size:.72rem">No data</div>';return;}}
  var cw=c.clientWidth||330,ML=28,MR=8,MB=40,svgH=130;
  var n=labels.length,bW=Math.max(3,Math.floor((cw-ML-MR)/n/3)-0.5),gW=bW*3+2;
  var allR=[].concat(wkR2,moR2,cuR2).filter(function(v){{return v!==null;}});
  var yMax=allR.length?Math.max(Math.max.apply(null,allR),0.05):0.15;
  var yS=scl(0,yMax,svgH-MB,8),svg=msvg(cw,svgH);
  [0,0.05,0.10,0.15,0.20].filter(function(t){{return t<=yMax+0.01;}}).forEach(function(t){{
    var yy=yS(t);
    svg.appendChild(mel("line",{{x1:ML,y1:yy,x2:cw-MR,y2:yy,stroke:"#1a2535","stroke-width":"1"}}));
    svg.appendChild(mtxt((t*100).toFixed(0)+"%",{{x:ML-2,y:yy+3,"text-anchor":"end","font-size":"7"}},"#607080"));
  }});
  // p=0.05 significance line (approx R²≈0.15 for n=25)
  var sig_r2=0.15;
  if(sig_r2<=yMax){{var yy=yS(sig_r2);svg.appendChild(mel("line",{{x1:ML,y1:yy,x2:cw-MR,y2:yy,stroke:"#68d391","stroke-width":"1","stroke-dasharray":"3,3"}}));svg.appendChild(mtxt("~p=0.05",{{x:cw-MR-1,y:yy-2,"text-anchor":"end","font-size":"7"}},"#68d391"));}}
  for(var i=0;i<n;i++){{
    var gx=ML+i*(gW+1);
    [[wkR2[i],MCOL.weekly],[moR2[i],MCOL.monthly],[cuR2[i],MCOL.cumulative]].forEach(function(item,j){{
      var rv=item[0]||0,col=item[1],yy=yS(rv),bh=Math.max(1,(svgH-MB)-yy);
      tt(svg.appendChild(mel("rect",{{x:gx+j*(bW+0.5),y:yy,width:bW,height:bh,rx:1,fill:col,opacity:"0.85"}})),
        labels[i]+": "+["Weekly","Monthly","Cumulative"][j]+" R²="+(item[0]!==null?(item[0]*100).toFixed(1)+"%":"N/A"));
    }});
    if(i%3===0){{var tx=mel("text",{{x:gx+gW/2,y:svgH-MB+9,"text-anchor":"middle",transform:"rotate(-40,"+(gx+gW/2)+","+(svgH-MB+9)+")","font-size":"7","fill":"#607080"}});tx.textContent=labels[i];svg.appendChild(tx);}};
  }}
  [["Weekly",MCOL.weekly],["Monthly",MCOL.monthly],["Cumul",MCOL.cumulative]].forEach(function(item,i){{
    var lx=ML+i*68;
    svg.appendChild(mel("rect",{{x:lx,y:svgH-11,width:8,height:6,rx:1,fill:item[1]}}));
    svg.appendChild(mtxt(item[0],{{x:lx+10,y:svgH-6,"font-size":"7"}},item[1]));
  }});
  c.appendChild(svg);
}}

// ── Init ──────────────────────────────────────────────────────────────────

// ══════════════════════════════════════════════════════════════════════════
// SEASONALITY LINE CHARTS
// ══════════════════════════════════════════════════════════════════════════
var ANALOG_COLS = ["#ff6b6b","#ffd93d","#6bcb77","#4d96ff","#c77dff"];
var ANALOGS = {{}};

function initAnalogs(){{
  VARS.forEach(function(v){{
    var d=SLINES.variables[v];
    ANALOGS[v]=d?d.analogs.slice():[];
  }});
  var sel=document.getElementById("sl-year"); if(!sel) return;
  var d=SLINES.variables["D1-D4"]; if(!d) return;
  d.all_hist_years.forEach(function(yr){{
    sel.innerHTML+='<option value="'+yr+'">'+yr+'</option>';
  }});
}}

function swapAnalog(){{
  var slot=parseInt(document.getElementById("sl-slot").value)||0;
  var yr=parseInt(document.getElementById("sl-year").value);
  if(isNaN(yr)) return;
  VARS.forEach(function(v){{
    if(slot<ANALOGS[v].length) ANALOGS[v][slot]=yr;
    else ANALOGS[v].push(yr);
  }});
  drawSeasonCharts();
}}

function drawSeasonCharts(){{
  if(!SLINES||!SLINES.variables) return;
  var selVar=document.getElementById("sl-var")&&document.getElementById("sl-var").value;
  var grid=document.getElementById("season-grid"); if(!grid) return;
  grid.innerHTML="";
  var toShow = (selVar==="ALL"||!selVar) ? VARS : [selVar];
  // 2-per-row grid
  grid.style.gridTemplateColumns = toShow.length===1 ? "1fr" : "1fr 1fr";
  toShow.forEach(function(v){{
    var card=document.createElement("div"); card.className="chart-card";
    var cid="sl_"+v.replace(/[^a-z0-9]/gi,"_");
    card.innerHTML='<h3 style="color:'+VCOL[v]+'">'+v+'</h3><div id="'+cid+'"></div>';
    grid.appendChild(card);
    drawOneSeasonLine(cid,v);
  }});
}}

function drawOneSeasonLine(cid,vn){{
  var c=document.getElementById(cid); if(!c) return; c.innerHTML="";
  var d=SLINES.variables[vn]; if(!d) return;
  var weeks=SLINES.iso_weeks, wlabels=SLINES.weeks;
  var currYr=SLINES.curr_yr, analogs=ANALOGS[vn]||[], series=d.series;
  var allYrs=d.all_hist_years;
  var cw=c.clientWidth||340, ML=38,MR=14,MT=14,MB=50, svgH=195, pw=cw-ML-MR, ph=svgH-MT-MB;

  var allV=[];
  Object.keys(series).forEach(function(yr){{ (series[yr]||[]).forEach(function(v){{ if(v!==null) allV.push(v); }}); }});
  var yMax=allV.length?Math.min(100,Math.max.apply(null,allV)):100;
  var xS=scl(0,weeks.length-1,0,pw), yS=scl(0,yMax,ph,0);
  var svg=msvg(cw,svgH), g=mel("g",{{transform:"translate("+ML+","+MT+")"}});

  // Y gridlines
  [0,25,50,75,100].filter(function(t){{return t<=yMax+1;}}).forEach(function(t){{
    var yy=yS(t);
    g.appendChild(mel("line",{{x1:0,y1:yy,x2:pw,y2:yy,stroke:"#1a2535","stroke-width":"1"}}));
    g.appendChild(mtxt(t+"%",{{x:-4,y:yy+3,"text-anchor":"end","font-size":"7"}},"#607080"));
  }});
  // X axis
  wlabels.forEach(function(lbl,i){{
    if(i%4===0){{
      var xx=xS(i);
      g.appendChild(mel("line",{{x1:xx,y1:ph,x2:xx,y2:ph+3,stroke:"#607080"}}));
      var tx=mel("text",{{x:xx,y:ph+13,"text-anchor":"middle",
        transform:"rotate(-40,"+xx+","+(ph+13)+")","font-size":"7","fill":"#607080"}});
      tx.textContent=lbl; g.appendChild(tx);
    }}
  }});

  function makeLine(yr,col,sw,op,dash){{
    var line=series[String(yr)]; if(!line) return;
    var pts=[]; line.forEach(function(v,i){{if(v!==null)pts.push([xS(i),yS(v)]);}});
    if(pts.length<2) return;
    var pd="M"+pts[0][0].toFixed(1)+","+pts[0][1].toFixed(1);
    for(var i=1;i<pts.length;i++) pd+="L"+pts[i][0].toFixed(1)+","+pts[i][1].toFixed(1);
    var a={{d:pd,stroke:col,"stroke-width":String(sw),fill:"none",opacity:String(op)}};
    if(dash) a["stroke-dasharray"]=dash;
    tt(g.appendChild(mel("path",a)),String(yr));
  }}

  // 1. All historical — thin grey
  allYrs.forEach(function(yr){{ if(analogs.indexOf(yr)<0) makeLine(yr,"#2a3a4a",0.7,0.5,null); }});

  // 2. Analog years — colored dashed + end label
  analogs.forEach(function(yr,i){{
    var col=ANALOG_COLS[i%5];
    makeLine(yr,col,1.8,0.85,"5,3");
    var line=series[String(yr)]; if(!line) return;
    var lv=null,li=-1; line.forEach(function(v,j){{if(v!==null){{lv=v;li=j;}}}});
    if(lv!==null) g.appendChild(mtxt(String(yr),{{x:xS(li)+4,y:yS(lv)+3,"font-size":"7.5","font-weight":"600"}},col));
  }});

  // 3. Current year — bold solid
  makeLine(currYr,VCOL[vn]||"#68d391",2.5,1,null);
  var cl=series[String(currYr)];
  if(cl){{
    var lv=null,li=-1; cl.forEach(function(v,j){{if(v!==null){{lv=v;li=j;}}}});
    if(lv!==null){{
      g.appendChild(mel("circle",{{cx:xS(li),cy:yS(lv),r:"4",fill:VCOL[vn]||"#68d391"}}));
      g.appendChild(mtxt(currYr+": "+lv.toFixed(1)+"%",{{x:xS(li)+(li>weeks.length*0.7?-6:6),y:yS(lv)-6,
        "text-anchor":li>weeks.length*0.7?"end":"start","font-size":"8","font-weight":"700"}},VCOL[vn]||"#68d391"));
    }}
  }}

  // Vertical line at latest available week
  var latIdx=weeks.indexOf(SLINES.latest_iw);
  if(latIdx>=0){{
    var xx=xS(latIdx);
    g.appendChild(mel("line",{{x1:xx,y1:0,x2:xx,y2:ph,stroke:"#607080","stroke-width":"1","stroke-dasharray":"3,3"}}));
  }}

  svg.appendChild(g); c.appendChild(svg);

  // Mini legend below chart
  var leg=document.createElement("div");
  leg.style.cssText="display:flex;flex-wrap:wrap;gap:5px;margin-top:4px;font-size:.66rem";
  var items=[[String(currYr),VCOL[vn]||"#68d391","700"]];
  analogs.forEach(function(yr,i){{items.push([String(yr),ANALOG_COLS[i%5],"600"]);}});
  items.push(["Other yrs","#405060","400"]);
  items.forEach(function(item){{
    leg.innerHTML+='<span style="display:flex;align-items:center;gap:2px">'+
      '<span style="width:12px;height:2px;background:'+item[1]+';display:inline-block;border-radius:1px"></span>'+
      '<span style="color:'+item[1]+';font-weight:'+item[2]+'">'+item[0]+'</span></span>';
  }});
  c.appendChild(leg);
}}

function drawLast6(){{
  var c=document.getElementById("last6-tbl"); if(!c) return;
  var rows=SLINES.last6_weeks;
  if(!rows||!rows.length){{c.innerHTML='<div style="color:#607080;padding:10px">No data.</div>';return;}}
  var hdr='<tr><th class="wk">Week</th><th class="wk" style="color:#607080">Date</th>'+
    VARS.map(function(v){{return '<th style="color:'+VCOL[v]+'">'+v+'</th>';}}).join("")+'</tr>';
  var body=rows.map(function(r){{
    return '<tr><td class="wk">'+r.label+'</td><td class="wk" style="color:#607080">'+r.date+'</td>'+
      VARS.map(function(v){{var val=r[v];return '<td class="pt">'+(val!=null?val.toFixed(1)+"%":"—")+'</td>';}}).join("")+'</tr>';
  }}).join("");
  c.innerHTML='<table><thead>'+hdr+'</thead><tbody>'+body+'</tbody></table>';
}}

// ══════════════════════════════════════════════════════════════════════════
// PRODUCTION TAB
// ══════════════════════════════════════════════════════════════════════════
function renderProduction(){{
  var P=parseInt(document.getElementById("period-sel").value)||5;

  // Banner
  var bn=document.getElementById("best-pred-banner");
  if(bn){{
    if(BEST_TX&&BEST_TX.point!==null){{
      bn.innerHTML='🎯 <b>Best TX abandonment model:</b> '+BEST_TX.variable+
        ' ('+BEST_TX.model+', as of '+BEST_TX.label+') &nbsp;·&nbsp; R²='+(BEST_TX.r2*100).toFixed(1)+
        '% &nbsp;·&nbsp; TX abandonment prediction: <b style="color:#68d391">'+(BEST_TX.point*100).toFixed(1)+
        '%</b> ['+((BEST_TX.lo||0)*100).toFixed(1)+'–'+((BEST_TX.hi||0)*100).toFixed(1)+'%] ('+CI_PCT+' CI)';
    }} else {{
      bn.innerHTML='⚠ No regression prediction available yet.';
    }}
  }}

  var bestAb = BEST_TX&&BEST_TX.point!==null ? BEST_TX.point : null;
  buildStateTable("matA", P, null);
  buildStateTable("matB", P, bestAb);
  buildGrid("gridC", "TX", TX_PLANT, PROD.tx_ab_range, PROD.tx_yld_range, bestAb, P);
  buildGrid("gridD", PROD.us_geo, null, PROD.tx_ab_range, PROD.us_yld_range, bestAb, P);
}}

function getStateData(st, P){{
  var sd=PROD.states[st]; if(!sd) return null;
  if(P===1) return sd.last_yr ? {{
    ab:sd.last_yr.ab, yld:sd.last_yr.yld, plt:sd.last_yr.plt, prd:sd.last_yr.prd
  }} : null;
  var pd=sd.periods&&sd.periods[P]; if(!pd) return null;
  return {{ab:pd.ab_avg, yld:pd.yld_avg, plt:pd.plt_avg, prd:pd.prd_avg}};
}}

function buildStateTable(cid, P, txAbOverride){{
  var c=document.getElementById(cid); if(!c) return;
  var stateList=Object.keys(PROD.states).filter(function(s){{return s!==PROD.us_geo;}}).sort();
  stateList.push(PROD.us_geo);

  var hdr='<tr><th class="wk">State</th><th>Planted<br>(K ac)</th>'+
    '<th>Abandon %</th><th>Yield<br>(lb/ac)</th><th>Prod<br>(mn bales)</th></tr>';
  var rows="";
  stateList.forEach(function(st){{
    var d=getStateData(st,P); if(!d) return;
    var ab  = (st==="TX"&&txAbOverride!==null) ? txAbOverride : d.ab;
    var yld = d.yld, plt = d.plt, prd = d.prd;
    var prod=null;
    if(plt!==null&&ab!==null&&yld!==null){{
      prod=plt*1000*(1-ab)*yld/480000000;
    }} else if(prd!==null) {{
      prod=prd/1000;
    }}
    var isTX=st==="TX", isUS=st===PROD.us_geo;
    var txMod=isTX&&txAbOverride!==null;
    rows+='<tr style="'+(isUS?"background:#0e1d2e":"")+'">'
      +'<td class="wk" style="color:'+(isTX?"#68d391":isUS?"#7fb3d3":"#c8d8e8")+'">'+st+(txMod?' <span style="font-size:.59rem;color:#68d391">★model</span>':'')+'</td>'
      +'<td>'+(plt!==null?Math.round(plt).toLocaleString():"—")+'</td>'
      +'<td style="color:'+(txMod?"#68d391":"#f6e05e")+'">'+(ab!==null?(ab*100).toFixed(1)+"%":"—")+'</td>'
      +'<td>'+(yld!==null?Math.round(yld):"—")+'</td>'
      +'<td style="color:#c4a0f0;font-weight:600">'+(prod!==null?prod.toFixed(3):"—")+'</td>'
      +'</tr>';
  }});
  c.innerHTML='<table><thead>'+hdr+'</thead><tbody>'+rows+'</tbody></table>';
}}

function buildGrid(cid, stLabel, planted_k, abRange, yldRange, modelAb, P){{
  var c=document.getElementById(cid); if(!c) return;

  // For US grid use planned area from states data
  if(!planted_k && stLabel===PROD.us_geo){{
    var usd=getStateData(PROD.us_geo, P);
    planted_k=usd&&usd.plt?usd.plt:null;
  }}

  if(!planted_k||!abRange||!abRange.length||!yldRange||!yldRange.length){{
    c.innerHTML='<div style="color:#fc8181;font-size:.71rem;padding:8px">'+
      (stLabel==="TX"?'TX planted area not set in data/config.json':'Planted data not available')+
      '</div>';
    return;
  }}

  // Find model abandonment row
  var modelRow=null;
  if(modelAb!==null){{
    var diffs=abRange.map(function(a){{return Math.abs(a-modelAb);}});
    modelRow=diffs.indexOf(Math.min.apply(null,diffs));
  }}

  var hdr='<tr><th style="text-align:left;background:#0e1d2e;position:sticky;left:0;min-width:58px">Ab%↓ Yld→</th>'+
    yldRange.map(function(y){{return '<th style="background:#0e1d2e;color:#7fb3d3">'+y+'</th>';}}).join("")+'</tr>';

  var body=abRange.map(function(ab,ri){{
    var isMod=ri===modelRow;
    var cells=yldRange.map(function(yld){{
      var prod=planted_k*1000*(1-ab)*yld/480000000;
      return '<td style="'+(isMod?"color:#68d391;font-weight:600":"")+'">'+prod.toFixed(2)+'</td>';
    }}).join("");
    return '<tr style="'+(isMod?"background:#0a1f14":"")+'"><th style="color:'+(isMod?"#68d391":"#90a4ae")+';background:'+(isMod?"#0a1f14":"#111820")+';position:sticky;left:0">'+
      (ab*100).toFixed(0)+"%"+(isMod?" ★":"")+'</th>'+cells+'</tr>';
  }}).join("");

  c.innerHTML='<div style="font-size:.66rem;color:#607080;margin-bottom:4px">'+
    stLabel+' planted: '+Math.round(planted_k).toLocaleString()+'K ac · mn 480-lb bales'+(modelAb!==null?" · ★ = model row":"")+
    '</div><div style="overflow-x:auto"><table class="mtbl" style="border-collapse:collapse;font-size:.69rem;white-space:nowrap">'+
    '<thead>'+hdr+'</thead><tbody>'+body+'</tbody></table></div>';
}}

window.onload=function(){{
  initAnalogs();
  drawSeasonCharts();
  drawLast6();
  drawScatterSet("sc-w", WK_ROWS);
  drawScatterSet("sc-m", MO_ROWS);
  drawScatterSet("sc-c", CU_ROWS);
  drawTableSet("tbl-w", WK_ROWS, "label");
  drawTableSet("tbl-m", MO_ROWS, "label");
  drawTableSet("tbl-c", CU_ROWS, "label");
}};
window.onresize=function(){{
  drawSeasonCharts();
  drawScatterSet("sc-w",WK_ROWS);
  drawScatterSet("sc-m",MO_ROWS);
  drawScatterSet("sc-c",CU_ROWS);
  var dg=document.getElementById("dg-grid");
  if(dg&&dg.children.length>1) drawDiag();
}};
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("Texas Cotton Abandonment — Drought Predictor + Production Estimator")
    print("=" * 60)

    if not COTTON_CSV.exists():
        print(f"\nERROR: {COTTON_CSV} not found. Place CSV in data/")
        import sys; sys.exit(1)
    if not DROUGHT_CSV.exists():
        print(f"\nERROR: {DROUGHT_CSV} not found. Place CSV in data/")
        import sys; sys.exit(1)

    print("\nLoading config…")
    config = load_config()

    print("Loading cotton data (TX regression)…")
    cotton = load_cotton(COTTON_CSV)

    print("Loading cotton data (all states)…")
    cotton_all = load_cotton_all(COTTON_ALL_CSV)

    print("Loading drought data…")
    drought = load_drought(DROUGHT_CSV)

    print("\nBuilding regression models…")
    wk_rows, curr_yr = build_weekly(cotton, drought)
    print(f"  Weekly:     {len(wk_rows)} rows")
    mo_rows = build_monthly(cotton, drought)
    print(f"  Monthly:    {len(mo_rows)} rows")
    cu_rows = build_cumulative(cotton, drought)
    print(f"  Cumulative: {len(cu_rows)} rows")

    print("Finding best TX prediction…")
    best_tx = best_tx_prediction(wk_rows, mo_rows, cu_rows)
    print(f"  Best: {best_tx.get('variable','—')} ({best_tx.get('model','—')}) "
          f"R²={best_tx.get('r2',0)*100:.1f}% → {(best_tx.get('point') or 0)*100:.1f}%")

    print("Building seasonality line chart data…")
    season_lines = build_seasonality_lines(drought)

    print("Building production data…")
    max_yr = int(cotton.index.max())
    prod_data = build_production_data(cotton_all, max_yr)
    print(f"  States: {len(prod_data['states'])} · "
          f"TX ab range: {prod_data['tx_ab_range'][0]*100:.0f}%–"
          f"{prod_data['tx_ab_range'][-1]*100:.0f}%")

    ci_pct = str(int(CI_LEVEL * 100))
    print(f"\nGenerating HTML ({ci_pct}% CI)…")
    OUTPUT_HTML.parent.mkdir(exist_ok=True)
    html = make_html(wk_rows, mo_rows, cu_rows, season_lines,
                     prod_data, best_tx, config, cotton, drought, ci_pct)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"\n✓ Done  →  {OUTPUT_HTML}  ({OUTPUT_HTML.stat().st_size//1024} KB)")
    print("\nOpen docs/index.html in any browser.")


if __name__ == "__main__":
    main()
