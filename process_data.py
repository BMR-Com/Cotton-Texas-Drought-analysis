"""
Texas Cotton Abandonment — Drought Predictor
=============================================
USAGE:
    python process_data.py

INPUTS  (place in same folder as this script):
    cotton_texas.csv   — USDA NASS upland cotton planted/harvested for TX
    drought_texas.csv  — USDA Drought Monitor weekly TX cotton area data

OUTPUT:
    cotton_drought_dashboard.html  — fully self-contained, open in any browser

INSTALL:
    pip install numpy pandas scipy

NO web server needed. No GitHub. Just run and open the HTML file.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
CI_LEVEL      = 0.90          # 90% prediction interval
MIN_TRAIN_YRS = 15            # minimum years before first prediction
DROUGHT_VARS  = ["D1-D4", "D2-D4", "D3-D4", "D4"]
SEASON_START  = 14            # ISO week ~ Apr W1
SEASON_END    = 43            # ISO week ~ Oct W4
COTTON_CSV    = Path("data/cotton_texas.csv")
DROUGHT_CSV   = Path("data/drought_texas.csv")
OUTPUT_HTML   = Path("docs/index.html")

# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
def load_cotton(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df[df["geography"].str.strip() == "TX"]
    df = df[df["category"].isin([
        "upland_cotton_planted_acreage",
        "upland_cotton_harvested_acreage"
    ])]
    df["period"] = pd.to_numeric(df["period"], errors="coerce")
    df["value"]  = pd.to_numeric(df["value"],  errors="coerce")
    df = df.dropna(subset=["period", "value"])
    pivot = df.pivot_table(
        index="period", columns="category", values="value", aggfunc="first"
    ).reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={"period": "mkt_year"})
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
    return pivot[["mkt_year", "abandonment",
                  "upland_cotton_planted_acreage",
                  "upland_cotton_harvested_acreage"]].set_index("mkt_year")


def load_drought(path: Path) -> pd.DataFrame:
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
    df["mkt_year"] = df.apply(
        lambda r: r["cal_year"] if r["Week"].month >= 4 else r["cal_year"] - 1,
        axis=1
    )
    df = df[(df["iso_week"] >= SEASON_START) & (df["iso_week"] <= SEASON_END)]
    print(f"  Drought: {len(df)} weekly rows "
          f"({df['Week'].min().date()} – {df['Week'].max().date()})")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. WEEK LABELS
# ═══════════════════════════════════════════════════════════════════════════
def week_label(iso_week: int) -> str:
    """Map ISO week number to human label like 'Apr W1'."""
    # Use a reference year to map iso_week → month
    ref = pd.Timestamp("2019-01-01") + pd.to_timedelta(iso_week * 7 - 4, unit="D")
    months = {4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct"}
    m = months.get(ref.month, f"M{ref.month}")
    wom = (ref.day - 1) // 7 + 1
    return f"{m} W{wom}"


# ═══════════════════════════════════════════════════════════════════════════
# 3. OLS WITH PREDICTION INTERVAL
# ═══════════════════════════════════════════════════════════════════════════
def ols_predict(x_tr, y_tr, x_pred, ci=CI_LEVEL):
    """
    Fit y ~ x via OLS. Return (point, lo, hi, r2, pvalue, slope, intercept).
    x_pred can be scalar or 1-D array.
    """
    mask = np.isfinite(x_tr) & np.isfinite(y_tr)
    x_tr, y_tr = x_tr[mask], y_tr[mask]
    n = len(x_tr)
    if n < 5:
        nan = np.nan
        return nan, nan, nan, nan, nan, nan, nan

    slope, intercept, r, p, _ = stats.linregress(x_tr, y_tr)
    r2 = r ** 2

    x_pred_arr = np.atleast_1d(np.asarray(x_pred, dtype=float))
    x_mean = x_tr.mean()
    ss_x   = np.sum((x_tr - x_mean) ** 2)
    y_hat  = intercept + slope * x_tr
    s      = np.sqrt(np.sum((y_tr - y_hat) ** 2) / (n - 2))  # residual std

    t_crit   = stats.t.ppf((1 + ci) / 2, df=n - 2)
    se_pred  = s * np.sqrt(1 + 1/n + (x_pred_arr - x_mean)**2 / ss_x)
    y_pred   = intercept + slope * x_pred_arr
    lo       = np.clip(y_pred - t_crit * se_pred, 0, 1)
    hi       = np.clip(y_pred + t_crit * se_pred, 0, 1)
    y_pred   = np.clip(y_pred, 0, 1)

    if x_pred_arr.size == 1:
        return (float(y_pred[0]), float(lo[0]), float(hi[0]),
                float(r2), float(p), float(slope), float(intercept))
    return y_pred, lo, hi, float(r2), float(p), float(slope), float(intercept)


# ═══════════════════════════════════════════════════════════════════════════
# 4. BUILD ALL MODELS
# ═══════════════════════════════════════════════════════════════════════════
def build_models(cotton: pd.DataFrame, drought: pd.DataFrame):
    """
    For each ISO week × drought variable × approach (single / cumulative):
    - Fit rolling OLS on all prior years (min MIN_TRAIN_YRS)
    - Store scatter data, regression line, current-season prediction

    Returns: list of row-dicts, one per ISO week, sorted Apr W1 → Oct W4
    """
    # All ISO weeks in season order
    all_weeks = sorted(
        w for w in drought["iso_week"].unique()
        if SEASON_START <= w <= SEASON_END
    )

    # Pre-build per-year drought features for every variable × approach
    # single[var][mkt_year] = drought value at that exact iso_week
    # cumul[var][mkt_year]  = mean drought from SEASON_START up to iso_week
    rows = []

    for iso_week in all_weeks:
        label = week_label(iso_week)
        row = {"iso_week": int(iso_week), "label": label, "vars": {}}

        for approach in ("single", "cumulative"):
            # Build per-year drought values up to this iso_week
            yd = {v: {} for v in DROUGHT_VARS}
            for mkt_year, grp in drought.groupby("mkt_year"):
                sub = grp[grp["iso_week"] <= iso_week]
                if sub.empty:
                    continue
                if approach == "single":
                    # row closest to this iso_week
                    idx = (sub["iso_week"] - iso_week).abs().idxmin()
                    r   = sub.loc[idx]
                    for v in DROUGHT_VARS:
                        yd[v][mkt_year] = float(r[v]) if v in r and pd.notna(r[v]) else np.nan
                else:
                    for v in DROUGHT_VARS:
                        if v in sub.columns:
                            yd[v][mkt_year] = float(sub[v].mean())

            for v in DROUGHT_VARS:
                common = sorted(set(yd[v].keys()) & set(cotton.index))
                if len(common) < MIN_TRAIN_YRS + 1:
                    row["vars"].setdefault(v, {})[approach] = None
                    continue

                x_all = np.array([yd[v][y] for y in common])
                y_all = np.array([cotton.loc[y, "abandonment"] for y in common])

                # Overall model (all known years)
                _, _, _, r2, pv, sl, ic = ols_predict(x_all, y_all, x_all[0])

                # Regression line endpoints
                xv = x_all[np.isfinite(x_all)]
                reg_x = [round(float(xv.min()), 2), round(float(xv.max()), 2)] if len(xv) else []
                reg_y = ([round(float(np.clip(ic + sl * reg_x[0], 0, 1)), 4),
                           round(float(np.clip(ic + sl * reg_x[1], 0, 1)), 4)]
                          if reg_x else [])

                # Clean scatter arrays
                mask = np.isfinite(x_all) & np.isfinite(y_all)
                sx = [round(float(v2), 2) for v2 in x_all[mask]]
                sy = [round(float(v2), 4) for v2 in y_all[mask]]
                sy_list = [int(y) for y, m in zip(common, mask) if m]

                # Current-season prediction (latest year not yet in cotton)
                latest_yr = max(yd[v].keys())
                if latest_yr not in cotton.index:
                    x_curr = yd[v][latest_yr]
                    pt, lo, hi, _, _, _, _ = ols_predict(x_all, y_all, x_curr)
                    curr = {
                        "year":  int(latest_yr),
                        "x":     round(float(x_curr), 2) if np.isfinite(x_curr) else None,
                        "point": round(pt, 4) if np.isfinite(pt) else None,
                        "lo":    round(lo, 4) if np.isfinite(lo) else None,
                        "hi":    round(hi, 4) if np.isfinite(hi) else None,
                    }
                else:
                    curr = {"year": None, "x": None, "point": None, "lo": None, "hi": None}

                row["vars"].setdefault(v, {})[approach] = {
                    "r2":      round(float(r2), 4) if np.isfinite(r2) else None,
                    "pvalue":  round(float(pv), 4) if np.isfinite(pv) else None,
                    "reg_x":   reg_x,
                    "reg_y":   reg_y,
                    "scatter_x":     sx,
                    "scatter_y":     sy,
                    "scatter_years": sy_list,
                    "curr":    curr,
                }

        # Best variable (by R² in single approach)
        best_var, best_r2 = None, -1.0
        for v in DROUGHT_VARS:
            d = row["vars"].get(v, {}).get("single")
            if d and d["r2"] is not None and d["r2"] > best_r2:
                best_r2 = d["r2"]
                best_var = v
        row["best_var"] = best_var
        rows.append(row)

    return rows


# ═══════════════════════════════════════════════════════════════════════════
# 5. SEASONALITY HEATMAP DATA
# ═══════════════════════════════════════════════════════════════════════════
def build_seasonality(drought: pd.DataFrame, n_years: int = 20) -> dict:
    max_yr = drought["cal_year"].max()
    df = drought[drought["cal_year"] >= max_yr - n_years + 1].copy()
    all_weeks = sorted(
        w for w in df["iso_week"].unique()
        if SEASON_START <= w <= SEASON_END
    )
    weeks_labels = [week_label(w) for w in all_weeks]
    years = sorted(df["cal_year"].unique().tolist())
    out = {"weeks": weeks_labels, "years": years, "variables": {}}
    for v in DROUGHT_VARS:
        matrix = []
        for yr in years:
            row = []
            yr_df = df[df["cal_year"] == yr]
            for w in all_weeks:
                sub = yr_df[yr_df["iso_week"] == w]
                val = (float(sub[v].iloc[0])
                       if not sub.empty and v in sub.columns and pd.notna(sub[v].iloc[0])
                       else None)
                row.append(val)
            matrix.append(row)
        out["variables"][v] = matrix
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 6. GENERATE SELF-CONTAINED HTML
# ═══════════════════════════════════════════════════════════════════════════
def make_html(model_rows, seasonality, cotton, drought, ci_pct):
    # Serialise all data as JSON strings embedded in the HTML
    def jdump(obj):
        return json.dumps(obj, separators=(",", ":"),
                          default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else o)

    j_rows    = jdump(model_rows)
    j_season  = jdump(seasonality)
    ci_label  = f"{ci_pct}%"
    max_yr    = int(drought["cal_year"].max())
    cotton_range = f"{int(cotton.index.min())}–{int(cotton.index.max())}"
    drought_range = f"{drought['Week'].min().date()} – {drought['Week'].max().date()}"
    run_time  = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    # Actual abandonment stats for reference
    ab = cotton["abandonment"]
    ab_stats = (f"Actual abandonment: min={ab.min()*100:.0f}%  "
                f"mean={ab.mean()*100:.0f}%  max={ab.max()*100:.0f}%  "
                f"std={ab.std()*100:.0f}pp")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>TX Cotton Abandonment – Drought Predictor</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,Arial,sans-serif;background:#0d1117;color:#e2e8f0;min-height:100vh}}
code{{background:#1e2a3a;padding:2px 6px;border-radius:3px;font-size:.85em}}
.hdr{{background:linear-gradient(135deg,#0f2a1a 0%,#1a3050 60%,#0f2a1a 100%);
  padding:16px 28px;border-bottom:2px solid #2d6a4f;
  display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:8px}}
.hdr h1{{font-size:1.35rem;font-weight:700;color:#f0fff4}}
.hdr p{{font-size:.78rem;color:#90a4ae;margin-top:3px}}
.badge{{background:#1a3050;border:1px solid #2d6a4f;border-radius:20px;
  padding:4px 12px;font-size:.73rem;color:#68d391;white-space:nowrap;align-self:center}}
.meta{{background:#111820;padding:6px 28px;font-size:.73rem;color:#607080;
  border-bottom:1px solid #1e2a3a;display:flex;gap:16px;flex-wrap:wrap}}
.meta b{{color:#68d391}}
.tabs{{display:flex;background:#0d1117;border-bottom:2px solid #1e2a3a;padding:0 20px;overflow-x:auto}}
.tab{{padding:10px 16px;cursor:pointer;font-size:.83rem;color:#607080;
  border-bottom:3px solid transparent;margin-bottom:-2px;white-space:nowrap;transition:color .2s}}
.tab:hover{{color:#c0d0e0}}
.tab.active{{color:#68d391;border-bottom-color:#68d391;font-weight:600}}
.panel{{display:none;padding:18px 22px 40px}}
.panel.active{{display:block}}
.st{{font-size:.92rem;font-weight:600;color:#68d391;margin:16px 0 3px}}
.sn{{font-size:.74rem;color:#607080;line-height:1.5;margin-bottom:10px}}
.ctrls{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px;align-items:flex-end}}
.ctrl{{display:flex;flex-direction:column;gap:2px;font-size:.74rem;color:#90a4ae}}
.ctrl span{{font-size:.68rem;color:#607080}}
select{{background:#1a2535;border:1px solid #2d3e50;color:#e2e8f0;
  padding:4px 8px;border-radius:5px;font-size:.74rem;cursor:pointer}}
.tgl{{display:flex;border:1px solid #2d3e50;border-radius:5px;overflow:hidden}}
.tgl button{{padding:4px 11px;font-size:.74rem;background:#1a2535;color:#90a4ae;border:none;cursor:pointer}}
.tgl button.on{{background:#1a4030;color:#68d391;font-weight:600}}
.card{{background:#111820;border:1px solid #1e2a3a;border-radius:9px;padding:12px;margin-bottom:16px;overflow-x:auto}}
.hm-leg{{display:flex;align-items:center;gap:7px;margin-top:9px;font-size:.7rem;color:#607080}}
.hm-grad{{width:130px;height:8px;border-radius:3px}}
.sc-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:18px}}
@media(max-width:640px){{.sc-grid{{grid-template-columns:1fr}}}}
.sc-card{{background:#111820;border:1px solid #1e2a3a;border-radius:9px;padding:10px}}
.sc-card h3{{font-size:.78rem;font-weight:600;margin-bottom:5px}}
.abanner{{background:#0a1f14;border:1px solid #2d6a4f;border-radius:6px;
  padding:5px 12px;font-size:.76rem;color:#68d391;margin-bottom:8px;display:inline-block}}
.tw{{overflow-x:auto;border-radius:9px;border:1px solid #1e2a3a;margin-bottom:8px}}
table{{width:100%;border-collapse:collapse;font-size:.73rem;white-space:nowrap}}
thead th{{padding:7px 8px;text-align:center;position:sticky;z-index:5;border-bottom:1px solid #1e2a3a}}
thead tr:first-child th{{background:#0e1d2e;color:#7fb3d3;font-weight:600;top:0}}
thead tr:last-child th{{background:#0a1520;color:#607080;font-weight:400;font-size:.67rem;top:34px;border-bottom:2px solid #2d4060}}
th.wk,td.wk{{text-align:left;padding-left:10px}}
th.vh{{background:#0a1f14!important}}
th.vs{{background:#061410!important}}
th.bh{{background:#150d25!important}}
th.bs{{background:#0d0820!important}}
tbody tr:nth-child(even){{background:#0e1420}}
tbody tr:nth-child(odd){{background:#0d1117}}
tbody tr:hover{{background:#132030}}
td{{padding:5px 7px;text-align:center;vertical-align:middle;border-bottom:1px solid #141e28}}
td.wk{{font-weight:600;color:#c8d8e8;min-width:72px}}
td.pt{{color:#f6e05e;font-weight:600}}
td.ci{{color:#607080;font-size:.67rem}}
td.bv{{font-weight:600;font-size:.69rem}}
td.bp{{color:#c4a0f0;font-weight:700}}
td.bc{{color:#9070c0;font-size:.67rem}}
.r2w{{display:inline-flex;align-items:center;gap:3px}}
.r2bg{{width:30px;height:4px;background:#1e2a3a;border-radius:2px}}
.r2f{{height:4px;border-radius:2px}}
.dg-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
@media(max-width:640px){{.dg-grid{{grid-template-columns:1fr}}}}
.dg-card{{background:#111820;border:1px solid #1e2a3a;border-radius:9px;padding:10px}}
.dg-card h3{{font-size:.78rem;font-weight:600;margin-bottom:7px}}
.about{{max-width:660px;font-size:.82rem;line-height:1.75;color:#b0c0d0}}
.about h3{{color:#68d391;font-size:.86rem;margin:14px 0 4px}}
.about p{{margin-bottom:7px}}
.nbox{{background:#1a2535;border-left:3px solid #68d391;padding:8px 12px;
  border-radius:0 5px 5px 0;font-size:.76rem;color:#90c4ae;margin:8px 0;line-height:1.6}}
.about table{{border-collapse:collapse;width:100%;margin:7px 0;font-size:.77rem}}
.about table th{{text-align:left;padding:4px 9px;background:#111820;color:#90a4ae;border-bottom:1px solid #1e2a3a}}
.about table td{{padding:4px 9px;border-bottom:1px solid #141e28}}
</style>
</head>
<body>
<div class="hdr">
  <div>
    <h1>🌾 Texas Upland Cotton · Drought-Based Abandonment Predictor</h1>
    <p>USDA Drought Monitor (weekly) × NASS planted/harvested · Rolling OLS · {ci_label} prediction intervals · {ab_stats}</p>
  </div>
  <div class="badge">Run: {run_time}</div>
</div>
<div class="meta">
  <span>Cotton: <b>{cotton_range}</b></span>
  <span>Drought: <b>{drought_range}</b></span>
  <span>CI level: <b>{ci_label}</b></span>
  <span>Min train years: <b>{MIN_TRAIN_YRS}</b></span>
  <span>Predictions for MY: <b>{max_yr}/{str(max_yr+1)[-2:]}</b></span>
</div>
<div class="tabs">
  <div class="tab active" onclick="showTab('t1')">🌡 Drought Seasonality</div>
  <div class="tab"        onclick="showTab('t2')">📊 Predictions</div>
  <div class="tab"        onclick="showTab('t3')">🔬 Diagnostics</div>
  <div class="tab"        onclick="showTab('t4')">ℹ About</div>
</div>

<div id="t1" class="panel active">
  <div class="st" style="margin-top:0">Drought Monitor Seasonality — Last 20 Years</div>
  <div class="sn">% of Texas cotton area in each drought category, Apr–Oct by week. Hover for exact values.</div>
  <div class="ctrls">
    <div class="ctrl"><span>Variable</span>
      <select id="hv" onchange="drawHeatmap()">
        <option value="D1-D4">D1-D4 — Moderate or worse</option>
        <option value="D2-D4">D2-D4 — Severe or worse</option>
        <option value="D3-D4">D3-D4 — Extreme or worse</option>
        <option value="D4">D4 — Exceptional only</option>
      </select>
    </div>
    <div class="ctrl"><span>Color scale</span>
      <select id="hs" onchange="drawHeatmap()">
        <option value="abs">Absolute (0–100%)</option>
        <option value="rel">Relative (per-variable max)</option>
      </select>
    </div>
  </div>
  <div class="card" style="padding:10px 10px 7px">
    <div id="hm"></div>
    <div class="hm-leg"><span>Low</span><div class="hm-grad" id="hm-leg"></div><span>High</span>
      <span style="margin-left:10px;color:#405060">% area in drought</span></div>
  </div>
  <div class="st">Average Weekly Drought Level</div>
  <div class="sn">Mean % across all years for each week.</div>
  <div class="card"><div id="avgbar"></div></div>
</div>

<div id="t2" class="panel">
  <div class="st" style="margin-top:0">Scatter Plots — Latest Available Week</div>
  <div class="sn">Dots = actual historical abandonment ratios. Dashed = OLS fit.
    <b style="color:#fff">★</b> = current season prediction with {ci_label} CI whisker.</div>
  <div class="ctrls">
    <div class="ctrl"><span>Drought approach</span>
      <div class="tgl">
        <button class="on" id="bs" onclick="setApp('single')">Single-week</button>
        <button           id="bc" onclick="setApp('cumulative')">Cumulative avg</button>
      </div>
    </div>
    <div class="ctrl" style="margin-left:auto"><span>Table filter</span>
      <select id="pf" onchange="drawTable()">
        <option value="0">All weeks</option>
        <option value="0.05">p &lt; 0.05 only</option>
        <option value="0.10">p &lt; 0.10 only</option>
      </select>
    </div>
  </div>
  <div class="sc-grid" id="sc-grid"></div>
  <div class="st">Weekly Prediction Table</div>
  <div class="sn">Predicted abandonment ratio for MY {max_yr}/{str(max_yr+1)[-2:]} using drought through each week.
    R² bar: <span style="color:#68d391">green p&lt;0.05</span> · <span style="color:#f6e05e">yellow p&lt;0.10</span> · <span style="color:#607080">grey n.s.</span></div>
  <div class="abanner" id="abanner">📍 Single-week drought readings</div>
  <div class="tw"><div id="tbl"></div></div>
  <div style="font-size:.68rem;color:#405060;margin-top:7px;line-height:1.6">
    † {ci_label} prediction interval = range that covers {ci_label} of individual years' outcomes given that drought reading.<br>
    † Single-week = that week's drought % only. Cumulative avg = mean drought % from Apr W1 through that week.<br>
    † Apr W1 CI is wider (less data accumulated); narrows as season progresses and R² grows.
  </div>
</div>

<div id="t3" class="panel">
  <div class="st" style="margin-top:0">Model Diagnostics</div>
  <div class="sn">R² and significance for each drought variable across season weeks.</div>
  <div class="ctrls">
    <div class="ctrl"><span>Approach</span>
      <div class="tgl">
        <button class="on" id="dbs" onclick="setDApp('single')">Single-week</button>
        <button           id="dbc" onclick="setDApp('cumulative')">Cumulative</button>
      </div>
    </div>
    <div class="ctrl"><span>Metric</span>
      <div class="tgl">
        <button class="on" id="dbr" onclick="setDMet('r2')">R²</button>
        <button           id="dbp" onclick="setDMet('pvalue')">Significance</button>
      </div>
    </div>
  </div>
  <div class="dg-grid" id="dg-grid"></div>
</div>

<div id="t4" class="panel">
  <div class="about">
    <h3>What this tool does</h3>
    <p>Predicts Texas upland cotton <b>abandonment ratio</b> (1 − harvested ÷ planted) using weekly USDA Drought Monitor data as the leading indicator.</p>
    <h3>Understanding the Confidence Interval</h3>
    <div class="nbox">
      The {ci_label} CI is a <b>prediction interval for abandonment ratio</b> — the range expected to contain {ci_label} of individual years' outcomes given a particular drought reading.
      Actual abandonment ranges {ab.min()*100:.0f}%–{ab.max()*100:.0f}% with std dev ~{ab.std()*100:.0f}pp.
      With low R² (especially early season), the CI is wide but honest.
      It narrows from July onward as cumulative drought becomes more predictive.
    </div>
    <h3>Season Alignment</h3>
    <p>Drought weeks Apr–Oct of year <b>Y</b> → marketing year <b>Y</b> (e.g. Apr–Oct 2025 → MY 2025/26).</p>
    <h3>Model</h3>
    <table>
      <tr><th>Parameter</th><th>Value</th></tr>
      <tr><td>Model</td><td>OLS: abandonment ~ drought_variable</td></tr>
      <tr><td>Training</td><td>Rolling window, all prior years, min {MIN_TRAIN_YRS}</td></tr>
      <tr><td>CI level</td><td>{ci_label} prediction interval (t-distribution, n−2 df)</td></tr>
      <tr><td>Best predictor</td><td>Highest overall R² at that ISO-week</td></tr>
    </table>
    <h3>How to update</h3>
    <p>1. Replace <code>cotton_texas.csv</code> and/or <code>drought_texas.csv</code> with updated files.<br>
    2. Run <code>python process_data.py</code><br>
    3. Open <code>cotton_drought_dashboard.html</code> in any browser.</p>
  </div>
</div>

<script>
var ROWS    = {j_rows};
var SEASON  = {j_season};
var VARS    = ["D1-D4","D2-D4","D3-D4","D4"];
var VCOL    = {{"D1-D4":"#68d391","D2-D4":"#f6e05e","D3-D4":"#fc8181","D4":"#d6bcfa"}};
var VBG     = {{"D1-D4":"#0a1f14","D2-D4":"#1f1a08","D3-D4":"#1f0a0a","D4":"#130a1f"}};
var APP     = "single";
var DAPP    = "single";
var DMET    = "r2";
var CI_PCT  = "{ci_label}";

function showTab(id){{
  ["t1","t2","t3","t4"].forEach(function(t,i){{
    document.querySelectorAll(".tab")[i].classList.toggle("active",t===id);
    document.getElementById(t).classList.toggle("active",t===id);
  }});
  if(id==="t3") drawDiag();
}}

// ── SVG helpers ──────────────────────────────────────────────────────────
var NS="http://www.w3.org/2000/svg";
function mel(tag,a){{
  var e=document.createElementNS(NS,tag);
  Object.keys(a).forEach(function(k){{e.setAttribute(k,a[k]);}});
  return e;
}}
function msvg(w,h){{
  var s=document.createElementNS(NS,"svg");
  s.setAttribute("width","100%");
  s.setAttribute("viewBox","0 0 "+w+" "+h);
  s.setAttribute("style","font-family:inherit;display:block;overflow:visible");
  return s;
}}
function tt(el,txt){{var t=document.createElementNS(NS,"title");t.textContent=txt;el.appendChild(t);return el;}}
function mtxt(txt,a,fill){{var e=mel("text",a);e.textContent=txt;if(fill)e.setAttribute("fill",fill);return e;}}
function scl(d0,d1,r0,r1){{return function(v){{return d1===d0?r0:(r0+(v-d0)/(d1-d0)*(r1-r0));}}}}
function clamp(v,lo,hi){{return Math.max(lo,Math.min(hi,v));}}
function lerpc(c1,c2,t){{
  var r1=parseInt(c1.slice(1,3),16),g1=parseInt(c1.slice(3,5),16),b1=parseInt(c1.slice(5,7),16);
  var r2=parseInt(c2.slice(1,3),16),g2=parseInt(c2.slice(3,5),16),b2=parseInt(c2.slice(5,7),16);
  return "rgb("+Math.round(r1+(r2-r1)*t)+","+Math.round(g1+(g2-g1)*t)+","+Math.round(b1+(b2-b1)*t)+")";
}}

// ── HEATMAP ──────────────────────────────────────────────────────────────
function drawHeatmap(){{
  var c=document.getElementById("hm"); if(!c) return;
  c.innerHTML="";
  var vn=document.getElementById("hv").value;
  var sc=document.getElementById("hs").value;
  var mat=SEASON.variables[vn], weeks=SEASON.weeks, years=SEASON.years;
  if(!mat) return;
  var cw=c.clientWidth||800, ML=52,MT=70,cellH=18;
  var cellW=Math.max(11,Math.floor((cw-ML-6)/weeks.length));
  var svgW=ML+weeks.length*cellW, svgH=MT+years.length*cellH+6;
  var allV=[];
  for(var yi=0;yi<mat.length;yi++) for(var wi=0;wi<mat[yi].length;wi++) if(mat[yi][wi]!==null) allV.push(mat[yi][wi]);
  var maxV=sc==="rel"?(allV.length?Math.max.apply(null,allV):100):100;
  var svg=msvg(svgW,svgH), colTo=VCOL[vn]||"#68d391";
  for(var wi=0;wi<weeks.length;wi++){{
    var tx=mel("text",{{x:ML+wi*cellW+cellW/2,y:MT-5,
      transform:"rotate(-55,"+(ML+wi*cellW+cellW/2)+","+(MT-5)+")",
      "text-anchor":"end","font-size":"9","fill":"#607080"}});
    tx.textContent=weeks[wi]; svg.appendChild(tx);
  }}
  for(var yi=0;yi<years.length;yi++){{
    var tx=mel("text",{{x:ML-5,y:MT+yi*cellH+cellH/2+3,
      "text-anchor":"end","font-size":"9","fill":"#8090a0"}});
    tx.textContent=String(years[yi]); svg.appendChild(tx);
  }}
  for(var yi=0;yi<years.length;yi++){{
    for(var wi=0;wi<mat[yi].length;wi++){{
      var val=mat[yi][wi];
      var fill=val===null?"#1a2030":lerpc("#0d1520",colTo,clamp(val/maxV,0,1));
      tt(svg.appendChild(mel("rect",{{x:ML+wi*cellW,y:MT+yi*cellH,
        width:cellW-1,height:cellH-1,rx:2,fill:fill}})),
        years[yi]+" "+weeks[wi]+": "+(val===null?"N/A":val.toFixed(1)+"%"));
    }}
  }}
  c.appendChild(svg);
  document.getElementById("hm-leg").style.background="linear-gradient(90deg,#0d1520,"+colTo+")";
}}

// ── AVG BAR ──────────────────────────────────────────────────────────────
function drawAvgBar(){{
  var c=document.getElementById("avgbar"); if(!c) return;
  c.innerHTML="";
  var cw=c.clientWidth||800, ML=44,MR=10,MB=48,MT=8;
  var weeks=SEASON.weeks, nw=weeks.length;
  var avgs={{}};
  VARS.forEach(function(v){{
    var mat=SEASON.variables[v]; avgs[v]=[];
    for(var wi=0;wi<nw;wi++){{
      var sum=0,cnt=0;
      for(var yi=0;yi<mat.length;yi++) if(mat[yi][wi]!==null){{sum+=mat[yi][wi];cnt++;}}
      avgs[v].push(cnt?sum/cnt:0);
    }}
  }});
  var svgH=125, bW=Math.max(2,Math.floor((cw-ML-MR)/nw/VARS.length)-0.5), gW=bW*VARS.length+2;
  var allA=[]; VARS.forEach(function(v){{avgs[v].forEach(function(x){{allA.push(x);}});}});
  var yMax=allA.length?Math.max.apply(null,allA):100;
  var yS=scl(0,yMax,svgH-MB,MT), svg=msvg(cw,svgH);
  [0,25,50,75,100].forEach(function(t){{
    if(t>yMax+5) return;
    var yy=yS(t);
    svg.appendChild(mel("line",{{x1:ML,y1:yy,x2:cw-MR,y2:yy,stroke:"#1a2535","stroke-width":"1"}}));
    svg.appendChild(mtxt(t+"%",{{x:ML-3,y:yy+3,"text-anchor":"end","font-size":"8"}},"#607080"));
  }});
  for(var wi=0;wi<nw;wi++){{
    var gx=ML+wi*gW;
    for(var vi=0;vi<VARS.length;vi++){{
      var val=avgs[VARS[vi]][wi]||0, yy=yS(val), bh=Math.max(1,(svgH-MB)-yy);
      tt(svg.appendChild(mel("rect",{{x:gx+vi*bW,y:yy,width:Math.max(1,bW-0.5),height:bh,rx:1,fill:VCOL[VARS[vi]]}})),
         weeks[wi]+" "+VARS[vi]+": "+val.toFixed(1)+"%");
    }}
    if(wi%3===0){{
      var tx=mel("text",{{x:gx+gW/2,y:svgH-MB+10,"text-anchor":"middle",
        transform:"rotate(-40,"+(gx+gW/2)+","+(svgH-MB+10)+")","font-size":"8","fill":"#607080"}});
      tx.textContent=weeks[wi]; svg.appendChild(tx);
    }}
  }}
  VARS.forEach(function(v,i){{
    var lx=ML+i*88;
    svg.appendChild(mel("rect",{{x:lx,y:svgH-13,width:9,height:7,rx:2,fill:VCOL[v]}}));
    svg.appendChild(mtxt(v,{{x:lx+12,y:svgH-7,"font-size":"8"}},VCOL[v]));
  }});
  c.appendChild(svg);
}}

// ── APPROACH ─────────────────────────────────────────────────────────────
function setApp(a){{
  APP=a;
  document.getElementById("bs").classList.toggle("on",a==="single");
  document.getElementById("bc").classList.toggle("on",a==="cumulative");
  document.getElementById("abanner").innerHTML=
    a==="single"?"📍 <b>Single-week</b> — each row uses only that week's drought %":
                 "📍 <b>Cumulative avg</b> — each row uses mean drought % from Apr W1 through that week";
  drawScatter();
  drawTable();
}}

// ── SCATTER ───────────────────────────────────────────────────────────────
function drawScatter(){{
  var grid=document.getElementById("sc-grid"); if(!grid) return;
  grid.innerHTML="";
  VARS.forEach(function(v){{
    // Find latest row that has scatter data for this var + approach
    var wkData=null, wkLabel="";
    for(var i=ROWS.length-1;i>=0;i--){{
      var d=ROWS[i].vars[v]&&ROWS[i].vars[v][APP];
      if(d&&d.scatter_x&&d.scatter_x.length>0&&d.curr&&d.curr.point!==null){{
        wkData=d; wkLabel=ROWS[i].label; break;
      }}
    }}
    var card=document.createElement("div"); card.className="sc-card";
    var sid="sc_"+v.replace(/[^a-z0-9]/gi,"_");
    card.innerHTML='<h3 style="color:'+VCOL[v]+'">'+v+
      ' <span style="color:#607080;font-weight:400;font-size:.69rem">as of '+wkLabel+
      ' ('+APP.replace("single","single-wk").replace("cumulative","cumul avg")+')</span></h3>'+
      '<div id="'+sid+'"></div>';
    grid.appendChild(card);
    if(!wkData){{
      var el=document.getElementById(sid);
      if(el) el.innerHTML='<div style="color:#607080;font-size:.73rem;padding:16px">No current-season data.</div>';
      return;
    }}
    drawOneScatter(sid,v,wkData);
  }});
}}

function starPath(cx,cy,or_,ir,pts){{
  var d="";
  for(var i=0;i<pts*2;i++){{
    var r=i%2===0?or_:ir, a=(Math.PI/pts)*i-Math.PI/2;
    d+=(i===0?"M":"L")+(cx+r*Math.cos(a)).toFixed(2)+","+(cy+r*Math.sin(a)).toFixed(2);
  }}
  return d+"Z";
}}

function drawOneScatter(cid,vn,d){{
  var c=document.getElementById(cid); if(!c) return;
  c.innerHTML="";
  var xs=d.scatter_x||[], ys=d.scatter_y||[], yrs=d.scatter_years||[];
  var curr=d.curr||{{}};
  var cx=curr.x, cy=curr.point, clo=curr.lo, chi=curr.hi, cyr=curr.year;
  var rx=d.reg_x||[], ry=d.reg_y||[];
  var r2=d.r2, pv=d.pvalue;

  if(!xs.length){{ c.innerHTML='<div style="color:#607080;padding:10px;font-size:.73rem">No data</div>'; return; }}

  var cw=c.clientWidth||300, ML=42,MR=14,MT=18,MB=44, svgW=cw, svgH=220, pw=svgW-ML-MR, ph=svgH-MT-MB;

  var allX=xs.slice(); if(cx!==null&&cx!==undefined) allX.push(cx);
  var allY=ys.slice();
  if(cy!==null) allY.push(cy);
  if(clo!==null) allY.push(clo);
  if(chi!==null) allY.push(chi);

  var xMin=Math.max(0,Math.min.apply(null,allX)), xMax=Math.min(100,Math.max.apply(null,allX));
  var yMin=Math.max(0,Math.min.apply(null,allY)), yMax=Math.min(1, Math.max.apply(null,allY));
  var xPad=(xMax-xMin)*0.14||5, yPad=(yMax-yMin)*0.16||0.05;
  xMin=Math.max(0,xMin-xPad); xMax=Math.min(100,xMax+xPad);
  yMin=Math.max(0,yMin-yPad); yMax=Math.min(1, yMax+yPad);

  var xS=scl(xMin,xMax,0,pw), yS=scl(yMin,yMax,ph,0);
  var svg=msvg(svgW,svgH);
  var g=mel("g",{{transform:"translate("+ML+","+MT+")"}});

  // Grid
  var xTks=[0,10,20,30,40,50,60,70,80,90,100].filter(function(t){{return t>=xMin&&t<=xMax;}});
  var yTks=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0].filter(function(t){{return t>=yMin-0.01&&t<=yMax+0.01;}});
  xTks.forEach(function(t){{
    var xx=xS(t);
    g.appendChild(mel("line",{{x1:xx,y1:0,x2:xx,y2:ph,stroke:"#1a2535","stroke-width":"1"}}));
    g.appendChild(mtxt(t+"%",{{x:xx,y:ph+13,"text-anchor":"middle","font-size":"8"}},"#607080"));
  }});
  yTks.forEach(function(t){{
    var yy=yS(t);
    g.appendChild(mel("line",{{x1:0,y1:yy,x2:pw,y2:yy,stroke:"#1a2535","stroke-width":"1"}}));
    g.appendChild(mtxt((t*100).toFixed(0)+"%",{{x:-4,y:yy+3,"text-anchor":"end","font-size":"8"}},"#607080"));
  }});

  // Axis labels
  g.appendChild(mtxt(vn+" (% area in drought)",{{x:pw/2,y:ph+32,"text-anchor":"middle","font-size":"8.5"}},"#90a4ae"));
  var ayl=mel("text",{{x:0,y:0,"text-anchor":"middle","font-size":"8.5",fill:"#90a4ae",
    transform:"rotate(-90) translate("+(-(ph/2))+","+-32+")"}});
  ayl.textContent="Abandonment ratio"; g.appendChild(ayl);

  // Regression line
  if(rx.length===2){{
    g.appendChild(mel("line",{{x1:xS(rx[0]),y1:yS(ry[0]),x2:xS(rx[1]),y2:yS(ry[1]),
      stroke:VCOL[vn],"stroke-width":"1.5","stroke-dasharray":"5,3",opacity:"0.6"}}));
  }}

  // CI whisker
  if(cx!==null&&clo!==null&&chi!==null){{
    var xcx=xS(cx), yclo=yS(clo), ychi=yS(chi);
    g.appendChild(mel("line",{{x1:xcx,y1:ychi,x2:xcx,y2:yclo,stroke:"#fff","stroke-width":"2",opacity:"0.55"}}));
    g.appendChild(mel("line",{{x1:xcx-6,y1:yclo,x2:xcx+6,y2:yclo,stroke:"#fff","stroke-width":"2",opacity:"0.55"}}));
    g.appendChild(mel("line",{{x1:xcx-6,y1:ychi,x2:xcx+6,y2:ychi,stroke:"#fff","stroke-width":"2",opacity:"0.55"}}));
  }}

  // Historical dots + year labels on extremes
  var maxI=0,minI=0;
  for(var i=0;i<xs.length;i++){{
    tt(g.appendChild(mel("circle",{{cx:xS(xs[i]),cy:yS(ys[i]),r:"4.5",
      fill:VCOL[vn],opacity:"0.72",stroke:"#0d1117","stroke-width":"0.5"}})),
      yrs[i]+": drought="+xs[i].toFixed(1)+"%, abandon="+(ys[i]*100).toFixed(1)+"%");
    if(ys[i]>ys[maxI]) maxI=i;
    if(ys[i]<ys[minI]) minI=i;
  }}
  if(xs.length>0){{
    [maxI,minI].forEach(function(i){{
      g.appendChild(mtxt(String(yrs[i]),{{x:xS(xs[i])+6,y:yS(ys[i])-5,"font-size":"8"}},"#a0b0c0"));
    }});
  }}

  // Star for current prediction
  if(cx!==null&&cy!==null){{
    var sx=xS(cx), sy=yS(cy);
    g.appendChild(mel("path",{{d:starPath(sx,sy,11,4.5,5),fill:"#fff",stroke:"#f0a020","stroke-width":"1.5"}}));
    var onR=sx<=pw*0.60, anch=onR?"start":"end", lx=sx+(onR?14:-14);
    var onT=sy>ph*0.35, ly=sy+(onT?-16:18);
    g.appendChild(mtxt((cyr||"Curr")+": "+(cy*100).toFixed(1)+"%",
      {{x:lx,y:ly,"text-anchor":anch,"font-size":"9.5","font-weight":"bold"}},"#ffffff"));
    if(clo!==null&&chi!==null){{
      g.appendChild(mtxt(CI_PCT+" CI: ["+(clo*100).toFixed(1)+"–"+(chi*100).toFixed(1)+"%]",
        {{x:lx,y:ly+13,"text-anchor":anch,"font-size":"7.5"}},"#c0c0c0"));
    }}
  }}

  // R² badge
  var pTxt=pv!==null?(pv<0.05?"★ p<0.05":pv<0.1?"▲ p<0.10":"p="+(pv).toFixed(2)):"";
  var pCol=pv!==null?(pv<0.05?"#68d391":pv<0.1?"#f6e05e":"#607080"):"#607080";
  g.appendChild(mtxt("R²="+(r2!==null?(r2*100).toFixed(1)+"%":"—")+"  "+pTxt,
    {{x:pw-2,y:12,"text-anchor":"end","font-size":"8.5","font-weight":"600"}},pCol));

  svg.appendChild(g); c.appendChild(svg);
}}

// ── TABLE ─────────────────────────────────────────────────────────────────
function drawTable(){{
  var c=document.getElementById("tbl"); if(!c||!ROWS.length) return;
  var app=APP, thresh=parseFloat(document.getElementById("pf").value)||0;
  var rows=ROWS;
  if(thresh>0) rows=ROWS.filter(function(row){{
    return VARS.some(function(v){{
      var d=row.vars[v]&&row.vars[v][app];
      return d&&d.pvalue!==null&&d.pvalue<=thresh;
    }});
  }});

  var vH=VARS.map(function(v){{
    return '<th class="vh" colspan="3" style="color:'+VCOL[v]+';background:'+VBG[v]+'">'+v+'</th>';
  }}).join("");
  var sH=VARS.map(function(){{return '<th class="vs">Pred%</th><th class="vs">'+CI_PCT+' CI</th><th class="vs">R²</th>';}}).join("");

  var tb="";
  rows.forEach(function(row){{
    var cells=VARS.map(function(v){{
      var d=row.vars[v]&&row.vars[v][app];
      if(!d||!d.curr||d.curr.point===null||d.curr.point===undefined)
        return '<td class="pt">—</td><td class="ci">—</td><td>—</td>';
      var pt=(d.curr.point*100).toFixed(1)+"%";
      var lo=d.curr.lo!==null?(d.curr.lo*100).toFixed(1)+"%":"?";
      var hi=d.curr.hi!==null?(d.curr.hi*100).toFixed(1)+"%":"?";
      var r2v=d.r2!=null?d.r2:0, pv=d.pvalue;
      var col=pv===null?"#607080":pv<0.05?"#68d391":pv<0.1?"#f6e05e":"#607080";
      var pct=Math.round((r2v||0)*100);
      var r2h='<div class="r2w"><div class="r2bg"><div class="r2f" style="width:'+pct+'%;background:'+col+'"></div></div>'+
              '<span style="font-size:.67rem;color:'+col+'">'+pct+'%</span></div>';
      return '<td class="pt">'+pt+'</td><td class="ci">['+lo+'–'+hi+']</td><td>'+r2h+'</td>';
    }}).join("");
    var bv=row.best_var||"—", bcol=VCOL[bv]||"#c4a0f0";
    var bd=row.vars[bv]&&row.vars[bv][app];
    var bpt=bd&&bd.curr&&bd.curr.point!=null?(bd.curr.point*100).toFixed(1)+"%":"—";
    var blo=bd&&bd.curr&&bd.curr.lo!=null?(bd.curr.lo*100).toFixed(1)+"%":"?";
    var bhi=bd&&bd.curr&&bd.curr.hi!=null?(bd.curr.hi*100).toFixed(1)+"%":"?";
    tb+='<tr><td class="wk">'+row.label+'</td>'+cells+
      '<td class="bv" style="color:'+bcol+'">'+bv+'</td>'+
      '<td class="bp">'+bpt+'</td><td class="bc">['+blo+'–'+bhi+']</td></tr>';
  }});
  if(!tb) tb='<tr><td colspan="16" style="text-align:center;padding:18px;color:#607080">No weeks pass the filter.</td></tr>';

  c.innerHTML='<table><thead>'+
    '<tr><th class="wk" rowspan="2">Week</th>'+vH+
    '<th class="bh" colspan="3" style="color:#c4a0f0">★ Best Variable</th></tr>'+
    '<tr>'+sH+'<th class="bs">Var</th><th class="bs">Pred%</th><th class="bs">'+CI_PCT+' CI</th></tr>'+
    '</thead><tbody>'+tb+'</tbody></table>';
}}

// ── DIAGNOSTICS ────────────────────────────────────────────────────────────
function setDApp(a){{
  DAPP=a;
  document.getElementById("dbs").classList.toggle("on",a==="single");
  document.getElementById("dbc").classList.toggle("on",a==="cumulative");
  drawDiag();
}}
function setDMet(m){{
  DMET=m;
  document.getElementById("dbr").classList.toggle("on",m==="r2");
  document.getElementById("dbp").classList.toggle("on",m==="pvalue");
  drawDiag();
}}
function drawDiag(){{
  var grid=document.getElementById("dg-grid"); if(!grid) return;
  grid.innerHTML="";
  VARS.forEach(function(v){{
    var card=document.createElement("div"); card.className="dg-card";
    var did="dg_"+v.replace(/[^a-z0-9]/gi,"_");
    card.innerHTML='<h3 style="color:'+VCOL[v]+'">'+v+' — '+(DMET==="r2"?"R² by week":"Significance by week")+'</h3><div id="'+did+'"></div>';
    grid.appendChild(card);
    var labels=[],vals=[];
    ROWS.forEach(function(row){{
      var d=row.vars[v]&&row.vars[v][DAPP];
      labels.push(row.label);
      vals.push(DMET==="r2"?(d?d.r2:null):(d?d.pvalue:null));
    }});
    drawDiagBars(did,labels,vals,VCOL[v]);
  }});
}}
function drawDiagBars(cid,labels,data,color){{
  var c=document.getElementById(cid); if(!c) return;
  c.innerHTML="";
  var cw=c.clientWidth||330, ML=28,MR=8,MB=42, svgH=125;
  var bW=Math.max(2,Math.floor((cw-ML-MR)/labels.length)-1);
  var isR2=DMET==="r2", yS=scl(0,1,svgH-MB,8), svg=msvg(cw,svgH);
  [0,0.25,0.5,0.75,1].forEach(function(t){{
    var yy=yS(t);
    svg.appendChild(mel("line",{{x1:ML,y1:yy,x2:cw-MR,y2:yy,stroke:"#1a2535","stroke-width":"1"}}));
    var lbl=isR2?(t*100).toFixed(0)+"%":(t===0?"p=1":t===1?"p=0":"");
    svg.appendChild(mtxt(lbl,{{x:ML-2,y:yy+3,"text-anchor":"end","font-size":"7"}},"#607080"));
  }});
  if(!isR2){{
    [[0.95,"#68d391","p=0.05"],[0.9,"#f6e05e","p=0.10"]].forEach(function(item){{
      var yy=yS(item[0]);
      svg.appendChild(mel("line",{{x1:ML,y1:yy,x2:cw-MR,y2:yy,stroke:item[1],"stroke-width":"1","stroke-dasharray":"3,3"}}));
      svg.appendChild(mtxt(item[2],{{x:cw-MR-1,y:yy-2,"text-anchor":"end","font-size":"7"}},item[1]));
    }});
  }}
  for(var i=0;i<labels.length;i++){{
    var v=data[i], yv=isR2?(v||0):(v===null?0:Math.max(0,1-v));
    var yy=yS(yv), bh=Math.max(1,(svgH-MB)-yy);
    var col=isR2?lerpc("#1a3050",color,yv):(yv>0.95?"#68d391":yv>0.9?"#f6e05e":"#607080");
    tt(svg.appendChild(mel("rect",{{x:ML+i*(bW+1),y:yy,width:bW,height:bh,rx:1,fill:col}})),
       labels[i]+": "+(isR2?(v!==null?(v*100).toFixed(1)+"%":"N/A"):(v!==null?"p="+v.toFixed(3):"N/A")));
    if(i%4===0){{
      var tx=mel("text",{{x:ML+i*(bW+1)+bW/2,y:svgH-MB+9,"text-anchor":"middle",
        transform:"rotate(-45,"+(ML+i*(bW+1)+bW/2)+","+(svgH-MB+9)+")","font-size":"7","fill":"#607080"}});
      tx.textContent=labels[i]; svg.appendChild(tx);
    }}
  }}
  c.appendChild(svg);
}}

// ── Init ──────────────────────────────────────────────────────────────────
window.onload=function(){{
  drawHeatmap(); drawAvgBar(); drawScatter(); drawTable();
}};
window.onresize=function(){{
  drawHeatmap(); drawAvgBar(); drawScatter();
  var dg=document.getElementById("dg-grid");
  if(dg&&dg.children.length>1) drawDiag();
}};
</script>
</body>
</html>"""
    return html


# ═══════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("Texas Cotton Abandonment — Drought Predictor")
    print("=" * 44)

    # Check input files
    for p in [COTTON_CSV, DROUGHT_CSV]:
        if not p.exists():
            print(f"\nERROR: {p} not found.")
            print(f"Place your CSV files in the data/ subfolder.")
            sys.exit(1)

    print("\nLoading data…")
    cotton  = load_cotton(COTTON_CSV)
    drought = load_drought(DROUGHT_CSV)

    print("\nBuilding models (rolling OLS, all weeks × all variables × 2 approaches)…")
    model_rows = build_models(cotton, drought)
    print(f"  {len(model_rows)} season weeks processed")

    print("\nBuilding seasonality heatmap data…")
    seasonality = build_seasonality(drought)

    ci_pct = f"{int(CI_LEVEL * 100)}"
    print(f"\nGenerating self-contained HTML ({ci_pct}% CI)…")
    html = make_html(model_rows, seasonality, cotton, drought, ci_pct)

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    size_kb = OUTPUT_HTML.stat().st_size / 1024
    print(f"\n✓ Done! Output: {OUTPUT_HTML}  ({size_kb:.0f} KB)")
    print(f"\nOpen {OUTPUT_HTML} in any browser — no web server needed.")


if __name__ == "__main__":
    main()
