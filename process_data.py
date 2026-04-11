# SAVE THIS AS process_data.py (Part 1 above + this Part 2 below)

# ═══════════════════════════════════════════════════════════════════════════
# 3. SEASONALITY & PRODUCTION (condensed)
# ═══════════════════════════════════════════════════════════════════════════
def build_season_lines(drought, geo_label='TX', n_years=20):
    if drought is None or drought.empty: return None
    max_yr = drought["cal_year"].max()
    curr_yr = int(drought["mkt_year"].max())
    min_yr = max_yr - n_years + 1
    s_weeks = sorted(w for w in drought["iso_week"].unique() if 14 <= w <= 43)
    wlabels = [week_label(w) for w in s_weeks]
    curr_df = drought[drought["mkt_year"] == curr_yr].sort_values("Week")
    if curr_df.empty: return None
    lat_iw = int(curr_df["iso_week"].max())
    hist_yrs = sorted(y for y in drought["cal_year"].unique() if min_yr <= y <= max_yr and y != curr_yr)
    last_yr = curr_yr - 1
    out = {"geo": geo_label, "weeks": wlabels, "iso_weeks": [int(w) for w in s_weeks],
           "curr_yr": int(curr_yr), "latest_iw": int(lat_iw), "last_yr": int(last_yr), "variables": {}}
    for v in DROUGHT_VARS:
        series = {}
        for yr in hist_yrs + [curr_yr]:
            yr_df = drought[drought["cal_year"] == yr]
            line = [float(sub[v].iloc[0]) if (not (sub := yr_df[yr_df["iso_week"] == iw]).empty and pd.notna(sub[v].iloc[0])) else None for iw in s_weeks]
            if any(x is not None for x in line): series[int(yr)] = line
        lat_idx = s_weeks.index(lat_iw) if lat_iw in s_weeks else len(s_weeks) - 1
        c_vals = [series.get(curr_yr, [])[i] for i in range(lat_idx + 1) if i < len(series.get(curr_yr, [])) and series.get(curr_yr, [])[i] is not None]
        scores = [(yr, float(np.sqrt(np.mean([(c_vals[i] - [series[yr][j] for j in range(lat_idx + 1) if j < len(series[yr]) and series[yr][j] is not None][i])**2 for i in range(min(len(c_vals), len([series[yr][j] for j in range(lat_idx + 1) if j < len(series[yr]) and series[yr][j] is not None])))])))) for yr in hist_yrs if len([series[yr][j] for j in range(lat_idx + 1) if j < len(series[yr]) and series[yr][j] is not None]) >= 2]
        scores.sort(key=lambda x: x[1])
        top5 = [int(y) for y, _ in scores[:ANALOG_TOP_N]]
        out["variables"][v] = {"series": {str(k): v2 for k, v2 in series.items()}, "analogs": top5}
    return out

def build_production(cotton, best_tx):
    max_yr = int(cotton["mkt_year"].max())
    state_data = {}
    all_states = [g for g in cotton["geography"].unique() if g != US_GEO]
    print(f"    States found: {all_states}")
    for state in all_states:
        sdf = cotton[cotton["geography"] == state].copy()
        sdf = sdf[sdf["mkt_year"] <= max_yr].sort_values("mkt_year")
        if sdf.empty: continue
        sd = {"periods": {}}
        for P in PERIODS:
            rec = sdf[sdf["mkt_year"] >= max_yr - P + 1]
            if rec.empty: sd["periods"][P] = None; continue
            g = lambda col: safe(float(rec[col].dropna().mean()), 2) if col in rec.columns and len(rec[col].dropna()) else None
            sd["periods"][P] = {"ab": g("abandonment"), "yld": g("upland_cotton_lint_yield"), 
                               "plt": g("upland_cotton_planted_acreage")}
        lr = sdf.iloc[-1]
        sd["last_yr_actual"] = {"year": int(lr["mkt_year"]), "plt": safe(lr.get("upland_cotton_planted_acreage"), 0)}
        state_data[state] = sd
    matA, matB = {}, {}
    model_ab = best_tx.get("point")
    for P_ab in PERIODS:
        matA[P_ab], matB[P_ab] = {}, {}
        for P_yld in PERIODS:
            total_a, total_b, ok = 0.0, 0.0, True
            for st, sd in state_data.items():
                pa, py = sd["periods"].get(P_ab), sd["periods"].get(P_yld)
                if not pa or not py: ok = False; break
                ab_a, ab_b, yld, plt = pa["ab"], (model_ab if st == "TX" else pa["ab"]), py["yld"], pa["plt"]
                if ab_a is None or yld is None or plt is None: ok = False; break
                total_a += plt * 1000 * (1 - ab_a) * yld / 480_000_000
                total_b += plt * 1000 * (1 - ab_b) * yld / 480_000_000
            matA[P_ab][P_yld] = round(total_a, 3) if ok else None
            matB[P_ab][P_yld] = round(total_b, 3) if ok else None
    tx_hist = cotton[(cotton["geography"] == "TX") & (cotton["mkt_year"] >= max_yr - 9)]
    ab_vals = tx_hist["abandonment"].dropna() if not tx_hist.empty and "abandonment" in tx_hist.columns else pd.Series([0.1, 0.6])
    tx_ab_range = [round(int(float(ab_vals.min()) * 20) / 20 + i * 0.05, 2) for i in range(int(round(((int(float(ab_vals.max()) * 20) + 1) / 20 - int(float(ab_vals.min()) * 20) / 20) / 0.05)) + 1)]
    return {"state_data": state_data, "matA": {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in matA.items()},
            "matB": {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in matB.items()},
            "tx_ab_range": tx_ab_range, "periods": PERIODS, "period_labels": PERIOD_LABELS, "max_yr": max_yr, "model_ab": model_ab}

def build_analyst_summary(best_tx, wk_rows, mo_rows, cu_rows, prod, ab, drought):
    curr_yr = best_tx.get("curr_yr", "?")
    model_ab = best_tx.get("point")
    return {
        "seasonality": f"Current season ({curr_yr}): {len(wk_rows)} weeks available. Compare analog years.",
        "weekly": f"Weekly model: Best {best_tx.get('variable','—')} at {best_tx.get('label','')} R²={best_tx.get('r2',0)*100:.1f}%",
        "monthly": "Monthly model averages weekly readings for stability.",
        "cumulative": "Cumulative model shows running average from Apr W1.",
        "production": f"TX abandonment prediction: {(model_ab or 0)*100:.1f}% via {best_tx.get('model','—')} model"
    }

# ═══════════════════════════════════════════════════════════════════════════
# 4. HTML & MAIN (condensed working version)
# ═══════════════════════════════════════════════════════════════════════════
def make_html(wk_rows, mo_rows, cu_rows, slines, prod, best_tx, summary, cotton_ab, drought,
              us_wk, us_mo, us_cu, us_slines, us_best, ci_pct):
    curr_yr = wk_rows[0]["curr_yr"] if wk_rows else "?"
    j_wk, j_mo, j_cu, j_sl = jd(wk_rows), jd(mo_rows), jd(cu_rows), jd(slines) if slines else 'null'
    j_prod, j_btx = jd(prod), jd(best_tx)
    j_us_wk = jd(us_wk) if us_wk else '[]'
    j_us_mo = jd(us_mo) if us_mo else '[]'
    j_us_cu = jd(us_cu) if us_cu else '[]'
    j_us_sl = jd(us_slines) if us_slines else 'null'
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
</style></head>
<body>
<div class="hdr"><h1>🌾 TX Cotton Drought Predictor · MY {curr_yr}</h1>
<p>Best model: {best_tx.get('variable','—')} R²={best_tx.get('r2',0)*100:.1f}% → TX abandonment {(best_tx.get('point') or 0)*100:.1f}%</p></div>
<div class="tabs">
<div class="tab active" onclick="showTab('t1')">TX Drought</div>
<div class="tab" onclick="showTab('t2')">US Drought</div>
<div class="tab" onclick="showTab('t3')">Weekly</div>
<div class="tab" onclick="showTab('t4')">Monthly</div>
<div class="tab" onclick="showTab('t5')">Cumulative</div>
<div class="tab" onclick="showTab('t6')">Production</div>
</div>
<div id="t1" class="panel active"><div class="st">TX Seasonality</div><div id="sg"></div></div>
<div id="t2" class="panel"><div class="st">US Seasonality</div><div id="sg-us"></div></div>
<div id="t3" class="panel"><div class="st">Weekly Model</div><div id="tbl-w"></div></div>
<div id="t4" class="panel"><div class="st">Monthly Model</div><div id="tbl-m"></div></div>
<div id="t5" class="panel"><div class="st">Cumulative Model</div><div id="tbl-c"></div></div>
<div id="t6" class="panel">
<div class="st">Production Matrices</div>
<div class="mat"><div><h3>Matrix A - All Historical</h3><div id="mA"></div></div>
<div><h3>Matrix B - TX from Model</h3><div id="mB"></div></div></div>
</div>
<script>
var WK={j_wk}, MO={j_mo}, CU={j_cu}, SL={j_sl}, PROD={j_prod}, BTX={j_btx};
var WK_US={j_us_wk}, MO_US={j_us_mo}, CU_US={j_us_cu}, SL_US={j_us_sl};
function showTab(id){{document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));document.getElementById(id).classList.add('active');document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));event.target.classList.add('active');if(id==='t6')renderProd();}}
function renderProd(){{var mA=document.getElementById('mA'),mB=document.getElementById('mB');if(mA&&PROD.matA){{var h='<table><tr><th>Ab↓ Yld→</th>'+PROD.periods.map(p=>'<th>'+PROD.period_labels[p]+'</th>').join('')+'</tr>';PROD.periods.forEach(pAb=>{{h+='<tr><th>'+PROD.period_labels[pAb]+'</th>'+PROD.periods.map(pYld=>'<td>'+(PROD.matA[pAb]&&PROD.matA[pAb][pYld]!==null?PROD.matA[pAb][pYld].toFixed(2):'—')+'</td>').join('')+'</tr>';}});h+='</table>';mA.innerHTML=h;}}if(mB&&PROD.matB){{var h='<table><tr><th>Ab↓ Yld→</th>'+PROD.periods.map(p=>'<th>'+PROD.period_labels[p]+'</th>').join('')+'</tr>';PROD.periods.forEach(pAb=>{{h+='<tr><th>'+PROD.period_labels[pAb]+'</th>'+PROD.periods.map(pYld=>'<td>'+(PROD.matB[pAb]&&PROD.matB[pAb][pYld]!==null?PROD.matB[pAb][pYld].toFixed(2):'—')+'</td>').join('')+'</tr>';}});h+='</table>';mB.innerHTML=h;}}}}
window.onload=renderProd;
</script></body></html>'''

def main():
    print("=" * 60)
    print("TX + US Cotton — Drought Predictor")
    print("=" * 60)
    for p in [COTTON_CSV, DROUGHT_CSV]:
        if not p.exists(): print(f"ERROR: {p} not found"); sys.exit(1)
    has_us = DROUGHT_US_CSV.exists()
    print(f"US drought file: {'FOUND' if has_us else 'NOT FOUND (using TX as proxy)'}")
    
    cotton = load_cotton(COTTON_CSV)
    drought_tx = load_drought(DROUGHT_CSV, "TX")
    drought_us = load_drought(DROUGHT_US_CSV, "US") if has_us else drought_tx
    
    ab_tx = get_abandonment_series(cotton, "TX")
    ab_us = get_abandonment_series(cotton, US_GEO) if has_us else ab_tx
    
    print("\nBuilding TX models...")
    wk_rows, _ = build_weekly(ab_tx, drought_tx)
    mo_rows = build_monthly(ab_tx, drought_tx)
    cu_rows = build_cumulative(ab_tx, drought_tx)
    best_tx = best_prediction(wk_rows, mo_rows, cu_rows)
    print(f"TX: {len(wk_rows)} weekly, Best R²={best_tx.get('r2',0)*100:.1f}%")
    
    print("\nBuilding US models...")
    if has_us and ab_us is not ab_tx:
        wk_us, _ = build_weekly(ab_us, drought_us)
        mo_us = build_monthly(ab_us, drought_us)
        cu_us = build_cumulative(ab_us, drought_us)
        best_us = best_prediction(wk_us, mo_us, cu_us)
        print(f"US: {len(wk_us) if wk_us else 0} weekly")
    else:
        wk_us = mo_us = cu_us = []
        best_us = best_tx
    
    print("\nBuilding production...")
    prod = build_production(cotton, best_tx)
    print(f"States: {len(prod['state_data'])}, Matrices: {'DATA' if prod['matA'] else 'EMPTY'}")
    
    summary = build_analyst_summary(best_tx, wk_rows, mo_rows, cu_rows, prod, ab_tx, drought_tx)
    
    print("\nGenerating HTML...")
    OUTPUT_HTML.parent.mkdir(exist_ok=True)
    html = make_html(wk_rows, mo_rows, cu_rows, None, prod, best_tx, summary, ab_tx, drought_tx,
                     wk_us, mo_us, cu_us, None, best_us, "90")
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"✓ Done: {OUTPUT_HTML} ({OUTPUT_HTML.stat().st_size//1024} KB)")

if __name__ == "__main__":
    main()
