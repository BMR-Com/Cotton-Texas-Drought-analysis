#!/usr/bin/env python3
"""
Refresh data/drought_texas.csv and data/drought_US.csv from USDA/NDMC "Agriculture in Drought"
(https://agindrought.unl.edu/Table.aspx): weekly share of COTTON area in each US Drought Monitor
category, for Texas and the United States. Same layout as before:
    Week, None, D0-D4, D1-D4, D2-D4, D3-D4, D4   (newest week first)
"""
import sys
from pathlib import Path
import requests
import pandas as pd

B = "https://agindrought.unl.edu/Home.aspx/"
H = {"User-Agent": "Mozilla/5.0 (BMR-Com research)", "Content-Type": "application/json; charset=utf-8",
     "Referer": "https://agindrought.unl.edu/Table.aspx"}
TARGETS = {("1", "United States"): Path("data/drought_US.csv"), ("2", "Texas"): Path("data/drought_texas.csv")}


def call(fn, **params):
    r = requests.get(B + fn, params={k: f'"{v}"' for k, v in params.items()}, headers=H, timeout=90)
    r.raise_for_status()
    return r.json()["d"]


def main():
    ok = 0
    for (ltype, name), out in TARGETS.items():
        locs = [l for l in call("ReturnLocations2020", ltype=ltype, ctype="cotton") if l["Text"].strip() == name]
        if not locs:
            print("location not offered:", name); continue
        data = call("ReturnCropData2020", ctype="cotton", ltype=ltype, loc=locs[0]["Value"])
        df = pd.DataFrame(data)
        df = pd.DataFrame({"Week": pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d"), "None": df["None"], "D0-D4": df["D0"],
                           "D1-D4": df["D1"], "D2-D4": df["D2"], "D3-D4": df["D3"], "D4": df["D4"]})
        if len(df) < 1000:
            print("too few rows for", name, len(df)); continue
        old_last = None
        if out.exists():          # the site skips the odd week; keep weeks we already have, new values win
            old = pd.read_csv(out, encoding="utf-8-sig")
            old.columns = df.columns
            old_last = old.Week.max()
            df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates("Week", keep="last").sort_values("Week", ascending=False)
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"{name}: {len(df)} weeks, latest {df.Week.iloc[0]} (was {old_last})")
        ok += 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
