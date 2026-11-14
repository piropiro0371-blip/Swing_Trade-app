"""
開示イベントの調査: 決算・業績予想の修正が出た後、株価が市場平均に対してどう動いたか。

考え方（未来の情報を使わないための決まり）:
- 「反応日」= 開示を市場が最初に織り込む日。15:30より前の開示は当日、以降の開示は翌営業日
- 買うのは反応日の大引け後に条件を確認して、その翌営業日の寄り付き
- そこから n 営業日後の終値までの損益を、同じ期間のTOPIX連動ETFと比べる（コスト往復0.2%差引後）

Freeプランは2年分しか無いため、学習期間と検証期間に分けた正式な合否判定はできない。
ここでの結果は「10年分のデータで本格的に検証する価値があるか」を見る目安。

使い方:  python -m src.events      （先に python -m src.jquants で fins.parquet を作る）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data
from .engine import BENCH, Panel
from .jquants import FINS_PATH
from .research import load_config

HORIZONS = (5, 10, 20, 40)
COST = 0.002


def load_events(p: Panel) -> pd.DataFrame:
    f = pd.read_parquet(FINS_PATH)
    f = f[f["Code"].str.endswith("0")].copy()          # 普通株のみ（5桁目が0）
    f["code"] = f["Code"].str[:4]
    f = f[f["code"].isin(p.codes)]
    for c in ("Sales", "OP", "NP", "EPS", "FSales", "FOP", "FNP", "FEPS", "NxFOP", "NxFNp", "NxFEPS"):
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f["ts"] = pd.to_datetime(f["DiscDate"] + " " + f["DiscTime"].replace({"": "15:30:00", "nan": "15:30:00"}),
                             errors="coerce")
    f = f.dropna(subset=["ts"]).sort_values(["code", "ts"]).reset_index(drop=True)
    f["kind"] = np.where(f["DocType"].str.contains("EarnForecastRevision"), "修正",
                         np.where(f["DocType"].str.contains("FinancialStatements"), "決算", "他"))
    f = f[f["kind"] != "他"]

    # 反応日（日付×銘柄の表での行番号）
    dates = p.dates
    day = f["ts"].dt.normalize()
    after = f["ts"].dt.time >= pd.Timestamp("15:30").time()
    pos = dates.searchsorted(day)                       # 開示日以降で最初の営業日
    same = (pos < len(dates)) & (dates[np.minimum(pos, len(dates) - 1)] == day.to_numpy())
    f["r"] = np.where(same & ~after, pos, np.where(same, pos + 1, pos))
    f = f[f["r"] < len(dates) - 1].copy()

    # 今期の会社予想（営業利益）。期末決算では来期予想が「今期の予想」にあたる
    f["guide_fy"] = np.where(f["CurPerType"] == "FY", "next", f["CurFYEn"])
    f["guide_op"] = np.where(f["CurPerType"] == "FY", f["NxFOP"], f["FOP"])
    f["guide_eps"] = np.where(f["CurPerType"] == "FY", f["NxFEPS"], f["FEPS"])
    # 同じ会計年度に対する、前回の会社予想からの変化（上方修正・下方修正）
    # 「前回の予想」= 同じ会計年度について、それより前の時刻に出ていた直近の予想。
    # 本決算で出る来期予想（NxFOP）は、翌会計年度の最初の予想として扱う。
    cur = f.loc[f["FOP"].notna(), ["code", "CurFYEn", "ts", "FOP"]]
    nxt = f.loc[(f["CurPerType"] == "FY") & f["NxFOP"].notna(), ["code", "CurFYEn", "ts", "NxFOP"]].copy()
    nxt["CurFYEn"] = (pd.to_datetime(nxt["CurFYEn"], errors="coerce") + pd.DateOffset(years=1)).dt.strftime("%Y-%m-%d")
    hist = pd.concat([cur, nxt.rename(columns={"NxFOP": "FOP"})]).dropna(subset=["CurFYEn"])
    hist = hist.groupby(["code", "CurFYEn", "ts"], as_index=False)["FOP"].last().sort_values(["code", "CurFYEn", "ts"])
    hist["prev_fop"] = hist.groupby(["code", "CurFYEn"])["FOP"].shift(1)
    f = f.merge(hist[["code", "CurFYEn", "ts", "prev_fop"]], on=["code", "CurFYEn", "ts"], how="left")
    prev = f["prev_fop"]
    f["rev_op"] = np.where((f["CurPerType"] != "FY") & (prev > 0) & (f["FOP"] > 0), f["FOP"] / prev - 1, np.nan)
    # 期末決算: 来期予想 ÷ 今期実績（増益予想か減益予想か）
    f["nx_growth"] = np.where((f["CurPerType"] == "FY") & (f["OP"] > 0) & (f["NxFOP"] > 0),
                              f["NxFOP"] / f["OP"] - 1, np.nan)
    # 前年同期比（同じ四半期区分の1年前の実績と比べる。2年分しか無いので半分ほどしか計算できない）
    f["fy_year"] = pd.to_datetime(f["CurFYEn"], errors="coerce").dt.year
    last = f[f["kind"] == "決算"].drop_duplicates(["code", "CurPerType", "fy_year"], keep="last")
    last = last[["code", "CurPerType", "fy_year", "OP"]].rename(columns={"OP": "OP_prev"})
    last["fy_year"] += 1
    f = f.merge(last, on=["code", "CurPerType", "fy_year"], how="left")
    f["yoy_op"] = np.where((f["kind"] == "決算") & (f["OP_prev"] > 0) & (f["OP"] > 0),
                           f["OP"] / f["OP_prev"] - 1, np.nan)
    # 通期予想に対する進み具合（累計の営業利益 ÷ 通期予想）を、期間の経過割合と比べる
    frac = f["CurPerType"].map({"1Q": 0.25, "2Q": 0.5, "3Q": 0.75})
    f["progress"] = np.where((f["FOP"] > 0) & frac.notna(), f["OP"] / f["FOP"] - frac, np.nan)
    return f


def attach_returns(p: Panel, f: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    O, C = p.o.to_numpy("float64"), p.c.to_numpy("float64")
    bo, bc = p.o[BENCH].to_numpy("float64"), p.c[BENCH].to_numpy("float64")
    n = len(p.dates)
    r = f["r"].to_numpy()
    j = p.codes.get_indexer(f["code"])
    f = f.copy()
    f["ok"] = base.to_numpy()[r, j]                                  # 反応日の時点で買える銘柄か
    prev_c = C[np.maximum(r - 1, 0), j]
    f["react"] = C[r, j] / prev_c - 1                                # 反応日の騰落（前日終値→当日終値）
    f["react_date"] = p.dates[r]
    e = r + 1
    for h in HORIZONS:
        x = np.minimum(e + h - 1, n - 1)
        valid = (e + h - 1) < n
        ret = C[x, j] / O[e, j] - 1 - COST
        mkt = bc[x] / bo[e] - 1
        f[f"x{h}"] = np.where(valid, ret - mkt, np.nan)
    return f


def summarize(f: pd.DataFrame, name: str, m: pd.Series) -> dict:
    g = f[m & f["ok"]]
    row = {"条件": name, "件数": int(len(g))}
    for h in HORIZONS:
        v = g[["react_date", f"x{h}"]].dropna()
        if len(v) < 30:
            row[f"{h}日"] = "–"
            continue
        mon = v.groupby(v["react_date"].dt.to_period("M"))[f"x{h}"].mean()   # 月ごとの平均でt値
        t = mon.mean() / (mon.std(ddof=1) / np.sqrt(len(mon))) if len(mon) > 3 and mon.std(ddof=1) > 0 else 0.0
        row[f"{h}日"] = f"{v[f'x{h}'].mean() * 100:+.2f}% (t{t:+.1f})"
        row[f"_x{h}"], row[f"_t{h}"] = float(v[f"x{h}"].mean()), float(t)
    return row


def run() -> pd.DataFrame:
    cfg = load_config()
    p = Panel(data.load("bt"))
    base = p.tradable(cfg["capital"] / cfg["lot"], cfg["min_turnover"])
    f = attach_returns(p, load_events(p), base)
    print(f"開示 {len(f):,}件（決算 {(f.kind == '決算').sum():,} / 修正 {(f.kind == '修正').sum():,}）"
          f" 期間 {f.react_date.min().date()}〜{f.react_date.max().date()} / うち買える銘柄 {int(f.ok.sum()):,}件")
    k, s = f["kind"] == "決算", f["kind"] == "修正"
    up, dn = f["react"] > 0.03, f["react"] < -0.03
    tests = [
        ("【基準】決算の全件", k),
        ("【基準】修正の全件", s),
        ("修正: 営業利益予想を+10%以上 上方修正", s & (f.rev_op >= 0.10)),
        ("修正: 営業利益予想を+30%以上 上方修正", s & (f.rev_op >= 0.30)),
        ("修正: 上方修正 & 反応日に+3%超", s & (f.rev_op >= 0.10) & up),
        ("修正: 上方修正 & 反応日が+3%以下", s & (f.rev_op >= 0.10) & ~up),
        ("修正: 営業利益予想を-10%以下 下方修正", s & (f.rev_op <= -0.10)),
        ("決算: 同時に営業利益予想を+5%以上 上方修正", k & (f.rev_op >= 0.05)),
        ("決算: 同時に営業利益予想を-5%以下 下方修正", k & (f.rev_op <= -0.05)),
        ("決算: 営業利益が前年同期比+30%以上", k & (f.yoy_op >= 0.30)),
        ("決算: 営業利益が前年同期比-30%以下", k & (f.yoy_op <= -0.30)),
        ("決算: 進み具合が計画より+10pt以上先行", k & (f.progress >= 0.10)),
        ("決算: 進み具合が計画より-10pt以上遅れ", k & (f.progress <= -0.10)),
        ("期末決算: 来期の営業利益予想が+15%以上", k & (f.nx_growth >= 0.15)),
        ("期末決算: 来期の営業利益予想が減益", k & (f.nx_growth < 0)),
        ("決算: 反応日に+5%超（好感された）", k & (f.react > 0.05)),
        ("決算: 反応日に-5%未満（失望された）", k & (f.react < -0.05)),
        ("決算: 前年同期比+30%以上 & 反応日に+3%超", k & (f.yoy_op >= 0.30) & up),
        ("決算: 前年同期比+30%以上 & 反応日に-3%未満", k & (f.yoy_op >= 0.30) & dn),
        ("決算: 進み具合が先行 & 反応日に+3%超", k & (f.progress >= 0.10) & up),
    ]
    out = pd.DataFrame([summarize(f, n, m) for n, m in tests])
    print(out[["条件", "件数"] + [f"{h}日" for h in HORIZONS]].to_string(index=False))
    return out


if __name__ == "__main__":
    run()
