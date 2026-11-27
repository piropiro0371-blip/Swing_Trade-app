"""
日本株で学術的に確認されている現象の予選（2026-11-27）。学習期間（in_sample_end まで）の成績だけを見る。

 週単位のリバーサル : 直近5日の下落率が下位5%の銘柄を買い、5営業日で売る
 月単位のリバーサル : 直近20日の下落率が下位5%の銘柄を買い、20営業日で売る（Iihara・Kato・Tokunaga 2004 の1か月反転）
 低ボラティリティ株 : 毎月最初の営業日に、直近60日の値動きが小さい下位10%の銘柄を買い、20営業日で売る

予選の合格ライン（学習ループと同じ）: 学習期間の取引200件以上・市場平均との差がプラス・t値2.0以上。
試した設定はすべて学習ノートの試行数に数える。検証期間の成績はここでは計算しない。

使い方: python scripts/study_anomalies.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd

from src import data, journal
from src.engine import Panel, simulate, stats
from src.research import load_config
from src.strategies import build_signal

PRELIM_T, PRELIM_N = 2.0, 200
CASES = [
    ("週単位のリバーサル（5日の下位5%・5日保有）", "reversal", {"look": 5, "bottom": 0.05, "hold": 5}),
    ("月単位のリバーサル（20日の下位5%・20日保有）", "reversal", {"look": 20, "bottom": 0.05, "hold": 20}),
    ("低ボラティリティ株（60日の下位10%・毎月・20日保有）", "lowvol", {"look": 60, "bottom": 0.10, "hold": 20}),
]


def main():
    cfg = load_config()
    is_end = pd.Timestamp(cfg["in_sample_end"])
    p = Panel(data.load("bt"))
    base = p.tradable(max_price=cfg["capital"] / cfg["lot"], min_turnover=cfg["min_turnover"])
    print(f"データ {p.dates[0]:%Y-%m-%d}〜{p.dates[-1]:%Y-%m-%d} / {len(p.codes)}銘柄 / 学習期間〜{is_end:%Y-%m-%d}")
    j = journal.load()
    rows, keys = [], []
    for label, name, params in CASES:
        sig, score, ex = build_signal(p, base, name, params)
        t = simulate(p, sig, score, ex, cost=cfg["cost_per_side"])
        t = t[t["entry_date"] <= is_end]
        s = stats(t)
        keys.append(journal.key(name, params))
        ok = s.get("n", 0) >= PRELIM_N and s["avg_excess"] > 0 and s["t_excess"] >= PRELIM_T
        rows.append({"案": label, "名前": name, "設定": params, "取引": s.get("n", 0), "勝率": s.get("win_rate"),
                     "平均損益": s.get("avg_ret"), "市場平均との差": s.get("avg_excess"), "t値": s.get("t_excess"),
                     "PF": s.get("profit_factor"), "予選": "通過" if ok else "落選"})
        print(f"  {label}: {s.get('n', 0)}件 勝率{(s.get('win_rate') or 0):.0%} 平均{(s.get('avg_ret') or 0):+.2%} "
              f"差{(s.get('avg_excess') or 0):+.2%} t{(s.get('t_excess') or 0):+.2f} → {rows[-1]['予選']}", flush=True)
    new = journal.register(j, keys)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "research", "anomalies_prelim.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    journal.save(j)
    print(f"新たに数えた試行 {new}件 / これまでの試行 {journal.trials(j)}件")


if __name__ == "__main__":
    main()
