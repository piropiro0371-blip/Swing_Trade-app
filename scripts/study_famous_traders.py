"""
著名スイングトレーダーの手法の予選（2026-11-27）。学習期間（in_sample_end まで）の成績だけを見る。

 仮説1 trend_vcp : ミネルヴィニ流（トレンドテンプレート＋収縮後のブレイク、-7%損切り、50日線割れで売り）
 仮説2 qulla     : クラマギー流（急騰株のもみ合い放れ、シグナル日の安値で損切り、10日線割れで売り）
 仮説3 出口の入れ替え: 既存3戦略の買いはそのままに、売りを「-7%損切り＋移動平均割れ（10/20/50日）」にする

予選の合格ライン（学習ループと同じ）: 学習期間の取引200件以上・市場平均との差がプラス・t値2.0以上。
試した設定はすべて学習ノートの試行数に数える。検証期間の成績はここでは計算しない。

使い方: python scripts/study_famous_traders.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from src import data, journal
from src.engine import ExitSpec, Panel, simulate, stats
from src.research import load_config
from src.strategies import build_signal

PRELIM_T, PRELIM_N = 2.0, 200


def top_share(t: pd.DataFrame) -> float:
    """利益の合計のうち、上位5%の取引が占める割合（大勝ちに頼る度合い）"""
    r = t["ret"].sort_values(ascending=False)
    pos = r[r > 0].sum()
    return float(r.head(max(1, len(r) // 20)).sum() / pos) if pos > 0 else float("nan")


def main():
    cfg = load_config()
    is_end = pd.Timestamp(cfg["in_sample_end"])
    p = Panel(data.load("bt"))
    base = p.tradable(max_price=cfg["capital"] / cfg["lot"], min_turnover=cfg["min_turnover"])
    print(f"データ {p.dates[0]:%Y-%m-%d}〜{p.dates[-1]:%Y-%m-%d} / {len(p.codes)}銘柄 / 学習期間〜{is_end:%Y-%m-%d}")

    j = journal.load()
    rows, keys = [], []

    def run(label, name, params, ex=None, key_params=None):
        sig, score, ex0 = build_signal(p, base, name, params)
        t = simulate(p, sig, score, ex or ex0, cost=cfg["cost_per_side"])
        t = t[t["entry_date"] <= is_end]   # 学習期間だけ
        s = stats(t)
        keys.append(journal.key(name, key_params or params))
        ok = s.get("n", 0) >= PRELIM_N and s["avg_excess"] > 0 and s["t_excess"] >= PRELIM_T
        rows.append({"案": label, "取引": s.get("n", 0), "勝率": s.get("win_rate"), "平均損益": s.get("avg_ret"),
                     "市場平均との差": s.get("avg_excess"), "t値": s.get("t_excess"), "PF": s.get("profit_factor"),
                     "平均保有": s.get("avg_hold"), "上位5%の利益占有": top_share(t) if len(t) else None,
                     "期限切れ": int((t["reason"] == "期限").sum()) if len(t) else 0, "予選": "通過" if ok else "落選"})
        print(f"  {label}: {rows[-1]['取引']}件 差{(s.get('avg_excess') or 0):+.2%} t{(s.get('t_excess') or 0):+.2f} → {rows[-1]['予選']}",
              flush=True)

    print("仮説1・2（本人の説明どおりの既定値）")
    run("仮説1 ミネルヴィニ流（既定）", "trend_vcp", {})
    run("仮説2 クラマギー流（既定）", "qulla", {})

    print("仮説3 出口の入れ替え（買いは既存の設定のまま）")
    for spec in cfg["watch"] + cfg["adopted"]:
        if spec["name"] not in ("pullback", "breakout", "gap_drift") or spec.get("filters"):
            continue
        for ma in (10, 20, 50):
            ex = ExitSpec(stop_pct=0.07, exit_below_sma=ma, max_hold=60)
            run(f"仮説3 {spec['name']} → -7%損切り＋{ma}日線割れで売り", spec["name"], spec["params"], ex,
                {**spec["params"], "exit": f"pct7_sma{ma}"})

    new = journal.register(j, keys)
    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "research", "famous_traders_prelim.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    journal.save(j)
    print(f"新たに数えた試行 {new}件 / これまでの試行 {journal.trials(j)}件 / 保存 {out}")


if __name__ == "__main__":
    main()
