"""
日本株のスイングトレーダーの手法と「利確」の予選（2026-11-27）。学習期間（in_sample_end まで）の成績だけを見る。

 仮説A bnf_reversal : BNF流（25日線乖離-25%以下で買い、乖離の半値戻しで利確、3営業日で撤退）
 仮説B 利確の指値   : 既存3戦略の買いと損切りはそのまま、+5%・+10%・ATR×1.5で利確し、最長10営業日で売る
 仮説C record_runup : 権利取りの先回り（3月・9月の権利付き最終日の60営業日前に買い、5営業日前に売る）
 仮説D kahanshin    : 相場師朗流の下半身（PPPの並びで5日線を陽線で上抜けた初動、5営業日で手仕舞い）

予選の合格ライン（学習ループと同じ）: 学習期間の取引200件以上・市場平均との差がプラス・t値2.0以上。
試した設定はすべて学習ノートの試行数に数える。検証期間の成績はここでは計算しない。

使い方: python scripts/study_japanese_traders.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd

from src import data, journal
from src.engine import ExitSpec, Panel, simulate, stats
from src.research import load_config
from src.strategies import build_signal

PRELIM_T, PRELIM_N = 2.0, 200


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
        reasons = t["reason"].value_counts(normalize=True).round(2).to_dict() if len(t) else {}
        rows.append({"案": label, "取引": s.get("n", 0), "勝率": s.get("win_rate"), "平均損益": s.get("avg_ret"),
                     "市場平均との差": s.get("avg_excess"), "t値": s.get("t_excess"), "PF": s.get("profit_factor"),
                     "平均保有": s.get("avg_hold"), "売った理由": reasons, "予選": "通過" if ok else "落選"})
        print(f"  {label}: {rows[-1]['取引']}件 勝率{(s.get('win_rate') or 0):.0%} 差{(s.get('avg_excess') or 0):+.2%} "
              f"t{(s.get('t_excess') or 0):+.2f} → {rows[-1]['予選']}  {reasons}", flush=True)
        return ok

    print("仮説A・C・D（本人・研究の説明どおりの既定値）")
    run("仮説A BNF流（乖離-25%・半値戻しで利確・3日）", "bnf_reversal", {})
    run("仮説C 権利取りの先回り（60日前に買い・5日前に売り）", "record_runup", {})
    run("仮説D 相場師朗流の下半身（5日で手仕舞い）", "kahanshin", {})

    print("仮説B 利確の指値（買いと損切りは既存の設定のまま・最長10営業日）")
    for spec in cfg["watch"] + cfg["adopted"]:
        if spec["name"] not in ("pullback", "breakout", "gap_drift") or spec.get("filters"):
            continue
        _, _, ex0 = build_signal(p, base, spec["name"], spec["params"])
        for tag, kw in (("+5%", {"target_pct": 0.05}), ("+10%", {"target_pct": 0.10}), ("ATR×1.5", {"target_atr": 1.5})):
            ex = ExitSpec(stop_atr=ex0.stop_atr, max_hold=10, **kw)
            run(f"仮説B {spec['name']} → {tag}で利確", spec["name"], spec["params"], ex,
                {**spec["params"], "exit": f"target_{tag}_hold10"})

    new = journal.register(j, keys)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "research", "japanese_traders_prelim.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    journal.save(j)
    print(f"新たに数えた試行 {new}件 / これまでの試行 {journal.trials(j)}件 / 保存 {out}")


if __name__ == "__main__":
    main()
