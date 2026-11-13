"""
バックテストによる戦略の検証（判定基準の合否判定つき）。

手順:
 1. 各戦略のパラメータ候補を全部試す
 2. 「学習期間」(〜in_sample_end) の成績だけを見て、戦略ごとに最良の設定を1つ選ぶ
 3. 選んだ設定を、選ぶときに見ていない「検証期間」(それ以降) で採点する
 4. 5つの判定基準すべてに合格した戦略だけを「採用」とする

使い方:  python -m src.research
出力:    docs/data/validation.json（Web画面用） / data/research/grid.csv（全組み合わせの成績）
"""

from __future__ import annotations

import json
import os
import time

import pandas as pd

from . import data
from .engine import Panel, portfolio, simulate, stats
from .strategies import LABELS, STRATEGIES, grid
from .universe import ROOT

CONFIG = os.path.join(ROOT, "config", "strategy.json")
OUT_WEB = os.path.join(ROOT, "docs", "data", "validation.json")
OUT_GRID = os.path.join(ROOT, "data", "research", "grid.csv")

CRITERIA = [
    ("c1", "学習期間で、市場平均を上回る利益が統計的に確かである（超過リターンのt値が2以上）"),
    ("c2", "検証期間（選ぶときに見ていない期間）でも、コスト込みで利益が出て市場平均を上回る"),
    ("c3", "年ごとに見て、7割以上の年でプラスである"),
    ("c4", "設定値を少し変えても崩れない（試した組み合わせの7割以上が全期間でプラス）"),
    ("c5", "実際の資金額で運用した想定でも検証期間がプラスで、最大下落が25%以内に収まる"),
]


def load_config() -> dict:
    with open(CONFIG, encoding="utf-8") as f:
        return json.load(f)


def slim(s: dict) -> dict:
    return {k: v for k, v in s.items() if k != "yearly"}


def run() -> dict:
    cfg = load_config()
    is_end = pd.Timestamp(cfg["in_sample_end"])
    oos_start = is_end + pd.Timedelta(days=1)
    cap, lot, cost = cfg["capital"], cfg["lot"], cfg["cost_per_side"]

    print("データ読み込み中...", flush=True)
    p = Panel(data.load("bt"))
    base = p.tradable(max_price=cap / lot, min_turnover=cfg["min_turnover"])
    print(f"期間 {p.dates[0].date()}〜{p.dates[-1].date()} / {len(p.codes)} 銘柄", flush=True)

    grid_rows = []
    result = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "period": [p.dates[0].strftime("%Y-%m-%d"), p.dates[-1].strftime("%Y-%m-%d")],
        "in_sample_end": cfg["in_sample_end"],
        "capital": cap,
        "n_codes": int(len(p.codes)),
        "criteria": [{"id": i, "text": t} for i, t in CRITERIA],
        "strategies": [],
    }

    for name, fn in STRATEGIES.items():
        t0 = time.time()
        best = None
        fam = []
        for params in grid(name):
            sig, score, ex = fn(p, base, **params)
            tr = simulate(p, sig, score, ex, cost=cost)
            s_is = stats(tr[tr["entry_date"] <= is_end])
            s_oos = stats(tr[tr["entry_date"] >= oos_start])
            s_all = stats(tr)
            fam.append(s_all.get("avg_ret", 0) > 0 if s_all["n"] else False)
            grid_rows.append({"strategy": name, **params,
                              **{f"is_{k}": v for k, v in slim(s_is).items()},
                              **{f"oos_{k}": v for k, v in slim(s_oos).items()}})
            # 学習期間の成績だけで選ぶ（十分な取引数があるものに限る）
            if s_is["n"] >= 200 and (best is None or s_is["t_excess"] > best["is"]["t_excess"]):
                best = {"params": params, "is": s_is, "oos": s_oos, "all": s_all, "trades": tr}
        robust = sum(fam) / len(fam)
        if best is None:
            print(f"[{name}] 取引数が足りず評価できません", flush=True)
            continue

        tr = best["trades"]
        ports = {}
        for mp in (1, 2):
            ports[str(mp)] = {
                "oos": portfolio(p, tr, cap, mp, lot, start=oos_start),
                "all": portfolio(p, tr, cap, mp, lot),
            }
        port = ports[str(cfg["max_positions"])]
        yr_ok = best["all"]["years_pos"] / best["all"]["years"]
        checks = {
            "c1": best["is"]["avg_excess"] > 0 and best["is"]["t_excess"] >= 2.0,
            "c2": best["oos"]["n"] >= 100 and best["oos"]["avg_ret"] > 0 and best["oos"]["avg_excess"] > 0,
            "c3": yr_ok >= 0.7,
            "c4": robust >= 0.7,
            "c5": port["oos"].get("total_ret", -1) > 0 and port["oos"].get("max_dd", -1) >= -0.25,
        }
        entry = {
            "name": name, "label": LABELS[name], "params": best["params"],
            "in_sample": slim(best["is"]), "out_sample": slim(best["oos"]), "all": best["all"],
            "robust": robust, "years_pos_ratio": yr_ok,
            "portfolio": ports, "checks": checks, "passed": all(checks.values()),
        }
        result["strategies"].append(entry)
        print(f"[{name}] {time.time() - t0:.0f}秒  最良設定 {best['params']}", flush=True)
        for tag, s in (("学習", best["is"]), ("検証", best["oos"])):
            print(f"   {tag}: n={s['n']} 勝率={s['win_rate']:.1%} 平均={s['avg_ret']:+.2%} "
                  f"超過={s['avg_excess']:+.2%} t(超過)={s['t_excess']:.2f} PF={s['profit_factor']:.2f} "
                  f"保有={s['avg_hold']:.1f}日", flush=True)
        print(f"   プラス年 {best['all']['years_pos']}/{best['all']['years']}  頑健性 {robust:.0%}  "
              f"資金{cap:,}円・{cfg['max_positions']}銘柄(検証期間): 損益 {port['oos'].get('total_ret', 0):+.1%} "
              f"最大下落 {port['oos'].get('max_dd', 0):.1%} "
              f"(TOPIX {port['oos'].get('bench_total_ret', 0):+.1%})", flush=True)
        print(f"   判定: {checks} → {'採用' if entry['passed'] else '不採用'}", flush=True)

    # 合格した戦略は「採用」、不合格は「検証中」として設定ファイルに反映する
    cfg["adopted"] = [{"name": s["name"], "params": s["params"]} for s in result["strategies"] if s["passed"]]
    cfg["watch"] = [{"name": s["name"], "params": s["params"]} for s in result["strategies"] if not s["passed"]]
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(OUT_WEB), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_GRID), exist_ok=True)
    from .daily import clean_json
    with open(OUT_WEB, "w", encoding="utf-8") as f:
        json.dump(clean_json(result), f, ensure_ascii=False, allow_nan=False)
    pd.DataFrame(grid_rows).to_csv(OUT_GRID, index=False, encoding="utf-8-sig")
    print(f"保存: {OUT_WEB}\n      {OUT_GRID}")
    return result


if __name__ == "__main__":
    run()
