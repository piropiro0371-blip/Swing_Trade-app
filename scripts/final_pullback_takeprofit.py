"""
本選（2026-11-27）: 予選を通った「押し目買い＋+5%で利確・最長10営業日」を、検証期間も含めた5基準で採点する。

学習ループ（src/learn.py）の本選と同じ手順:
 - 設定を変えても崩れないか（基準4）は、押し目買いの検証用の組み合わせすべてに同じ出口を付けて、全期間でプラスの割合を見る
 - 必要なt値は、これまでに試した仮説の数から決まる（src/journal.py の t_required）
合格した場合だけ config/strategy.json に「検証中（バックテスト合格）」として登録し、フォワード確認に回す。

使い方: python scripts/final_pullback_takeprofit.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import data, journal
from src.engine import Panel, simulate, stats
from src.research import OUT_WEB, evaluate, load_config, report, save_config, upsert, write_validation
from src.strategies import build_signal, grid

EXIT = {"target_pct": 0.05, "rebound_exit": False, "max_hold": 10}
SID, LABEL = "pullback_tp5", "押し目買い（+5%で利確・最長10日）"


def main():
    cfg = load_config()
    p = Panel(data.load("bt"))
    base = p.tradable(max_price=cfg["capital"] / cfg["lot"], min_turnover=cfg["min_turnover"])
    params = {**next(x for x in cfg["watch"] if x.get("id") == "pullback")["params"], **EXIT}
    fam = []
    for gp in grid("pullback"):
        sig, score, ex = build_signal(p, base, "pullback", {**gp, **EXIT})
        s = stats(simulate(p, sig, score, ex, cost=cfg["cost_per_side"]))
        fam.append(bool(s.get("n")) and s["avg_ret"] > 0)
    j = journal.load()
    t_req = journal.t_required(j)
    print(f"設定: {params} / 必要t値 {t_req} / 設定を変えてもプラスの割合 {sum(fam)}/{len(fam)}")
    entry, _ = evaluate(p, base, cfg, "pullback", params, None, sum(fam) / len(fam), t_req, SID, LABEL)
    report(entry, cfg)
    print("5基準:", entry["checks"], "→", "合格" if entry["passed"] else "不合格")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "research", "final_pullback_tp5.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({k: entry[k] for k in ("params", "in_sample", "out_sample", "robust", "years_pos_ratio",
                                         "t_required", "checks", "passed")}, f, ensure_ascii=False, indent=1, default=str)
    if entry["passed"]:
        with open(OUT_WEB, encoding="utf-8") as f:
            val = json.load(f)
        val["strategies"].append(entry)
        write_validation(val)
        upsert(cfg, SID, "pullback", params, None, LABEL, True)
        save_config(cfg)
        print("合格: 検証中（フォワード確認）として登録しました")


if __name__ == "__main__":
    main()
