"""
バックテストによる戦略の検証（判定基準の合否判定つき）。

手順:
 1. 各戦略のパラメータ候補を全部試す
 2. 「学習期間」(〜in_sample_end) の成績だけを見て、戦略ごとに最良の設定を1つ選ぶ
 3. 選んだ設定を、選ぶときに見ていない「検証期間」(それ以降) で採点する
 4. 5つの判定基準すべてに合格した戦略を「バックテスト合格」とする
    → その後フォワードテストで確認できて初めて「採用」になる（src/daily.py）

使い方:  python -m src.research
出力:    docs/data/validation.json（Web画面用） / data/research/grid.csv（全組み合わせの成績）
"""

from __future__ import annotations

import json
import os
import time

import pandas as pd

from . import data, features, journal
from .engine import Panel, portfolio, simulate, stats
from .strategies import LABELS, STRATEGIES, build_signal, grid
from .universe import ROOT

CONFIG = os.path.join(ROOT, "config", "strategy.json")
OUT_WEB = os.path.join(ROOT, "docs", "data", "validation.json")
OUT_GRID = os.path.join(ROOT, "data", "research", "grid.csv")

CRITERIA = [
    ("c1", "学習期間で、市場平均を上回る利益が統計的に確かである（超過リターンのt値が必要水準以上。必要水準は、試した仮説の数が増えるほど2.0から3.0まで上がる）"),
    ("c2", "検証期間（選ぶときに見ていない期間）でも、コスト込みで利益が出て市場平均を上回る"),
    ("c3", "年ごとに見て、7割以上の年でプラスである"),
    ("c4", "設定値を少し変えても崩れない（試した組み合わせの7割以上が全期間でプラス）"),
    ("c5", "実際の資金額で運用した想定でも検証期間がプラスで、最大下落が25%以内に収まる"),
]


def load_config() -> dict:
    with open(CONFIG, encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg: dict) -> None:
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def slim(s: dict) -> dict:
    return {k: v for k, v in s.items() if k != "yearly"}


def sid_of(item: dict) -> str:
    return item.get("id", item["name"])


def upsert(cfg: dict, sid: str, name: str, params: dict, filters: list | None, label: str, passed: bool) -> None:
    """設定ファイルの戦略一覧に反映する。バックテスト合格だけでは「採用」にしない
    （フォワードテストでの確認が必要）。合格を失った採用中の戦略は検証中に戻す。"""
    item, where = None, None
    for lst in ("adopted", "watch"):
        for it in cfg[lst]:
            if sid_of(it) == sid:
                item, where = it, lst
    if item is None:
        item, where = {"id": sid, "name": name}, "watch"
        cfg["watch"].append(item)
    item.update({"id": sid, "name": name, "params": params, "label": label, "passed_backtest": bool(passed)})
    if filters:
        item["filters"] = filters
    if passed:
        if item.get("demoted_by") == "backtest":   # バックテスト起因の降格は、再合格で解除
            item.pop("demoted", None)
            item.pop("demoted_by", None)
    elif where == "adopted":
        cfg["adopted"].remove(item)
        item["demoted"] = pd.Timestamp.now().strftime("%Y-%m-%d")
        item["demoted_by"] = "backtest"
        cfg["watch"].append(item)


def evaluate(p: Panel, base, cfg: dict, name: str, params: dict, filters: list | None,
             robust: float, t_req: float, sid: str | None = None, label: str | None = None):
    """1つの設定を学習・検証の両期間で採点し、5基準の合否を出す。"""
    is_end = pd.Timestamp(cfg["in_sample_end"])
    oos_start = is_end + pd.Timedelta(days=1)
    cap, lot = cfg["capital"], cfg["lot"]
    sig, score, ex = build_signal(p, base, name, params, filters)
    tr = simulate(p, sig, score, ex, cost=cfg["cost_per_side"])
    s_is = stats(tr[tr["entry_date"] <= is_end])
    s_oos = stats(tr[tr["entry_date"] >= oos_start])
    s_all = stats(tr)
    rp = cfg.get("risk_pct")
    ports = {str(mp): {"oos": portfolio(p, tr, cap, mp, lot, start=oos_start, risk_pct=rp),
                       "all": portfolio(p, tr, cap, mp, lot, risk_pct=rp)} for mp in (1, 2)}
    port = ports[str(cfg["max_positions"])]
    yr_ok = s_all["years_pos"] / s_all["years"] if s_all.get("years") else 0.0
    checks = {
        "c1": s_is.get("n", 0) >= 200 and s_is["avg_excess"] > 0 and s_is["t_excess"] >= t_req,
        "c2": s_oos.get("n", 0) >= 100 and s_oos["avg_ret"] > 0 and s_oos["avg_excess"] > 0,
        "c3": yr_ok >= 0.7,
        "c4": robust >= 0.7,
        "c5": port["oos"].get("total_ret", -1) > 0 and port["oos"].get("max_dd", -1) >= -0.25,
    }
    entry = {
        "id": sid or name, "name": name, "label": label or LABELS[name], "params": params,
        "filters": filters or [], "filters_text": [features.describe(f) for f in (filters or [])],
        "in_sample": slim(s_is), "out_sample": slim(s_oos), "all": s_all,
        "robust": robust, "years_pos_ratio": yr_ok, "t_required": t_req,
        "portfolio": ports, "checks": checks, "passed": all(checks.values()),
    }
    return entry, tr


def report(entry: dict, cfg: dict) -> None:
    for tag, s in (("学習", entry["in_sample"]), ("検証", entry["out_sample"])):
        if not s.get("n"):
            print(f"   {tag}: 取引なし", flush=True)
            continue
        print(f"   {tag}: n={s['n']} 勝率={s['win_rate']:.1%} 平均={s['avg_ret']:+.2%} "
              f"超過={s['avg_excess']:+.2%} t(超過)={s['t_excess']:.2f} PF={s['profit_factor']:.2f} "
              f"保有={s['avg_hold']:.1f}日", flush=True)
    port = entry["portfolio"][str(cfg["max_positions"])]["oos"]
    a = entry["all"]
    print(f"   プラス年 {a.get('years_pos', 0)}/{a.get('years', 0)}  頑健性 {entry['robust']:.0%}  "
          f"資金{cfg['capital']:,}円(検証期間): 損益 {port.get('total_ret', 0):+.1%} "
          f"最大下落 {port.get('max_dd', 0):.1%} (TOPIX {port.get('bench_total_ret', 0):+.1%})", flush=True)
    print(f"   判定(必要t値 {entry['t_required']}): {entry['checks']} → "
          f"{'バックテスト合格' if entry['passed'] else '不合格'}", flush=True)


def write_validation(result: dict) -> None:
    from .daily import clean_json
    os.makedirs(os.path.dirname(OUT_WEB), exist_ok=True)
    with open(OUT_WEB, "w", encoding="utf-8") as f:
        json.dump(clean_json(result), f, ensure_ascii=False, allow_nan=False)


def run() -> dict:
    cfg = load_config()
    is_end = pd.Timestamp(cfg["in_sample_end"])
    cap, lot, cost = cfg["capital"], cfg["lot"], cfg["cost_per_side"]
    j = journal.load()

    print("データ読み込み中...", flush=True)
    p = Panel(data.load("bt"))
    base = p.tradable(max_price=cap / lot, min_turnover=cfg["min_turnover"])
    print(f"期間 {p.dates[0].date()}〜{p.dates[-1].date()} / {len(p.codes)} 銘柄", flush=True)

    # 1〜2: 全設定を試し、学習期間の成績だけで最良を選ぶ
    grid_rows, picked, keys = [], {}, []
    for name in STRATEGIES:
        t0 = time.time()
        best, fam = None, []
        for params in grid(name):
            keys.append(journal.key(name, params))
            sig, score, ex = build_signal(p, base, name, params)
            tr = simulate(p, sig, score, ex, cost=cost)
            s_is = stats(tr[tr["entry_date"] <= is_end])
            s_oos = stats(tr[tr["entry_date"] > is_end])
            s_all = stats(tr)
            fam.append(bool(s_all["n"]) and s_all["avg_ret"] > 0)
            grid_rows.append({"strategy": name, **params,
                              **{f"is_{k}": v for k, v in slim(s_is).items()},
                              **{f"oos_{k}": v for k, v in slim(s_oos).items()}})
            if s_is["n"] >= 200 and (best is None or s_is["t_excess"] > best[1]):
                best = (params, s_is["t_excess"])
        if best:
            picked[name] = (best[0], sum(fam) / len(fam))
        print(f"[{name}] {len(fam)}通りを{time.time() - t0:.0f}秒で検証", flush=True)

    new = journal.register(j, keys)
    t_req = journal.t_required(j)

    # 3〜4: 選んだ設定を採点
    result = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "period": [p.dates[0].strftime("%Y-%m-%d"), p.dates[-1].strftime("%Y-%m-%d")],
        "in_sample_end": cfg["in_sample_end"], "capital": cap, "n_codes": int(len(p.codes)),
        "trials_total": journal.trials(j), "t_required": t_req,
        "criteria": [{"id": i, "text": t} for i, t in CRITERIA],
        "strategies": [],
    }
    for name, (params, robust) in picked.items():
        entry, _ = evaluate(p, base, cfg, name, params, None, robust, t_req)
        result["strategies"].append(entry)
        print(f"[{name}] 最良設定 {params}", flush=True)
        report(entry, cfg)
        upsert(cfg, name, name, params, None, LABELS[name], entry["passed"])

    if new:
        journal.add(j, "検証", f"基本戦略 {len(picked)}種の検証（新たに{new}通りを試行）",
                    "各戦略の設定候補をすべて試し、学習期間で最良の設定を選んで検証期間で採点した。",
                    [f"{e['label']}: 検証期間の市場平均との差 {e['out_sample'].get('avg_excess', 0):+.2%}"
                     f"（t値 {e['out_sample'].get('t_excess', 0):+.1f}）→ {'合格' if e['passed'] else '不合格'}"
                     for e in result["strategies"]],
                    result=f"{sum(e['passed'] for e in result['strategies'])}/{len(result['strategies'])} 合格")
    journal.save(j)
    save_config(cfg)
    write_validation(result)
    os.makedirs(os.path.dirname(OUT_GRID), exist_ok=True)
    pd.DataFrame(grid_rows).to_csv(OUT_GRID, index=False, encoding="utf-8-sig")
    print(f"保存: {OUT_WEB}\n      {OUT_GRID}")
    return result


if __name__ == "__main__":
    run()
