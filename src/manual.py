"""
手動バックテスト: 戦略と設定値を自分で指定して、過去10年のデータで試す。

- 結果は docs/data/manual.json に新しい順で貯まり、Web画面の「手動テスト」タブに表示される
- ここでの結果は参考で、戦略の採用・不採用には影響しない
  （採用は週1回の検証と学習ループの判定基準だけで決める）
- 試した回数は学習ノートの「試した仮説の数」に加算する。何度も試せば偶然良い結果が出るため

使い方（GitHubの実行フォームからは環境変数で渡される）:
  python -m src.manual --strategy pullback --params "rsi_max=5, stop_atr=2" --capital 200000 --positions 2 --risk 2
"""

from __future__ import annotations

import argparse
import inspect
import json
import os

import pandas as pd

from . import data, diagnose, journal
from .daily import clean_json
from .engine import Panel
from .research import evaluate, load_config, report
from .strategies import LABELS, STRATEGIES
from .universe import ROOT

OUT = os.path.join(ROOT, "docs", "data", "manual.json")
KEEP = 15   # 画面に残す件数

# 設定項目の説明（画面と実行フォームに出す）
PARAM_HELP = {
    "rsi_max": "2日RSIがこの値未満で買う（小さいほど強く売られた銘柄だけ）",
    "stop_atr": "損切り幅。買値から「1日の平均的な値幅(ATR)×この倍率」下",
    "max_hold": "最大保有日数（営業日）",
    "regime": "true なら市場全体が上昇基調のときだけ買う",
    "lookback": "何日間の高値を更新したら買うか",
    "vol_mult": "出来高が20日平均の何倍以上で買うか",
    "trail_atr": "終値の最高値から「ATR×この倍率」下へ逆指値を引き上げる",
    "gap": "前日終値より何割以上高く寄り付いたら買うか（0.05 = 5%）",
}


def defaults(name: str) -> dict:
    sig = inspect.signature(STRATEGIES[name])
    return {n: q.default for n, q in sig.parameters.items() if q.default is not inspect.Parameter.empty}


def parse_params(name: str, text: str, current: dict) -> dict:
    """「rsi_max=5, stop_atr=2」の形の指定を読む。指定の無い項目は現在の設定のまま。"""
    base = {**defaults(name), **current}
    out = dict(base)
    for part in [x.strip() for x in (text or "").replace("、", ",").replace("＝", "=").split(",") if x.strip()]:
        if "=" not in part:
            raise SystemExit(f"設定の書き方が違います: 「{part}」 → 「項目=値」の形で、カンマ区切りにしてください")
        k, v = (s.strip() for s in part.split("=", 1))
        if k not in base:
            raise SystemExit(f"「{k}」は {LABELS[name]} の設定項目にありません。使える項目: {', '.join(base)}")
        if isinstance(base[k], bool):
            if v.lower() not in ("true", "false"):
                raise SystemExit(f"{k} は true か false で指定してください")
            out[k] = v.lower() == "true"
        else:
            try:
                out[k] = type(base[k])(float(v)) if isinstance(base[k], int) else float(v)
            except ValueError:
                raise SystemExit(f"{k} は数値で指定してください（指定された値: {v}）")
    return out


def run(strategy: str, params_text: str, capital: float, positions: int, risk: float, memo: str) -> dict:
    if strategy not in STRATEGIES:
        raise SystemExit(f"戦略名が違います: {strategy}（使えるもの: {', '.join(STRATEGIES)}）")
    if not (10_000 <= capital <= 100_000_000) or not (1 <= positions <= 20) or not (0 <= risk <= 100):
        raise SystemExit("資金は1万〜1億円、同時保有は1〜20銘柄、損失上限は0〜100%の範囲で指定してください")
    cfg = load_config()
    current = next((s["params"] for s in cfg["adopted"] + cfg["watch"] if s.get("id", s["name"]) == strategy), {})
    params = parse_params(strategy, params_text, current)
    cfg.update({"capital": capital, "max_positions": positions, "risk_pct": risk / 100 if risk > 0 else None})

    p = Panel(data.load("bt"))
    base = p.tradable(max_price=capital / cfg["lot"], min_turnover=cfg["min_turnover"])
    j = journal.load()
    new = journal.register(j, [journal.key(strategy, params) + f"|manual:{int(capital)}:{positions}:{risk}"])
    t_req = journal.t_required(j)
    entry, tr = evaluate(p, base, cfg, strategy, params, None, robust=0.0, t_req=t_req)
    print(f"[{strategy}] {params} 資金{capital:,.0f}円・同時{positions}銘柄・損失上限{risk}%", flush=True)
    report(entry, cfg)

    port = entry["portfolio"][str(positions)]
    result = {
        "run_at": pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d %H:%M"),
        "memo": memo[:200], "strategy": strategy, "label": LABELS[strategy], "params": params,
        "changed": {k: v for k, v in params.items() if {**defaults(strategy), **current}.get(k) != v},
        "capital": capital, "max_positions": positions, "risk_pct": risk,
        "period": [p.dates[0].strftime("%Y-%m-%d"), p.dates[-1].strftime("%Y-%m-%d")],
        "in_sample_end": cfg["in_sample_end"], "t_required": t_req,
        "in_sample": entry["in_sample"], "out_sample": entry["out_sample"], "all": entry["all"],
        "portfolio": port, "reasons": diagnose.by_reason(tr) if len(tr) else [],
        # 基準4（設定を変えても崩れないか）は1通りだけの試行では判定できないので除く
        "checks": {k: v for k, v in entry["checks"].items() if k != "c4"},
    }
    hist = []
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            hist = json.load(f).get("runs", [])
    out = {"runs": ([result] + hist)[:KEEP],
           "strategies": {n: {"label": LABELS[n], "defaults": {**defaults(n), **next(
               (s["params"] for s in cfg["adopted"] + cfg["watch"] if s.get("id", s["name"]) == n), {})}}
               for n in STRATEGIES},
           "param_help": PARAM_HELP}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(clean_json(out), f, ensure_ascii=False, allow_nan=False)
    if new:
        journal.save(j)
    print(f"保存: {OUT}（履歴 {len(out['runs'])}件）")
    return result


if __name__ == "__main__":
    env = os.environ.get
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default=env("BT_STRATEGY", "pullback"))
    ap.add_argument("--params", default=env("BT_PARAMS", ""))
    ap.add_argument("--capital", type=float, default=float(env("BT_CAPITAL") or 200000))
    ap.add_argument("--positions", type=int, default=int(env("BT_POSITIONS") or 2))
    ap.add_argument("--risk", type=float, default=float(env("BT_RISK") or 2))
    ap.add_argument("--memo", default=env("BT_MEMO", ""))
    a = ap.parse_args()
    run(a.strategy.split()[0], a.params, a.capital, a.positions, a.risk, a.memo)
