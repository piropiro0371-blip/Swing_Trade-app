"""
毎日の判定（大引け後に実行）。

 1. 最新の株価から、各戦略の「明日の買い候補」を出す
 2. 買い候補を台帳（docs/data/ledger.json）に追記する。台帳は後から書き換えない
 3. 台帳に載ったシグナルだけを対象に、ルール通りに売買したと仮定した成績を計算する
    （＝フォワードテスト。過去データへの当てはめではなく、出した時点で未来だった判定の成績）
 4. Web画面用のデータ（docs/data/*.json）を書き出す

使い方:  python -m src.daily            （先に python -m src.data --period 2y --tag daily）
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd

from . import data, diagnose, features, journal
from .engine import BENCH, Panel, simulate, stats
from .research import load_config, save_config, sid_of
from .strategies import LABELS, build_signal
from .universe import ROOT, load_all_listed

DOCS = os.path.join(ROOT, "docs", "data")
LEDGER = os.path.join(DOCS, "ledger.json")
MIN_FWD = 30        # フォワードの成績を判断するのに必要な決済数
MILESTONES = (30, 60, 100, 150, 200, 300, 500)
TOP_N = 10         # 画面に出す買い候補の数（戦略ごと）
QUOTE_DAYS = 60     # 保有株の売り判定用に配る終値の日数


def _load_json(path, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def clean_json(obj):
    """JSONに書けない値（欠損値・無限大）を null に置き換える"""
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _dump(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean_json(obj), f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _r(x, nd=1):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def health(closed: pd.DataFrame, bt: dict) -> dict:
    """フォワードの平均損益が、バックテストの期待からどれだけずれているか。
    z は「ずれ ÷ 偶然で起こりうる幅」。-2 より小さければ、偶然とは考えにくいほど期待を下回っている。"""
    n = len(closed)
    out = {"n": n, "enough": False, "z": 0.0, "fwd_avg": None, "bt_avg": bt.get("avg_ret"),
           "status": f"判断不能（あと{max(MIN_FWD - n, 0)}件）"}
    if n == 0:
        return out
    out["fwd_avg"] = float(closed["ret"].mean())
    if n < MIN_FWD or bt.get("avg_ret") is None:
        return out
    sd = float(closed["ret"].std(ddof=1))
    z = (out["fwd_avg"] - bt["avg_ret"]) / (sd / math.sqrt(n)) if sd > 0 else 0.0
    out.update({"enough": True, "z": round(z, 2),
                "status": "期待を大きく下回る" if z < -2 else ("期待を上回る" if z > 2 else "想定の範囲内")})
    return out


def review(j: dict, sid: str, label: str, n: int, s: dict, bt: dict, hl: dict, ds: str) -> None:
    """フォワードの決済が30件・60件・100件…に達するたび、振り返りを学習ノートに1回だけ書く。"""
    done = j.setdefault("fwd_milestones", {})
    reached = max((m for m in MILESTONES if n >= m), default=0)
    if reached <= done.get(sid, 0):
        return
    done[sid] = reached
    journal.add(j, "監視", f"{label}: フォワードテスト{reached}件の振り返り",
                f"前向きの成績は 勝率{s['win_rate']:.0%}・平均損益{s['avg_ret']:+.2%}・市場平均との差{s['avg_excess']:+.2%}。"
                f"バックテストの期待は 勝率{bt.get('win_rate', 0):.0%}・平均損益{bt.get('avg_ret', 0):+.2%}。判定: {hl['status']}。",
                result=hl["status"], date=ds)


def run() -> None:
    cfg = load_config()
    cap, lot, cost = cfg["capital"], cfg["lot"], cfg["cost_per_side"]
    slot = cap / cfg["max_positions"]
    os.makedirs(DOCS, exist_ok=True)

    prices = data.load("daily")
    # 取引時間中（15:45より前）に実行した場合、当日の値は未確定なので使わない
    now = pd.Timestamp.now(tz="Asia/Tokyo")
    if now.hour * 60 + now.minute < 15 * 60 + 45:
        prices = prices[prices["date"] < pd.Timestamp(now.date())]
    p = Panel(prices)
    today = p.dates[-1]
    ds = today.strftime("%Y-%m-%d")
    meta = load_all_listed().set_index("code")
    base = p.tradable(max_price=cap / lot, min_turnover=cfg["min_turnover"])
    regime_up = bool(p.regime(200).iloc[-1])
    atr_now = p.atr(14).iloc[-1]
    close_now = p.c.iloc[-1]

    ledger = _load_json(LEDGER, {"start": ds, "signals": []})
    seen = {(s["date"], s["code"], s["strategy"]) for s in ledger["signals"]}

    def info(code):
        m = meta.loc[code] if code in meta.index else None
        return {"code": code, "name": m["name"] if m is not None else code,
                "market": m["market"] if m is not None else "", "sector": m["sector"] if m is not None else ""}

    strategies_out = []
    entries = [(s, True) for s in cfg.get("adopted", [])] + [(s, False) for s in cfg.get("watch", [])]
    j = journal.load()
    val = {s.get("id", s["name"]): s for s in _load_json(os.path.join(DOCS, "validation.json"), {"strategies": []})["strategies"]}
    diag = {s["id"]: s for s in _load_json(os.path.join(DOCS, "diagnosis.json"), {"strategies": []})["strategies"]}
    moves = []   # 昇格・降格（ループの後でまとめて設定に反映）
    for spec, adopted in entries:
        name, params = spec["name"], spec["params"]
        sid = sid_of(spec)                    # 台帳や画面で戦略を区別する名前
        filters = spec.get("filters") or []
        label = spec.get("label") or LABELS[name]
        sig, score, ex = build_signal(p, base, name, params, filters)

        # --- 明日の買い候補（今日の終値で条件を満たした銘柄をスコア順に）
        row = sig.iloc[-1]
        codes = row.index[row.to_numpy()]
        cands = sorted(codes, key=lambda c: -float(score.iloc[-1][c]) if np.isfinite(score.iloc[-1][c]) else 0)
        cand_out = []
        for rank, code in enumerate(cands, 1):
            c, a = float(close_now[code]), float(atr_now[code])
            key = (ds, code, sid)
            if key not in seen:
                ledger["signals"].append({"date": ds, "code": code, "strategy": sid, "rank": rank,
                                          "close": _r(c), "adopted": adopted})
                seen.add(key)
            if rank <= TOP_N:
                shares = int(slot // (c * lot)) * lot
                cand_out.append({**info(code), "rank": rank, "close": _r(c), "atr": _r(a, 2),
                                 "stop": _r(c - ex.stop_atr * a), "stop_pct": _r(-ex.stop_atr * a / c * 100),
                                 "shares": shares, "amount": round(shares * c),
                                 "affordable": shares > 0, "score": _r(float(score.iloc[-1][code]), 2),
                                 "vol_ratio": _r(float(p.v.iloc[-1][code] / p.vol_avg(20).iloc[-1][code]), 1),
                                 "ret1": _r(float(p.ret(1).iloc[-1][code]) * 100)})

        # --- フォワードテスト: 台帳に載ったシグナルだけをルール通りに売買したと仮定
        tr = simulate(p, sig, score, ex, cost=cost, include_open=True)
        led = {(s["date"], s["code"]): s for s in ledger["signals"] if s["strategy"] == sid}
        tr["key"] = list(zip(tr["signal_date"].dt.strftime("%Y-%m-%d"), tr["code"]))
        fwd = tr[tr["key"].isin(led.keys())].copy()
        fwd["rank"] = [led[k]["rank"] for k in fwd["key"]]
        closed = fwd[fwd["reason"] != "open"]
        top = closed[closed["rank"] <= cfg["max_positions"]]

        def trade_row(t):
            cur = float(close_now[t.code])
            is_open = t.reason == "open"
            return {**info(t.code), "rank": int(t.rank), "signal_date": t.signal_date.strftime("%Y-%m-%d"),
                    "entry_date": t.entry_date.strftime("%Y-%m-%d"), "entry": _r(t.entry),
                    "exit_date": None if is_open else t.exit_date.strftime("%Y-%m-%d"),
                    "exit": None if is_open else _r(t.exit),
                    "ret": _r((cur / t.entry - 1) * 100 if is_open else t.ret * 100, 2),
                    "hold": int(t.hold), "reason": "保有中" if is_open else t.reason,
                    "close": _r(cur), "stop": _r(t.stop_now) if is_open else None,
                    "sell_tomorrow": t.pending if is_open and t.pending else ""}

        open_rows = [trade_row(t) for t in fwd[fwd["reason"] == "open"].itertuples()]
        closed_rows = [trade_row(t) for t in closed.sort_values("exit_date", ascending=False).head(200).itertuples()]
        s_all, s_top = stats(closed), stats(top)

        # --- 監視: フォワードの成績がバックテストの期待と合っているか
        bt = val.get(sid, {}).get("all", {})
        hl = health(closed, bt)
        passed_bt = bool(spec.get("passed_backtest"))
        if adopted and hl["enough"] and (s_all["avg_ret"] <= 0 or hl["z"] < -2):
            moves.append((spec, "demote", hl))
        elif (not adopted and passed_bt and not spec.get("demoted") and hl["enough"]
              and s_all["avg_ret"] > 0 and s_all["avg_excess"] > 0 and hl["z"] >= -2):
            moves.append((spec, "promote", hl))
        stage = "採用中" if adopted else ("フォワード確認中" if passed_bt and not spec.get("demoted") else "検証中")

        # --- フォワード結果の原因分析（30件たまったら特徴ごとにも分解する）
        fdiag = {"reasons": diagnose.by_reason(closed) if len(closed) else [], "buckets": None, "findings": []}
        edges = diag.get(name, {}).get("edges")
        if len(closed) >= MIN_FWD and edges:
            fc = features.at_trades(p, closed)
            fdiag["buckets"] = diagnose.buckets(fc, edges)
            fdiag["findings"] = diagnose.findings(label, s_all, fdiag["reasons"], {}, None)
        review(j, sid, label, len(closed), s_all, bt, hl, ds)

        strategies_out.append({
            "id": sid, "name": name, "label": label, "adopted": adopted, "stage": stage, "params": params,
            "filters_text": [features.describe(f) for f in filters], "health": hl, "diag": fdiag,
            "exit": {"stop_atr": ex.stop_atr, "trail_atr": ex.trail_atr, "target_atr": ex.target_atr,
                     "exit_above_sma": ex.exit_above_sma, "exit_below_sma": ex.exit_below_sma,
                     "max_hold": ex.max_hold},
            "regime_required": bool(params.get("regime", False)),
            "n_candidates": len(cands), "candidates": cand_out,
            "open": open_rows, "closed": closed_rows,
            "forward": {"all": {k: v for k, v in s_all.items() if k != "yearly"},
                        "top": {k: v for k, v in s_top.items() if k != "yearly"}},
        })
        print(f"[{sid}] {stage} 候補 {len(cands)} / 仮想保有 {len(open_rows)} / 決済済み {len(closed)} / {hl['status']}",
              flush=True)

    # --- 昇格・降格を設定ファイルと学習ノートに反映（画面への反映は次回の実行から）
    for spec, kind, hl in moves:
        label = spec.get("label") or LABELS[spec["name"]]
        if kind == "promote":
            cfg["watch"].remove(spec)
            cfg["adopted"].append(spec)
            spec["promoted"] = ds
            journal.add(j, "監視", f"{label} を採用に昇格",
                        f"バックテスト合格に加え、フォワードテスト{hl['n']}件で平均損益{hl['fwd_avg']:+.2%}・"
                        "市場平均との差もプラスを確認した。", result="採用", date=ds)
        else:
            cfg["adopted"].remove(spec)
            cfg["watch"].append(spec)
            spec["demoted"], spec["demoted_by"] = ds, "forward"
            journal.add(j, "監視", f"{label} を採用から外した",
                        f"フォワードテスト{hl['n']}件の平均損益が{hl['fwd_avg']:+.2%}で、バックテストの期待"
                        f"{hl['bt_avg']:+.2%}を大きく下回った（ずれの大きさ z={hl['z']:.1f}）。",
                        ["過去データで合格したルールでも、実際の相場で通用しなくなることがある。原因を調べ直すまで買いシグナルは参考扱い。"],
                        result="降格", date=ds)
        print(f"[{sid_of(spec)}] {'昇格' if kind == 'promote' else '降格'}", flush=True)
    if moves:
        save_config(cfg)
    journal.save(j)
    _dump(LEDGER, ledger)

    b = p.c[BENCH].dropna()
    _dump(os.path.join(DOCS, "today.json"), {
        "date": ds, "generated": pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d %H:%M"),
        "forward_start": ledger["start"], "capital": cap, "max_positions": cfg["max_positions"],
        "slot": slot, "lot": lot,
        "regime": {"up": regime_up, "topix_etf": _r(float(b.iloc[-1])),
                   "sma200": _r(float(b.rolling(200).mean().iloc[-1])),
                   "ret20": _r(float(b.iloc[-1] / b.iloc[-21] - 1) * 100)},
        "n_tradable": int(base.iloc[-1].sum()), "strategies": strategies_out,
    })

    # --- 保有株の売り判定用: 銘柄ごとの直近の終値・ATR・5日移動平均
    tail = p.c.iloc[-QUOTE_DAYS:]
    sma5 = p.sma(5).iloc[-1]
    quotes = {}
    for code in p.codes:
        if code == BENCH or not np.isfinite(close_now[code]):
            continue
        s = tail[code]
        quotes[code] = {"n": info(code)["name"], "c": [_r(x) for x in s.to_numpy()],
                        "a": _r(float(atr_now[code]), 2), "s5": _r(float(sma5[code]))}
    _dump(os.path.join(DOCS, "quotes.json"),
          {"dates": [d.strftime("%Y-%m-%d") for d in tail.index], "q": quotes})
    print(f"{ds} 分を書き出しました（地合い: {'上昇基調' if regime_up else '下落基調'}）")


if __name__ == "__main__":
    run()
