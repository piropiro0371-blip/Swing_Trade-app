"""
学習ループ: バックテストの結果から負けの原因を探し、改善案を自動で作って検証する。

 1. 原因分析  各戦略のトレードを特徴（値動きの大きさ・売買代金・市場の状態など）で分解する
 2. 仮説づくり 「この状態のときだけ買う／この状態は避ける」という絞り込み条件を機械的に作る
 3. 予選      学習期間だけで採点し、有望な上位だけを残す（検証期間はここでは見ない）
 4. 本選      残った案を検証期間も含めて5基準で採点する。必要なt値は試した数に応じて上がる
 5. 記録      試した数・結果・分かったことを学習ノートに残す。合格した案はフォワードテストに回す

使い方:  python -m src.learn      （先に python -m src.research）
出力:    docs/data/diagnosis.json / docs/data/journal.json / validation.json に合格候補を追記
"""

from __future__ import annotations

import json
import os

import pandas as pd

from . import data, diagnose, features, journal
from .engine import Panel, simulate, stats
from .research import OUT_WEB, evaluate, load_config, report, save_config, upsert, write_validation
from .strategies import build_signal, grid
from .universe import ROOT

OUT_DIAG = os.path.join(ROOT, "docs", "data", "diagnosis.json")
FINALISTS = 3        # 戦略ごとに本選へ進める案の数（検証期間を見る回数を抑える）
PRELIM_T = 2.0       # 予選の足切り（学習期間のt値）
PRELIM_N = 300       # 予選の足切り（学習期間の取引数）

KINDS = {  # 絞り込みの型: (呼び名, どの境目を使うか)
    "only-low": "低いときだけ買う", "only-mid": "中間のときだけ買う", "only-high": "高いときだけ買う",
    "not-low": "低いときは避ける", "not-high": "高いときは避ける",
}


def make_filters(edges: dict) -> list[tuple[str, str, dict]]:
    out = []
    for name, (a, b) in edges.items():
        out += [
            (name, "only-low", {"feature": name, "lo": None, "hi": a}),
            (name, "only-mid", {"feature": name, "lo": a, "hi": b}),
            (name, "only-high", {"feature": name, "lo": b, "hi": None}),
            (name, "not-low", {"feature": name, "lo": a, "hi": None}),
            (name, "not-high", {"feature": name, "lo": None, "hi": b}),
        ]
    return out


def subset(t: pd.DataFrame, f: dict) -> pd.DataFrame:
    v = t[f"f_{f['feature']}"]
    m = v.notna()
    if f.get("lo") is not None:
        m &= v >= f["lo"]
    if f.get("hi") is not None:
        m &= v < f["hi"]
    return t[m]


def run() -> None:
    cfg = load_config()
    is_end = pd.Timestamp(cfg["in_sample_end"])
    cost = cfg["cost_per_side"]
    j = journal.load()
    with open(OUT_WEB, encoding="utf-8") as f:
        val = json.load(f)
    bases = [s for s in val["strategies"] if not s.get("filters")]
    val["strategies"] = bases

    print("データ読み込み中...", flush=True)
    p = Panel(data.load("bt"))
    base = p.tradable(max_price=cfg["capital"] / cfg["lot"], min_turnover=cfg["min_turnover"])

    diag = {"generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "features": {k: v[0] for k, v in features.FEATURES.items()}, "strategies": []}
    lessons, new_total, passed_total, finalists_total = [], 0, 0, 0

    for s in bases:
        name, params = s["name"], s["params"]
        sig, score, ex = build_signal(p, base, name, params)
        tr = features.at_trades(p, simulate(p, sig, score, ex, cost=cost))
        t_is, t_oos = tr[tr["entry_date"] <= is_end], tr[tr["entry_date"] > is_end]

        # 1. 原因分析（境目は学習期間だけで決める）
        edges = diagnose.edges_from(t_is)
        bk_is, bk_oos = diagnose.buckets(t_is, edges), diagnose.buckets(t_oos, edges)
        reasons = diagnose.by_reason(tr)
        found = diagnose.findings(s["label"], stats(tr), reasons, bk_is, bk_oos)

        # 2〜3. 仮説づくりと予選（学習期間のみ）
        cands, keys = [], []
        for feat, kind, f in make_filters(edges):
            keys.append(journal.key(name, params, [f]))
            st = stats(subset(t_is, f))
            cands.append({"feature": feat, "kind": kind, "filter": f, "text": features.describe(f),
                          "is_n": st.get("n", 0), "is_excess": st.get("avg_excess", 0.0),
                          "is_t": st.get("t_excess", 0.0), "stage": "予選落ち"})
        new_total += journal.register(j, keys)
        t_req = journal.t_required(j)
        ok = [c for c in cands if c["is_n"] >= PRELIM_N and c["is_excess"] > 0 and c["is_t"] >= PRELIM_T]
        finalists = sorted(ok, key=lambda c: -c["is_t"])[:FINALISTS]
        print(f"[{name}] 仮説 {len(cands)}件 → 予選通過 {len(ok)}件 → 本選 {len(finalists)}件 "
              f"(必要t値 {t_req})", flush=True)

        # 4. 本選（検証期間を含めて5基準で採点）
        for c in finalists:
            finalists_total += 1
            flt = [c["filter"]]
            fam = []
            for gp in grid(name):   # 設定を変えても崩れないか
                g_sig, g_score, g_ex = build_signal(p, base, name, gp, flt)
                g_all = stats(simulate(p, g_sig, g_score, g_ex, cost=cost))
                fam.append(bool(g_all["n"]) and g_all["avg_ret"] > 0)
            sid = f"{name}+{c['feature']}:{c['kind']}"
            label = f"{s['label']}（絞り込み）"
            entry, _ = evaluate(p, base, cfg, name, params, flt, sum(fam) / len(fam), t_req, sid, label)
            c.update({"stage": "本選合格" if entry["passed"] else "本選不合格", "checks": entry["checks"],
                      "oos_n": entry["out_sample"].get("n", 0),
                      "oos_excess": entry["out_sample"].get("avg_excess"),
                      "oos_t": entry["out_sample"].get("t_excess")})
            print(f"  案: {c['text']}", flush=True)
            report(entry, cfg)
            lessons.append(f"{s['label']}＋「{c['text']}」: 学習期間の差 {c['is_excess']:+.2%}(t{c['is_t']:+.1f}) → "
                           f"検証期間 {entry['out_sample'].get('avg_excess', 0):+.2%} → "
                           f"{'合格。フォワードテストで確認する' if entry['passed'] else '不合格'}")
            if entry["passed"]:
                passed_total += 1
                val["strategies"].append(entry)
                upsert(cfg, sid, name, params, flt, label, True)
        # 以前合格して今回は本選に残らなかった／落ちた絞り込み案は、合格を取り消す
        live = {f"{name}+{c['feature']}:{c['kind']}" for c in cands if c["stage"] == "本選合格"}
        for lst in ("adopted", "watch"):
            for it in list(cfg[lst]):
                if it["name"] == name and it.get("filters") and it["id"] not in live and it.get("passed_backtest"):
                    upsert(cfg, it["id"], name, it["params"], it["filters"], it["label"], False)

        diag["strategies"].append({
            "id": name, "label": s["label"], "findings": found, "reasons": reasons,
            "buckets_is": bk_is, "buckets_oos": bk_oos, "edges": edges,
            "candidates": sorted(cands, key=lambda c: -c["is_t"])[:12], "n_candidates": len(cands),
        })
        lessons += [f"{s['label']}: {x}" for x in found[:2]]

    diag["trials_total"], diag["t_required"] = journal.trials(j), journal.t_required(j)
    val["trials_total"], val["t_required"] = diag["trials_total"], diag["t_required"]
    if new_total or passed_total:
        journal.add(j, "学習", f"原因分析と絞り込み条件の探索（新たに{new_total}件の仮説を試行）",
                    f"各戦略のトレードを{len(features.FEATURES)}種類の特徴で分解し、絞り込み条件を機械的に作って"
                    f"学習期間で予選、上位{finalists_total}件だけを検証期間で本選にかけた。",
                    lessons[:10], result=f"本選 {finalists_total}件中 {passed_total}件合格")
    journal.save(j)
    save_config(cfg)
    write_validation(val)
    from .daily import clean_json
    with open(OUT_DIAG, "w", encoding="utf-8") as f:
        json.dump(clean_json(diag), f, ensure_ascii=False, allow_nan=False)
    print(f"これまでに試した仮説 {journal.trials(j)}件 / 必要t値 {journal.t_required(j)} / 今回の合格 {passed_total}件")
    print(f"保存: {OUT_DIAG}")


if __name__ == "__main__":
    run()
