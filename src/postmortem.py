"""
フォワードテストの失敗の逐次分析（ポストモーテム＝終わった取引の事後検証）。

毎日の判定（src/daily.py）から呼ばれ、新しく決済された「負けトレード」を1件ずつ分析して
docs/data/postmortem.json に追記する（追記のみ。過去の分析は書き換えない）。

 1件ごと  何が原因で負けたかを機械的に分類する
          （地合い／銘柄固有／窓開け／買うタイミング／売り方／値動きが出ない）
          さらに、バックテストの原因分析で「成績が悪い」と分かっていた状態に当たっていたかを調べる
 傾向     負けが MIN_LOSSES 件たまったら、負けやすい状態の偏りを調べる。
          偏りが統計的に明確（Z_PATTERN 以上）なら、研究リスト（hypotheses.json）に仮説として登録する
 学習へ   登録した仮説は、週1回の学習ループ（src/learn.py）が学習期間のデータだけで予選・本選にかける

守ること: フォワードの成績を見てルールを直接直さない。直すとフォワードテストが「後から合わせた成績」になり、
         ルールが通用するかを確かめる材料ではなくなる。傾向は仮説として登録するだけで、採否は5基準で決める。
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd

from . import features
from .universe import ROOT

PATH = os.path.join(ROOT, "docs", "data", "postmortem.json")
HYP = os.path.join(ROOT, "docs", "data", "hypotheses.json")
MIN_LOSSES = 10      # 傾向を調べ始める負けの件数
MIN_BUCKET = 8       # 偏りを判定する区分の最低件数
Z_PATTERN = 3.0      # 偏りの判定ライン（15特徴×2区分を一度に調べるので、2.0では偶然の偏りが出やすい）

CAUSES = {
    "gap":     "窓開けで逆指値を飛び越えた",
    "market":  "市場全体の下げに巻き込まれた",
    "stock":   "銘柄固有の下げ（市場平均より大きく負けた）",
    "timing":  "買った直後から一度も買値を上回らなかった",
    "exit":    "一時は含み益があったのに負けで終わった",
    "stall":   "値動きが出ないまま保有期限で売った",
}


def load() -> dict:
    if os.path.exists(PATH):
        with open(PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"cases": [], "strategies": {}}


def save(pm: dict) -> None:
    from .daily import clean_json
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(clean_json(pm), f, ensure_ascii=False, indent=1, allow_nan=False)


def _key(sid: str, t) -> str:
    return f"{sid}|{t.code}|{pd.Timestamp(t.signal_date):%Y-%m-%d}"


def _bucket(v: float, edge: list) -> int | None:
    if v is None or not np.isfinite(v):
        return None
    a, b = edge
    return 0 if v < a else (1 if v < b else 2)


def classify(t, mfe: float, mae: float) -> list[str]:
    """1件の負けの原因（複数当てはまることがある）。数値の目安はコスト往復0.2%を含めて決めた。"""
    out = []
    if t.reason == "損切り(窓)":
        out.append("gap")
    if t.mkt <= -0.03 and t.excess > -0.02:
        out.append("market")
    elif t.excess <= -0.02:
        out.append("stock")
    if mfe <= 0.005:
        out.append("timing")
    elif mfe >= max(0.02, t.risk * 0.5):
        out.append("exit")
    if t.reason == "期限":
        out.append("stall")
    return out or ["stock"]


def weak_spots(row: pd.Series, diag: dict | None) -> list[str]:
    """バックテストの原因分析で「3区分の中で最も成績が悪い」と分かっていた状態に当たっていたか。"""
    if not diag or not diag.get("edges"):
        return []
    out = []
    for name, edge in diag["edges"].items():
        bk = diag.get("buckets_is", {}).get(name)
        k = _bucket(row.get(f"f_{name}"), edge)
        if not bk or k is None or len(bk) != 3:
            continue
        ex = [b.get("avg_excess") if b.get("n") else None for b in bk]
        if None in ex:
            continue
        if ex[k] == min(ex) and max(ex) - ex[k] >= 0.005:   # 最良の区分より0.5%以上悪い
            out.append((max(ex) - ex[k], f"{features.FEATURES[name][0]}が「{bk[k]['label']}」"
                        f"（バックテストでは市場平均との差 {ex[k]:+.2%}、最も良い区分は {max(ex):+.2%}）"))
    return [x for _, x in sorted(out, key=lambda r: -r[0])[:3]]   # 差の大きい3つまで


def _z(k1: int, n1: int, k2: int, n2: int) -> float:
    """2つの割合（負け率）の差の大きさ。2つの区分の負け率が同じなら0付近になる。"""
    if min(n1, n2) == 0:
        return 0.0
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    return 0.0 if se == 0 else (k1 / n1 - k2 / n2) / se


def patterns(fc: pd.DataFrame, diag: dict | None) -> list[dict]:
    """決済済みの全トレードから、負け率が明確に高い状態（低い側・高い側）を探す。"""
    lose = (fc["ret"] < 0).to_numpy()
    out = []
    for name in features.FEATURES:
        v = fc[f"f_{name}"].to_numpy(dtype="float64")
        ok = np.isfinite(v)
        if ok.sum() < 2 * MIN_BUCKET:
            continue
        edge = (diag or {}).get("edges", {}).get(name) or list(np.quantile(v[ok], [1 / 3, 2 / 3]))
        for side, m, flt, kind in (("低い", v < edge[0], {"feature": name, "lo": edge[0], "hi": None}, "not-low"),
                                   ("高い", v >= edge[1], {"feature": name, "lo": None, "hi": edge[1]}, "not-high")):
            m = m & ok
            rest = ok & ~m
            n1, n2 = int(m.sum()), int(rest.sum())
            if n1 < MIN_BUCKET:
                continue
            k1, k2 = int(lose[m].sum()), int(lose[rest].sum())
            z = _z(k1, n1, k2, n2)
            out.append({"feature": name, "side": side, "kind": kind, "filter": flt,
                        "text": f"{features.FEATURES[name][0]}が{side}とき",
                        "n": n1, "loss_rate": k1 / n1, "rest_loss_rate": k2 / max(n2, 1), "z": round(z, 2),
                        "clear": z >= Z_PATTERN})
    return sorted(out, key=lambda r: -r["z"])[:8]


def _load_hyp() -> dict:
    with open(HYP, encoding="utf-8") as f:
        return json.load(f)


def _register(hy: dict, sid: str, name: str, label: str, pat: dict, forward_only: bool, ds: str) -> str | None:
    """偏りを研究リストに仮説として登録する（同じものは二重に登録しない）。登録したらIDを返す。"""
    src = f"{sid}|{pat['feature']}|{pat['kind']}"
    if any(x.get("source_key") == src for x in hy["items"]):
        return None
    nums = [int(x["id"][1:]) for x in hy["items"] if str(x.get("id", "")).startswith("H") and x["id"][1:].isdigit()]
    hid = f"H{max(nums, default=0) + 1:03d}"
    hy["items"].append({
        "id": hid, "title": f"{label}: {pat['text']}は買わない",
        "idea": (f"フォワードテストで、{pat['text']}の負け率が{pat['loss_rate']:.0%}（{pat['n']}件）と、"
                 f"それ以外の{pat['rest_loss_rate']:.0%}より明確に高かった（偏りの大きさ z={pat['z']}）。"
                 "フォワードの件数は少なく偶然の可能性があるため、10年分の学習期間で同じ傾向が出るかを確かめる。"),
        "data_needed": "なし（手元の株価データで検証できる）" if not forward_only else
                       "過去の各時点の決算データ（J-Quants Standardプラン）。この戦略は10年分の検証ができない",
        "status": "データ待ち" if forward_only else "未検証",
        "added": ds, "source": "フォワード失敗分析", "source_key": src,
        "strategy": name, "filter": pat["filter"],
    })
    return hid


def run(p, sid: str, name: str, label: str, closed: pd.DataFrame, diag: dict | None,
        forward_only: bool, j: dict, ds: str, pm: dict) -> dict:
    """1つの戦略の決済済みフォワードトレードを分析し、新しい負けを pm に追記する。画面用の要約を返す。"""
    from . import journal
    done = {c["key"] for c in pm["cases"]}
    fc = features.at_trades(p, closed) if len(closed) else closed
    new = []
    for t in fc.itertuples():
        if t.ret >= 0:
            continue
        k = _key(sid, t)
        if k in done:
            continue
        jj = p.codes.get_loc(t.code)
        hi = p.h.iloc[t.e:t.x + 1, jj].to_numpy(dtype="float64")
        lo = p.l.iloc[t.e:t.x + 1, jj].to_numpy(dtype="float64")
        mfe = float(np.nanmax(hi) / t.entry - 1) if np.isfinite(hi).any() else 0.0   # 保有中の最大の含み益
        mae = float(np.nanmin(lo) / t.entry - 1) if np.isfinite(lo).any() else 0.0   # 保有中の最大の含み損
        row = fc.loc[t.Index]
        causes = classify(t, mfe, mae)
        case = {
            "key": k, "strategy": sid, "label": label, "code": t.code, "analyzed": ds,
            "signal_date": f"{pd.Timestamp(t.signal_date):%Y-%m-%d}", "exit_date": f"{pd.Timestamp(t.exit_date):%Y-%m-%d}",
            "ret": round(float(t.ret), 4), "mkt": round(float(t.mkt), 4), "excess": round(float(t.excess), 4),
            "reason": t.reason, "hold": int(t.hold), "mfe": round(mfe, 4), "mae": round(mae, 4),
            "causes": causes, "cause_text": [CAUSES[c] for c in causes],
            "weak_spots": weak_spots(row, diag),
            "features": {n: features.fmt(n, row.get(f"f_{n}")) for n in features.FEATURES
                         if np.isfinite(row.get(f"f_{n}", np.nan))},
            "note": "",   # forward-analyst（Claude）が開示・ニュースなどを調べて書き足す欄
        }
        pm["cases"].append(case)
        done.add(k)
        new.append(case)

    mine = [c for c in pm["cases"] if c["strategy"] == sid]
    counts = {c: sum(c in x["causes"] for x in mine) for c in CAUSES}
    summary = {"label": label, "losses": len(mine), "closed": int(len(closed)),
               "causes": [{"id": c, "text": CAUSES[c], "n": n} for c, n in sorted(counts.items(), key=lambda kv: -kv[1]) if n],
               "known_weak": sum(bool(x["weak_spots"]) for x in mine), "patterns": [], "registered": []}

    if len(mine) >= MIN_LOSSES and len(fc):
        summary["patterns"] = patterns(fc, diag)
        clear = [x for x in summary["patterns"] if x["clear"]]
        if clear:
            hy = _load_hyp()
            for pat in clear[:2]:   # 1回に登録するのは2件まで
                hid = _register(hy, sid, name, label, pat, forward_only, ds)
                if hid:
                    summary["registered"].append(hid)
                    journal.add(j, "監視", f"フォワードの負けに偏り: {label}（{hid}として研究リストに登録）",
                                f"{pat['text']}の負け率 {pat['loss_rate']:.0%}（{pat['n']}件）、それ以外 "
                                f"{pat['rest_loss_rate']:.0%}。ルールは変えず、学習期間で同じ傾向が出るかを次の学習ループで確かめる。",
                                result="仮説登録", date=ds)
            with open(HYP, "w", encoding="utf-8") as f:
                json.dump(hy, f, ensure_ascii=False, indent=1)

    # 負けが10件たまるごとに、原因の内訳を学習ノートに残す
    prev = pm["strategies"].get(sid, {}).get("losses", 0)
    if len(mine) // MIN_LOSSES > prev // MIN_LOSSES:
        top = "、".join(f"{x['text']} {x['n']}件" for x in summary["causes"][:3])
        journal.add(j, "監視", f"フォワードの負け{len(mine)}件の原因分析: {label}",
                    f"決済{len(closed)}件中、負け{len(mine)}件。主な原因は {top}。"
                    f"バックテストで分かっていた弱い状態に当たっていた負けは {summary['known_weak']}件。",
                    date=ds)
    pm["strategies"][sid] = summary
    if new:
        print(f"[{sid}] 負けトレード {len(new)}件を分析（累計 {len(mine)}件）", flush=True)
    return summary
