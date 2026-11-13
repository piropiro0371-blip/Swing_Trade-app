"""
原因分析: トレードの成績を「どんな状態のときだったか」で分解し、勝ち負けの原因を言葉にする。

- 特徴ごとに、値が低い／中間／高いの3グループに分けて成績を比べる
- 売った理由（損切り・期限など）ごとの内訳を出す
- 統計的にはっきりした偏り（t値の絶対値が2以上）だけを「気づき」として文章にする
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import stats
from .features import FEATURES, fmt, round_sig

MIN_N = 200   # これ未満のグループは偶然の影響が大きいので気づきにしない
KEEP = ("n", "win_rate", "avg_ret", "avg_excess", "t_excess", "profit_factor", "avg_win", "avg_loss", "avg_hold")


def slim(s: dict) -> dict:
    return {k: s[k] for k in KEEP if k in s} if s.get("n") else {"n": 0}


def edges_from(t: pd.DataFrame) -> dict:
    """各特徴を3等分する境目（下から1/3と2/3の位置）"""
    out = {}
    for name in FEATURES:
        v = t[f"f_{name}"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(v) < 30:
            continue
        a, b = round_sig(v.quantile(1 / 3)), round_sig(v.quantile(2 / 3))
        if a < b:
            out[name] = [a, b]
    return out


def split(t: pd.DataFrame, name: str, e: list[float]):
    """(呼び名, lo, hi, そのグループのトレード) を3つ返す"""
    v = t[f"f_{name}"]
    return [
        (f"低い（{fmt(name, e[0])} 未満）", None, e[0], t[v < e[0]]),
        (f"中間（{fmt(name, e[0])}〜{fmt(name, e[1])}）", e[0], e[1], t[(v >= e[0]) & (v < e[1])]),
        (f"高い（{fmt(name, e[1])} 以上）", e[1], None, t[v >= e[1]]),
    ]


def buckets(t: pd.DataFrame, edges: dict) -> dict:
    out = {}
    for name, e in edges.items():
        out[name] = [{"label": lab, "lo": lo, "hi": hi, **slim(stats(g))} for lab, lo, hi, g in split(t, name, e)]
    return out


def by_reason(t: pd.DataFrame) -> list[dict]:
    rows = []
    for reason, g in t.groupby("reason"):
        rows.append({"reason": reason, "n": int(len(g)), "share": float(len(g) / len(t)),
                     "avg_ret": float(g["ret"].mean()), "win_rate": float((g["ret"] > 0).mean()),
                     "avg_hold": float(g["hold"].mean())})
    return sorted(rows, key=lambda r: -r["n"])


def findings(label: str, s_all: dict, reasons: list[dict], bk_is: dict, bk_oos: dict | None) -> list[str]:
    """成績の原因を、数字つきの短い文にする。"""
    out = []
    if not s_all.get("n"):
        return out
    w, lo = s_all["avg_win"], s_all["avg_loss"]
    wr = s_all["win_rate"]
    need = -lo / (w - lo) if (w - lo) > 0 else 1.0   # 損益がトントンになる勝率
    out.append(
        f"勝率{wr:.0%}・勝ち1回の平均{w:+.1%}・負け1回の平均{lo:+.1%}。この勝ち負けの幅だと損益がゼロになる勝率は{need:.0%}で、"
        + ("実際の勝率がそれを下回っているため、コスト込みで負けている。" if wr < need else "実際の勝率はそれを上回っている。"))
    if abs(s_all["avg_ret"] - s_all["avg_excess"]) > 0.001:
        out.append(f"1取引の平均損益{s_all['avg_ret']:+.2%}のうち、市場全体の上げ下げによる分が"
                   f"{s_all['avg_ret'] - s_all['avg_excess']:+.2%}、銘柄選びによる分（市場平均との差）が{s_all['avg_excess']:+.2%}。")
    stop = [r for r in reasons if r["reason"].startswith("損切り")]
    if stop:
        sh = sum(r["share"] for r in stop)
        avg = sum(r["avg_ret"] * r["n"] for r in stop) / sum(r["n"] for r in stop)
        out.append(f"損切りで終わった取引が全体の{sh:.0%}（平均{avg:+.1%}）。")
    gap = next((r for r in reasons if r["reason"] == "損切り(窓)"), None)
    if gap and gap["share"] >= 0.03:
        out.append(f"そのうち寄り付きで逆指値を飛び越えて約定したもの（窓開け）が{gap['share']:.0%}あり、想定より大きい損失（平均{gap['avg_ret']:+.1%}）になっている。")
    for name, rows in bk_is.items():
        for k, r in enumerate(rows):
            if r.get("n", 0) < MIN_N or abs(r["t_excess"]) < 2:
                continue
            good = r["avg_excess"] > 0
            txt = (f"「{FEATURES[name][0]}」が{r['label']}のとき、市場平均との差は{r['avg_excess']:+.2%}"
                   f"（学習期間 {r['n']}件・t値{r['t_excess']:+.1f}）で{'良い' if good else '悪い'}。")
            if bk_oos and name in bk_oos and bk_oos[name][k].get("n", 0) >= 50:
                o = bk_oos[name][k]
                same = (o["avg_excess"] > 0) == good
                txt += f" 検証期間は{o['avg_excess']:+.2%}で{'同じ向き' if same else '逆向き（当てにならない）'}。"
            out.append(txt)
    return out
