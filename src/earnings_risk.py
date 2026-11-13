"""
大負けと開示（決算・業績修正）の関係を調べる。

問い: 少数の大負けが成績を決めている。その大負けは、保有中に決算や業績修正が出たことが原因か？
      もしそうなら「保有中に決算をまたぐ取引を避ける」だけで成績が改善するはず。

調べ方:
- 各トレードの保有期間（買った日〜売った日）に、開示の「反応日」が含まれるかを調べる
- 「開示をまたいだ取引」と「またがなかった取引」で、損益と大負けの割合を比べる
- 決算の予定日は通常2週間以上前に公表されるので、「またぐかどうか」は買う前に分かる情報とみなす
  （業績修正は予告なく出るので、事前には避けられない。区別して集計する）

使い方:  python -m src.earnings_risk      （fins.parquet が必要。Freeプランは約1年9か月分）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data
from .engine import Panel, simulate
from .events import load_events
from .research import load_config
from .strategies import LABELS, build_signal


def run() -> list[dict]:
    cfg = load_config()
    p = Panel(data.load("bt"))
    base = p.tradable(cfg["capital"] / cfg["lot"], cfg["min_turnover"])
    ev = load_events(p)
    lo, hi = ev["r"].min(), ev["r"].max()
    print(f"開示データの期間: {p.dates[lo].date()}〜{p.dates[hi].date()}（{len(ev):,}件）")
    # 銘柄ごとの反応日（行番号）の一覧
    by = {k: {c: np.sort(g["r"].to_numpy()) for c, g in ev[ev["kind"] == k].groupby("code")} for k in ("決算", "修正")}

    def crosses(kind, code, a, b):
        arr = by[kind].get(code)
        if arr is None:
            return False
        i = np.searchsorted(arr, a)
        return i < len(arr) and arr[i] <= b

    out = []
    for spec in cfg["adopted"] + cfg["watch"]:
        name, params = spec["name"], spec["params"]
        sig, score, ex = build_signal(p, base, name, params, spec.get("filters"))
        t = simulate(p, sig, score, ex, cost=cfg["cost_per_side"])
        t = t[(t["e"] >= lo) & (t["x"] <= hi)].copy()      # 開示データがある期間の取引だけ
        if len(t) < 50:
            continue
        t["決算"] = [crosses("決算", c, a, b) for c, a, b in zip(t["code"], t["e"], t["x"])]
        t["修正"] = [crosses("修正", c, a, b) for c, a, b in zip(t["code"], t["e"], t["x"])]
        cut = t["ret"].quantile(0.05)
        big = t["ret"] <= cut
        print(f"\n===== {spec.get('label') or LABELS[name]}  対象 {len(t)}件 / 下位5%の境目 {cut:+.1%}")
        rows = []
        for lab, m in (("全取引", pd.Series(True, index=t.index)),
                       ("決算をまたいだ", t["決算"]), ("決算をまたがなかった", ~t["決算"]),
                       ("業績修正が出た", t["修正"]), ("どちらも無し", ~t["決算"] & ~t["修正"])):
            g = t[m]
            if len(g) == 0:
                continue
            rows.append({"区分": lab, "件数": len(g), "割合": len(g) / len(t), "勝率": (g["ret"] > 0).mean(),
                         "平均損益": g["ret"].mean(), "市場平均との差": g["excess"].mean(),
                         "大負けの割合": big[m].mean(), "最悪": g["ret"].min(), "損益の合計": g["ret"].sum()})
        df = pd.DataFrame(rows)
        pc, pc2 = "{:.0%}".format, "{:+.2%}".format
        print(df.to_string(index=False, formatters={"割合": pc, "勝率": pc, "平均損益": pc2, "市場平均との差": pc2,
                                                    "大負けの割合": "{:.1%}".format, "最悪": "{:+.0%}".format,
                                                    "損益の合計": "{:+.2f}".format}))
        share = (big & t["決算"]).sum() / max(big.sum(), 1)
        print(f"大負け{int(big.sum())}件のうち、決算をまたいでいたもの {share:.0%}"
              f"（全取引で決算をまたぐ割合は {t['決算'].mean():.0%}）")
        out.append({"id": spec.get("id", name), "label": spec.get("label") or LABELS[name], "table": rows,
                    "big_share_earnings": float(share), "base_share_earnings": float(t["決算"].mean()), "n": int(len(t))})
    return out


if __name__ == "__main__":
    run()
