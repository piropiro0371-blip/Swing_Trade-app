"""
ファンダメンタル指標の元データ（1株利益・1株純資産・配当）を、J-Quantsの決算データから作る。

出力: config/fundamentals.csv（銘柄ごとに最新の値。Git管理対象）
  eps  : 今期の1株利益の会社予想（PER = 株価 ÷ eps）
  bps  : 直近の本決算時点の1株純資産（PBR = 株価 ÷ bps）
  div  : 今期の年間配当の会社予想（配当利回り = div ÷ 株価）
  asof : その銘柄の最新の開示日

注意:
- Freeプランは直近12週間の開示が含まれない。最新の決算や業績修正は反映されていない
- 株式分割があった銘柄は、分割前の1株あたりの値のままになり、PER・PBRが実際とずれる。
  明らかにおかしい値（PERが1倍未満や300倍超など）は画面側で「–」にする

使い方:  python -m src.fundamentals      （先に python -m src.jquants）
毎日の自動実行（GitHub）はAPIキーを持たないので、このファイルを読むだけ。更新はこのPCで行う。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .jquants import FINS_PATH
from .universe import CONFIG_DIR

OUT = os.path.join(CONFIG_DIR, "fundamentals.csv")


def build() -> pd.DataFrame:
    f = pd.read_parquet(FINS_PATH)
    f = f[f["Code"].str.endswith("0")].copy()
    f["code"] = f["Code"].str[:4]
    for c in ("EPS", "FEPS", "NxFEPS", "BPS", "Eq", "ShOutFY", "DivAnn", "FDivAnn"):
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f = f.sort_values(["code", "DiscDate", "DiscTime"])
    fy = f["CurPerType"] == "FY"
    # 今期の会社予想: 本決算の開示では「来期予想」が、それ以外では「今期予想」が該当する
    f["eps_f"] = np.where(fy, f["NxFEPS"], f["FEPS"])
    f["bps_v"] = np.where(f["BPS"] > 0, f["BPS"], np.where((f["Eq"] > 0) & (f["ShOutFY"] > 0), f["Eq"] / f["ShOutFY"], np.nan))
    f["div_f"] = np.where(f["FDivAnn"].notna(), f["FDivAnn"], f["DivAnn"])
    g = f.groupby("code")
    out = pd.DataFrame({
        "eps": g["eps_f"].apply(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan),
        "eps_actual": g.apply(lambda d: d.loc[d["CurPerType"] == "FY", "EPS"].dropna().iloc[-1]
                              if (d["CurPerType"] == "FY").any() and d.loc[d["CurPerType"] == "FY", "EPS"].notna().any() else np.nan),
        "bps": g["bps_v"].apply(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan),
        "div": g["div_f"].apply(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan),
        "asof": g["DiscDate"].last(),
    }).reset_index()
    return out.round({"eps": 2, "eps_actual": 2, "bps": 2, "div": 2})


def load() -> pd.DataFrame:
    if not os.path.exists(OUT):
        return pd.DataFrame(columns=["code", "eps", "eps_actual", "bps", "div", "asof"]).set_index("code")
    return pd.read_csv(OUT, dtype={"code": str}).set_index("code")


if __name__ == "__main__":
    df = build()
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"{len(df)} 銘柄を保存: {OUT}（開示日の範囲 {df['asof'].min()}〜{df['asof'].max()}）")
    print(df.describe().loc[["count", "50%"]].to_string())
