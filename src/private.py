"""
非公開データ（J-Quants由来の分析結果）の保管。

J-Quantsの規約では、データそのものの配布に加えて「このデータで分析した結果を継続的に第三者へ提供・配信すること」が
禁止されている。割安株（低PBR）の候補・台帳・仮想売買の結果は J-Quants の決算データから作るため、公開ページには
平文で載せず、合言葉（SWING_PASSPHRASE）で暗号化した docs/data/private.enc にだけ書く。
画面は、合言葉を入れた端末だけがブラウザの中で復号して表示する（合言葉は外部に送らない）。

- 合言葉: 環境変数 SWING_PASSPHRASE（GitHubのSecrets）または .env の SWING_PASSPHRASE=…
- 暗号: PBKDF2-HMAC-SHA256（310,000回）で鍵を作り、AES-256-GCMで暗号化（ブラウザの Web Crypto で復号できる形式）
- PCでは data/private/bundle.json にも平文の控えを置く（Git管理外）。合言葉が無いときはこちらを使う
"""

from __future__ import annotations

import base64
import json
import os

from .universe import ROOT

ENC = os.path.join(ROOT, "docs", "data", "private.enc")
LOCAL = os.path.join(ROOT, "data", "private", "bundle.json")
ITER = 310_000


def passphrase() -> str | None:
    k = os.environ.get("SWING_PASSPHRASE", "")
    path = os.path.join(ROOT, ".env")
    if not k and os.path.exists(path):
        for line in open(path, encoding="utf-8-sig"):
            if line.startswith("SWING_PASSPHRASE="):
                k = line.split("=", 1)[1].strip()
    return k or None


def _key(pw: str, salt: bytes) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    return PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITER).derive(pw.encode("utf-8"))


def encrypt(obj: dict, pw: str) -> dict:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    salt, iv = os.urandom(16), os.urandom(12)
    ct = AESGCM(_key(pw, salt)).encrypt(iv, json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), None)
    b = lambda x: base64.b64encode(x).decode("ascii")
    return {"v": 1, "kdf": "PBKDF2-SHA256", "iter": ITER, "salt": b(salt), "iv": b(iv), "ct": b(ct)}


def decrypt(env: dict, pw: str) -> dict:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    d = lambda k: base64.b64decode(env[k])
    return json.loads(AESGCM(_key(pw, d("salt"))).decrypt(d("iv"), d("ct"), None))


def load() -> dict | None:
    """非公開データを読む。合言葉と暗号化ファイルがあれば復号、無ければPCの控え。どちらも無ければ None。"""
    pw = passphrase()
    if pw and os.path.exists(ENC):
        with open(ENC, encoding="utf-8") as f:
            return decrypt(json.load(f), pw)
    if os.path.exists(LOCAL):
        with open(LOCAL, encoding="utf-8") as f:
            return json.load(f)
    return None


def save(bundle: dict) -> bool:
    """PCの控えを書き、合言葉があれば暗号化ファイルも書く。暗号化できたら True。"""
    os.makedirs(os.path.dirname(LOCAL), exist_ok=True)
    with open(LOCAL, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False)
    pw = passphrase()
    if not pw:
        return False
    with open(ENC, "w", encoding="utf-8") as f:
        json.dump(encrypt(bundle, pw), f)
    return True
