#!/bin/bash
# クラウドのClaude Code（claude.ai/code）でセッションを始めたときに自動で実行される準備。
# .claude/settings.json の SessionStart フックから呼ばれる。PCでは何もしない。
#
# - Pythonの依存パッケージを入れる（入っていれば飛ばす）
# - 株価データは重いので自動では取らない。必要になったら CLAUDE.md の「クラウドでの開発」の手順で取る
set -u
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/..}" || exit 0
if python3 -c "import yfinance, pandas, numpy, pyarrow, requests, openpyxl" 2>/dev/null; then
  echo "依存パッケージは導入済み"
else
  python3 -m pip install -q -r requirements.txt || echo "依存パッケージの導入に失敗（ネットワーク設定を確認）"
fi
mkdir -p data/cache
exit 0
