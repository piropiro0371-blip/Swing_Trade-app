@echo off
rem ============================================================
rem  スイングトレード判定：このPCでの毎日の更新
rem  ・割安株（低PBR）の新しい候補は、決算データがあるこのPCでしか出せません
rem  ・大引け後（15時45分以降。おすすめは19時頃）に実行してください
rem  ・結果は logs\local_update.log に記録されます
rem ============================================================
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
echo ===== %date% %time% 開始 ===== >> logs\local_update.log

echo [1/7] GitHubの最新状態を取り込みます...
git pull --rebase >> logs\local_update.log 2>&1
if errorlevel 1 goto :error

echo [2/7] 株価を取得します（数分かかります）...
python -m src.data --period 2y --tag daily >> logs\local_update.log 2>&1
if errorlevel 1 goto :error

echo [3/7] 決算データの追加分を取得します...
python -m src.jquants >> logs\local_update.log 2>&1
if errorlevel 1 goto :error

echo [4/7] PBRの元データを作り直します...
python -m src.fundamentals >> logs\local_update.log 2>&1
if errorlevel 1 goto :error

echo [5/7] 株式分割の履歴を更新します...
python -m src.splits >> logs\local_update.log 2>&1
if errorlevel 1 goto :error

echo [6/7] 判定とフォワードテストを更新します...
python -m src.daily >> logs\local_update.log 2>&1
if errorlevel 1 goto :error

echo [7/7] 結果をGitHubへ送信します...
git add docs/data config >> logs\local_update.log 2>&1
git commit -m "local update (value candidates)" >> logs\local_update.log 2>&1
git push >> logs\local_update.log 2>&1
if errorlevel 1 goto :error

echo ===== %date% %time% 完了 ===== >> logs\local_update.log
echo.
echo 完了しました。数分後に公開画面へ反映されます。
goto :end

:error
echo ===== %date% %time% 失敗 ===== >> logs\local_update.log
echo.
echo 途中で失敗しました。logs\local_update.log の末尾を確認するか、Claudeに「毎日の更新が失敗した」と伝えてください。

:end
if "%1"=="" pause