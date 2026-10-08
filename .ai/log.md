## 2026-10-05 23:21 (refactor/directory-layout)
- catboost_info/ と src/catboost_info/ を git の追跡から外した（ファイルは残し、.gitignore 済み）
- CatBoost の出力先を tmp/catboost_info に変更（src/step3_train.py）
- main.py・simulation.py・occupation.py・ui/ を src/ へ移動。main.py のパス基準を修正し、起動コマンドを streamlit run src/main.py に変更
- README.md・docs/design.md のディレクトリ構成と起動コマンドを更新
- CLAUDE.md の「ディレクトリの例外」に models/ を追記

## 2026-10-05 23:27 (chore/remove-unused-files)
- 使われていないファイルを削除: .ai/Instructions.md・Memory.md・CodingStandards.md、.claude/skills/review・test、docs/tasks.md
- 削除する Memory.md の Todo を README.md の「今後の課題」へ移した

## 2026-10-05 23:33 (refactor/split-step3-train)
- step3_train.py（463行）から Wrapper クラスと add_features を src/model_wrappers.py へ分離（331行に）
- step1_to_processed.py（356行）から共通部を step1_common.py、マクロ経済系3関数を step1_macro.py へ分離（249行に）
- main.py の `_HERE` を `_ROOT` に改名、`except ImportError: pass` を削除して model_wrappers から直接 import
- CatBoostWrapper.fit で tmp/catboost_info を自動作成するよう修正（tmp/ が無いと学習が失敗する不具合）
- README.md・docs/design.md の構成図を更新し、README の「今後の課題」から step3 分割を削除

## 2026-10-06 05:04 (refactor/coding-standards)
- src/ 以下の全コードにコーディング規約を適用（関数・変数を lowerCamelCase、真偽値に is 接頭辞、print を logging に置換、型ヒント追加・Any 排除、マジックナンバーの定数化）
- 共通設定 src/log_config.py と型定義 src/model_types.py を追加
- step3_train.py から各モデルの訓練処理を src/step3_models.py に分割
- simulation.py の重複していた特徴量追加処理を model_wrappers.addFeatures に統一
- 例外の握りつぶしをなくし、警告ログを出すように変更
- サイドバーの年齢の上限を 64 歳（RETIREMENT_AGE - 1）にし、65 歳で IndexError になる不具合を修正
- README・docs/design.md の関数名・ファイル構成を更新、.ai/ERRORS.md に設計漏れを記録

## 2026-10-06 05:30 (docs/nonengineer-comments)
- src/ 以下 15 ファイル・137 か所に、非エンジニア向けの「意味・影響・注意」コメントを追記（コード本体は変更なし。AST 一致を確認）
- 現在の計算で使われていない値（age_curve.csv の raise_rate、macro_params.json の forecast_*、simulate の ageCurve 引数）をコメントで明記

## 2026-10-06 06:00 (docs/code-explainer-comments)
- src/ 以下 16 ファイル・438 か所に、入門レベルの技術解説コメント（構文・ライブラリの使い方）を追記（コード本体は変更なし。AST 一致を確認）

## 2026-10-09 05:10 (chore/migrate-to-uv)
- 依存管理を requirements.txt から uv（pyproject.toml・uv.lock・.python-version）に移行
- uv init のひな形 main.py（ルート）と requirements.txt を削除
- README のセットアップ・実行手順を uv sync / uv run に書き換え、docs/requirements.md の requirements.txt への言及を修正

## 2026-10-09 05:19 (docs/errors-ingested)
- .ai/ERRORS.md の未取り込み2件を dev-wiki に取り込み、取り込み欄を更新（dev-wiki PR #5）
- dev-wiki への外部アクセスを .ai/external-access.md に記録
- PR #8 のマージ後に push したため取り込まれなかった変更を、改めて PR にした
