## 2026-10-05 23:21 (refactor/directory-layout)
- catboost_info/ と src/catboost_info/ を git の追跡から外した（ファイルは残し、.gitignore 済み）
- CatBoost の出力先を tmp/catboost_info に変更（src/step3_train.py）
- main.py・simulation.py・occupation.py・ui/ を src/ へ移動。main.py のパス基準を修正し、起動コマンドを streamlit run src/main.py に変更
- README.md・docs/design.md のディレクトリ構成と起動コマンドを更新
- CLAUDE.md の「ディレクトリの例外」に models/ を追記
