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
