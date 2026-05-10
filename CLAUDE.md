# Project Core Settings

## AI Behavior Rules
1. **作業開始の儀式**: 新しいタスクを始める前に、必ず `.ai/Instructions.md` と `.ai/Memory.md` を読み込むこと。
2. **規約の遵守**: 実装・リファクタリングを行う際は、必ず `.ai/CodingStandards.md` を厳守すること。
3. **ローカル設定の優先**: プロジェクトルートに `CLAUDE.local.md` が存在する場合、その内容を最優先の環境設定として扱うこと。
4. **トークン節約**: 無闇なディレクトリスキャンを避け、必要なファイルのみをピンポイントで読み込み、必要な差分のみを出力すること。

## Commands
プロジェクトの構成ファイル（`package.json`, `pyproject.toml`, `Cargo.toml` 等）から、適切なパッケージマネージャーやテストランナーを自律的に判断して実行すること。