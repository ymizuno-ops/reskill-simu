# Project Memory

<project_overview>
厚生労働省・内閣府・総務省の公的統計（2006〜2024年）と機械学習モデルを用いて、
リスキリング・異業種転職を行った場合の50年間の年収推移をシミュレーションするStreamlit Webアプリ。

コア機能:
- 年齢・勤続年数・実績年収・目標職種・スキル引継ぎ率を入力してパーソナライズ予測
- Ridge / ElasticNet / Custom Ridge / Random Forest / Gradient Boosting /
  LightGBM / CatBoost / XGBoost / Stacking Ensemble の9モデルで比較
- 転職初年度年収は「1段下の年齢階級」ベースで算出
- GDP成長率・CPI・リアリティ補正パラメータによるシナリオ調整
- 50年間の累積差益・投資ROI・回収月数を定量化

技術スタック: Python 3.10+, Streamlit, scikit-learn, LightGBM/CatBoost/XGBoost, pandas, matplotlib
データパイプライン: src/step1〜step3.py でraw → processed → master → models.pkl を生成
</project_overview>

<current_status>
AI駆動開発のための5ファイル構成（設定・規約・記憶の分離）のセットアップが完了。
</current_status>

<task_list>
### Done
- [x] トークン最適化と推論精度向上のためのAIコンテキストファイルの設計・配置
- [x] app.py のリファクタリング（1578行 → モジュール分割・重複排除）

### Doing
- [ ] 次の機能開発またはバグ修正

### Todo
- [ ] テストコードの実装（tests/ 配下）
- [ ] src/step3_train.py のリファクタリング（661行 → 分割検討）
</task_list>

<decisions>
- **ファイル分割ルール**: 「UI」「ロジック」「API」「型定義」を明確に分離し、1ファイル最大300行程度に収める（過度な細分化は避ける）。
- **型定義の優先**: データ構造の変更時は、必ず `src/types/` 配下の共通型定義を先に修正すること。
- セキュリティ保護のため、ローカル環境固有の設定は `CLAUDE.local.md` で管理する。
</decisions>