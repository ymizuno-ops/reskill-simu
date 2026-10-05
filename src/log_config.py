"""
log_config.py
=============
バッチ処理（step1〜3）の進捗を標準出力へ出すロガーの共通設定。
"""

# 型ヒント（変数や引数の型の注記）を実行時に評価せず、文字列のまま扱わせる指定。`int | None` のような新しい書き方を使えるようにする。
from __future__ import annotations
# logging: Python 標準のログ出力の仕組み。print と違い、出力先・書式・重要度（INFO・WARNING など）を後から切り替えられる。
import logging
# sys: Python 本体に関する機能。ここでは標準出力（sys.stdout = ターミナルへの出力）を使うために読み込む。
import sys

# 全モジュールのロガーの親になる名前。親に設定した出力先を、子のロガー（reskill.〜）が共有する。
LOGGER_NAME = "reskill"
# 意味: 実行ログ1行の書式。%(message)s は「メッセージ本文だけ」を表す。
# 影響: "%(asctime)s %(message)s" にすると各行の先頭に日時が付く。処理結果は変わらない。
LOG_FORMAT = "%(message)s"


# 関数の定義。`name: str` は引数の型ヒント、`-> logging.Logger` は戻り値の型ヒント。
def getLogger(name: str) -> logging.Logger:
    """標準出力へメッセージだけを出すロガーを返す（ハンドラは初回だけ設定する）"""
    # 同じ名前で呼ぶと、毎回同じロガーのオブジェクトが返る（アプリ全体で1つ）。
    baseLogger = logging.getLogger(LOGGER_NAME)
    # 出力先（ハンドラ）がまだ登録されていなければ設定する。2回目以降に重複登録すると、同じ行が何度も出てしまうため。
    if not baseLogger.handlers:
        # 標準出力（ターミナル）へ書き出すハンドラを作る。
        handler = logging.StreamHandler(sys.stdout)
        # 1行の書式を設定する。
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        # 親ロガーに出力先を登録する。
        baseLogger.addHandler(handler)
        # INFO 以上（INFO・WARNING・ERROR）のメッセージを出す。DEBUG は出さない。
        baseLogger.setLevel(logging.INFO)
        # さらに上位のロガー（ルートロガー）へメッセージを渡さない。渡すと、同じ行が二重に出ることがある。
        baseLogger.propagate = False
    # 「reskill.<モジュール名>」という子ロガーを返す。出力の設定は親のものが使われる。
    return baseLogger.getChild(name)
