"""
log_config.py
=============
バッチ処理（step1〜3）の進捗を標準出力へ出すロガーの共通設定。
"""

from __future__ import annotations
import logging
import sys

LOGGER_NAME = "reskill"
LOG_FORMAT = "%(message)s"


def getLogger(name: str) -> logging.Logger:
    """標準出力へメッセージだけを出すロガーを返す（ハンドラは初回だけ設定する）"""
    baseLogger = logging.getLogger(LOGGER_NAME)
    if not baseLogger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        baseLogger.addHandler(handler)
        baseLogger.setLevel(logging.INFO)
        baseLogger.propagate = False
    return baseLogger.getChild(name)
