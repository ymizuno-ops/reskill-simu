#!/usr/bin/env bash
# sub-test.md の agent-scoped PreToolUse フック (matcher: "Edit|Write"):
# 「テスト生成担当」というsub-testの役割を、対象パスの面でも技術的に縛る。
#
# tools: だけではEdit/WriteのON/OFFしか制御できず、「どこに書き込むか」までは絞れない。
# このフックはそれを1枚重ねて、テストファイルらしくないパスへのEdit/Writeを拒否する。
#
# 判定基準（一般的な慣習に基づくヒューリスティック。全言語・全構成を網羅するものではない）:
#   test/ tests/ __tests__/ spec/ ディレクトリ配下、*.test.*、*.spec.*、
#   test_*.py（pytest慣習）、*_test.{go,py,rb}（Go/Ruby慣習）
#
# 捕捉できない: Bash経由のリダイレクト書き込み（例: echo ... > src/app.js）。
# sub-test.md は tools: で Bash を制限していないため、これは素通りする。
# 真に強制したい場合は sub-test.md の tools: から Bash を外すか、別途 Bash 用の
# 同様のフックを追加すること（このリポジトリの他のフックと同じ「完全ではない一枚の層」の方針）。

input=$(cat)

if command -v python3 >/dev/null 2>&1; then
  path=$(printf '%s' "$input" | python3 -c 'import sys, json
try:
    print(json.load(sys.stdin).get("tool_input", {}).get("file_path", ""))
except Exception:
    pass' 2>/dev/null)
else
  path=$(printf '%s' "$input" | tr "\n" " " \
        | grep -oE '"file_path"[[:space:]]*:[[:space:]]*"([^"\\]|\\.)*"' | head -1 \
        | sed -E 's/^"file_path"[[:space:]]*:[[:space:]]*"//; s/"$//')
fi

# パスを取れなかった場合は fail-open（通常運用をブロックしない）
[ -z "$path" ] && exit 0

if printf '%s' "$path" | grep -qE -- \
  '(^|/)(test|tests|__tests__|spec)(/|$)|\.(test|spec)\.[^/]+$|(^|/)test_[^/]+\.py$|(^|/)[^/]+_test\.(go|py|rb)$'; then
  exit 0
fi

printf 'BLOCKED: sub-testエージェントはテストファイル以外を編集できません: %s\n' "$path" >&2
printf '  -> 実装コードの修正が必要な場合は、担当のビルダーエージェントかメインエージェントに差し戻してください。\n' >&2
exit 2
