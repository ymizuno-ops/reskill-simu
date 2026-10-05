#!/usr/bin/env bash
# PreToolUse フック (matcher: Bash): 破壊的な git 操作と、秘密情報ファイルのコミットをブロックする。
#
# permissions.deny の Bash パターンはフラグ順の入れ替え・短縮形・`sh -c` / `eval`
# でくぐり抜けられる。このフックは生のコマンド文字列を検査して1枚重ねる層。
#   捕捉できる:   フラグ順の入れ替え、`sh -c '...'`、`eval '...'`、クォート、
#                 カレントブランチが main/master のときの裸の `git push`、
#                 コミット対象に .env系/.pem/.key/secrets.json/secrets/ が含まれる場合
#   捕捉できない: 変数展開 (g=git; $g ...)、文字列の再構築、`git -c ... <破壊操作>`
#                 真の強制が必要なら sandbox を使う。
#
# ヒアドキュメント本文（`cat <<'EOF' ... EOF` 等）はマスクしてから判定する（python3経路のみ）。
# 理由: 改行を `; ` に変換する下の処理と組み合わさると、ヒアドキュメント内に書いた
# 単なる文章・コード例（例: PR本文に例として書いた `git push` の一文）まで
# 「コマンド位置」として誤検知していた（実例: PR #17 作成時）。
# 副作用として、`bash <<'EOF' ... EOF` のように標準入力を実際にシェルへ渡す形は
# 本文が実行対象でもマスクされ捕捉できなくなる（既知のトレードオフ）。
#
# 誤検知したときは、内容を確認のうえ別ターミナルで手動実行してください。

input=$(cat)

# tool_input.command を取り出す（python3 優先、無ければ簡易 grep フォールバック）。
# python3経路はヒアドキュメント本文もマスクする（後述）。
if command -v python3 >/dev/null 2>&1; then
  cmd=$(printf '%s' "$input" | python3 -c 'import sys, json, re

def mask_heredocs(s):
    out = []
    i = 0
    n = len(s)
    heredoc_re = re.compile(r"<<(-)?\s*([\x27\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")
    while i < n:
        m = heredoc_re.match(s, i)
        if m:
            out.append(s[i:m.end()])
            i = m.end()
            dash = m.group(1)
            delim = m.group(3)
            nl = s.find("\n", i)
            if nl == -1:
                break
            out.append(s[i:nl+1])
            i = nl + 1
            while i < n:
                line_end = s.find("\n", i)
                line = s[i:line_end] if line_end != -1 else s[i:]
                check = line.lstrip("\t") if dash else line
                if check == delim:
                    out.append(line)
                    if line_end != -1:
                        out.append("\n")
                        i = line_end + 1
                    else:
                        i = n
                    break
                else:
                    out.append(" " * len(line))
                    if line_end != -1:
                        out.append(" ")
                        i = line_end + 1
                    else:
                        i = n
            continue
        out.append(s[i])
        i += 1
    return "".join(out)

try:
    c = json.load(sys.stdin).get("tool_input", {}).get("command", "")
    print(mask_heredocs(c))
except Exception:
    pass' 2>/dev/null)
else
  # フォールバック経路はヒアドキュメント本文をマスクしない（既知の制約）
  cmd=$(printf '%s' "$input" | tr "\n" " " \
        | grep -oE '"command"[[:space:]]*:[[:space:]]*"([^"\\]|\\.)*"' | head -1 \
        | sed -E 's/^"command"[[:space:]]*:[[:space:]]*"//; s/"$//')
fi

# コマンドを取れなかった場合は fail-open（通常運用をブロックしない）
[ -z "$cmd" ] && exit 0

# 改行はコマンド区切りとして扱う
cmd="${cmd//$'\n'/ ; }"

# `git [opts] <subcommand>` が「コマンド位置」に現れるか。
# コマンド位置 = 行頭 / ; & | ( の直後 / `sh -c "` `bash -c '` `eval "` の直後。
# これで `echo git push` や `git commit -m "...git push..."`（引数内の git push）は拾わない。
gitcmd() {
  printf '%s' "$cmd" | grep -qE -- \
    "(^|[;&|(]|(^|[[:space:]])(sh|bash|zsh|dash)[[:space:]]+-c[[:space:]]+[\"']|(^|[[:space:]])eval[[:space:]]+[\"'])[[:space:]]*git[[:space:]]+(-[^[:space:]]+[[:space:]]+)*($1)\b"
}


# `git commit` の引数（commit の直後から、次のコマンドの区切りまで）に、パターンに合う引数があるか。
# 同じ行の別コマンドの引数（grep -n 等）を拾わないよう commit の引数だけを見る。
# -m / --message / -F の値（コミットメッセージ等）は読み飛ばす（引用符で囲まれた複数語も1つの値として扱う）。
# sh -c '...' のように全体が引用符で囲まれた形も捕捉するため、引用符はトークンの前後から外して判定する。
commit_args_match() {
  local w core seen=0 skip=0 q=""
  local -a toks
  read -ra toks <<< "$cmd"
  for w in "${toks[@]}"; do
    if [ -n "$q" ]; then                       # 引用符で始まった値の途中。閉じる引用符まで読み飛ばす
      case "$w" in *"$q") q="" ;; esac
      continue
    fi
    if [ "$skip" = 1 ]; then                   # -m 等の直後の値
      skip=0
      case "$w" in
        \"*\"|\'*\') ;;
        \"*) q='"' ;;
        \'*) q="'" ;;
      esac
      continue
    fi
    core="${w#[\"\']}"; core="${core%[\"\']}"   # 前後の引用符を1個ずつ外す
    core="${core%%[;&|)]*}"                     # 末尾にくっついた区切り（-n; 等）を外す
    if [ "$seen" = 0 ]; then
      [ "$core" = "commit" ] && seen=1
    else
      case "$w" in
        "&&"|"||"|";"|"|"|"&") seen=0; continue ;;
        -m|--message|-F|--file) skip=1; continue ;;
      esac
      printf '%s' "$core" | grep -qE -- "$1" && return 0
    fi
    case "$w" in *[\;\&\|\)]) seen=0 ;; esac    # 区切りがくっついていたら、ここでコマンドが切れる
  done
  return 1
}

deny() {
  printf 'BLOCKED: %s\n' "$1" >&2
  printf '  -> %s\n' "$2" >&2
  exit 2
}

# `git push` に「別ブランチを明示する引数」が無い＝カレントブランチを push する、なら true。
# 裸の `git push` / `git push <remote>` / `git push -u <remote>` / `git push <remote> HEAD` を検出する。
push_targets_current_branch() {
  local flat="${cmd//$'\n'/ }"
  local -a toks
  read -ra toks <<< "$flat"
  local w seen=0 npos=0 last=""
  for w in "${toks[@]}"; do
    w="${w#[\"\']}"; w="${w%[\"\']}"          # 前後のクォートを1個外す
    if [ "$seen" = 0 ]; then
      [ "$w" = "push" ] && seen=1
      continue
    fi
    case "$w" in
      "&&"|"||"|";"|"|"|"&") break ;;         # ここでコマンドが切れる
      -*) continue ;;                          # フラグ
    esac
    npos=$((npos + 1))
    last="$w"
  done
  [ "$seen" = 1 ] || return 1
  [ "$npos" -le 1 ] && return 0                # remote だけ / 何も無し → カレント
  case "$last" in HEAD|@|HEAD:*|@:*) return 0 ;; *) return 1 ;; esac
}

# `git push` の引数（push 直後から chain区切りまでの範囲）に main/master 参照が
# あるか（origin/main, HEAD:main, :main 等）。push の外側にある main（例: 直前の
# `git checkout main`）を拾わないよう、push_targets_current_branch と同じ範囲に限定する。
push_args_have_main_master() {
  local flat="${cmd//$'\n'/ }"
  local -a toks
  read -ra toks <<< "$flat"
  local w seen=0
  for w in "${toks[@]}"; do
    w="${w#[\"\']}"; w="${w%[\"\']}"
    if [ "$seen" = 0 ]; then
      [ "$w" = "push" ] && seen=1
      continue
    fi
    case "$w" in
      "&&"|"||"|";"|"|"|"&") break ;;
      -*) continue ;;
    esac
    printf '%s' "$w" | grep -qE -- '(^|[/:])(main|master)([:]|$)' && return 0
  done
  return 1
}

# `git <サブコマンド>` の引数（サブコマンドの直後から、次のコマンドの区切りまで）に、パターンに合う引数があるか。
# 同じ行の別コマンドの引数（rm -f / sed -n / echo --hard 等）を拾わないよう、そのサブコマンドの引数だけを見る。
# 区切りで範囲が終わっても走査を続け、後ろに出てくる同じサブコマンドも見る（`echo push && git push -f` 等）。
# sh -c '...' のように全体が引用符で囲まれた形も捕捉するため、引用符はトークンの前後から外して判定する。
git_args_match() {
  local sub="$1" re="$2" w core seen=0
  local -a toks
  read -ra toks <<< "$cmd"
  for w in "${toks[@]}"; do
    core="${w#[\"\']}"; core="${core%[\"\']}"   # 前後の引用符を1個ずつ外す
    core="${core%%[;&|)]*}"                     # 末尾にくっついた区切り（-f; 等）を外す
    if [ "$seen" = 0 ]; then
      [ "$core" = "$sub" ] && seen=1
    else
      case "$w" in "&&"|"||"|";"|"|"|"&") seen=0; continue ;; esac
      printf '%s' "$core" | grep -qE -- "$re" && return 0
    fi
    case "$w" in *[\;\&\|\)]) seen=0 ;; esac    # 区切りがくっついていたら、ここでコマンドが切れる
  done
  return 1
}

current_branch() {
  git -C "${CLAUDE_PROJECT_DIR:-.}" rev-parse --abbrev-ref HEAD 2>/dev/null || true
}

# --- git push の保護 ---
if gitcmd push; then
  # (a) push の引数に main / master が明示されている（origin/main, HEAD:main, :main 等も）
  if push_args_have_main_master; then
    deny "main/master への push は禁止です。" \
         "ブランチを切って push してください: git switch -c <branch> && git push -u origin <branch>"
  fi
  # (b) 別ブランチ指定なし＝カレントを push。カレントが main/master なら禁止（裸の push 対策）
  if push_targets_current_branch; then
    cur=$(current_branch)
    if [ "$cur" = "main" ] || [ "$cur" = "master" ]; then
      deny "現在 '$cur' ブランチにいます。'$cur' への直接 push は禁止です。" \
           "git switch -c <branch> でブランチを切ってから push してください。"
    fi
  fi
  # (c) force push（--force-with-lease は許可）
  if git_args_match push '^(--force|-[a-zA-Z]*f[a-zA-Z]*)$'; then   # -uf のようにまとめた短いフラグも拾う
    deny "force push は禁止です。" \
         "巻き戻すなら --force-with-lease を使うか、新しいブランチに push してください。"
  fi
fi

# --- git reset --hard（未コミットの変更が失われる）---
if gitcmd reset && git_args_match reset '^--hard$'; then
  deny "git reset --hard は禁止です（未コミットの変更が失われます）。" \
       "git stash で退避するか、変更を確認してから個別に戻してください。"
fi

# --- git clean（--dry-run / -n 以外。未追跡ファイルが失われる）---
if gitcmd clean && ! git_args_match clean '^(--dry-run|-[a-zA-Z]*n[a-zA-Z]*)$'; then
  deny "git clean は禁止です（未追跡ファイルが失われます）。" \
       "git clean -n で対象を確認し、必要なら手動で削除してください。"
fi

# --- git commit --no-verify / -n（pre-commit フックの迂回）---
if gitcmd commit && commit_args_match '^(--no-verify|-[a-zA-Z]*n[a-zA-Z]*)$'; then
  deny "git commit --no-verify (-n) は禁止です（フックを飛ばします）。" \
       "フックが要求するチェックを先に通してからコミットしてください。"
fi

# --- git commit に秘密情報らしいファイルが含まれていないか ---
if gitcmd commit; then
  staged=$(git -C "${CLAUDE_PROJECT_DIR:-.}" diff --cached --name-only 2>/dev/null || true)
  # -a / --all はワーキングツリーの追跡済み変更もその場でコミット対象に含めるため合わせて見る
  # `-am`のような結合短縮フラグも拾う（commit の引数だけを見る。--no-verify の検知と同じ方式）
  if commit_args_match '^(--all|-[a-zA-Z]*a[a-zA-Z]*)$'; then
    staged="$staged
$(git -C "${CLAUDE_PROJECT_DIR:-.}" diff --name-only 2>/dev/null || true)"
  fi
  secret_hit=$(printf '%s\n' "$staged" \
    | grep -E -- '(^|/)\.env(\.[^/]*)?$|\.(pem|key)$|(^|/)secrets\.json$|(^|/)secrets/' \
    | head -1 || true)
  if [ -n "$secret_hit" ]; then
    deny "秘密情報らしいファイルがコミット対象に含まれています: $secret_hit" \
         "git restore --staged $secret_hit で外すか、本当に必要なら内容を確認してから手動でコミットしてください。"
  fi
fi

exit 0
