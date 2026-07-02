#!/usr/bin/env bash
# SessionStart hook for the multi-agent crew. Read-only and non-blocking: prints a
# reminder and injects protocol context into the model. Performs NO git/network/
# filesystem writes itself; always exits 0 so it can never block session start.
set -uo pipefail

echo "[crew] Load your persona: /einstein /archimedes /euclid /gauss /newton. Then: git fetch origin main; read coordination/log/*.md; check open issues/PRs." 1>&2

reminder='MULTI-AGENT CREW. You are one of five independent sessions that coordinate ONLY through git: conductor EINSTEIN; peer BUILDERS Archimedes/Euclid/Gauss; skeptic REVIEWER Newton. Run your slash command (/einstein, /archimedes, /euclid, /gauss, /newton) to load your full persona. Before any work this session:
1. git fetch origin main, then read EVERY coordination/log/*.md (each agent appends only to its own file): Einstein delegations, builder claims, blockers, handoffs, Newton verdicts.
2. Check open GitHub issues + PRs so you do not collide or claim taken work.
3. Builders: plan -> human approval -> GitHub issue -> one issue / one branch (<agent>/<issue#>-<slug>) / one draft PR -> never commit to main. Record claims/progress only in your own coordination/log/<agent>.md via SMALL log-only PRs (<agent>/coord-<topic>), never mixed with code.
4. GATE: a feature PR is merged by the human ONLY after Newton reviews it and comments "approved". Newton trusts no claim and re-verifies everything himself (diff vs issue scope vs spec vs CI logs). Einstein scopes/delegates/keeps-in-check but never merges or writes feature code.
Full protocol: coordination/README.md.'

if command -v jq >/dev/null 2>&1; then
  jq -n --arg ctx "$reminder" \
    '{hookSpecificOutput: {hookEventName: "SessionStart", additionalContext: $ctx}}'
elif command -v python3 >/dev/null 2>&1; then
  # python3 fallback: handles all JSON escaping portably (no GNU-sed dependency).
  python3 -c "
import json, sys
ctx = sys.argv[1]
print(json.dumps({'hookSpecificOutput': {'hookEventName': 'SessionStart', 'additionalContext': ctx}}))
" "$reminder"
else
  # Last-resort fallback: escape backslash, quote, newline manually via printf.
  # This is portable across bash versions; the only unhandled case is a literal
  # carriage-return in $reminder, which this reminder string does not contain.
  esc=$(printf '%s' "$reminder" \
    | tr '\\' '@BACKSLASH@' \
    | tr '"' '@QUOTE@' \
    | tr '\n' '@NEWLINE@' \
    | sed 's/@BACKSLASH@/\\\\/g;s/@QUOTE@/\\"/g;s/@NEWLINE@/\\n/g')
  printf '{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"%s"}}\n' "$esc"
fi

exit 0
