#!/usr/bin/env bash
# Scaffolder for the 5-agent crew "setup" skill. Idempotent: it merges and never
# clobbers existing files, so it is safe to re-run. Performs only local filesystem
# writes in the target repo — no git commits, no network.
set -uo pipefail

# --- locate the plugin (this script is <plugin>/skills/setup/scaffold.sh) ---------
PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-}"
if [ -z "$PLUGIN_ROOT" ]; then
  self_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PLUGIN_ROOT="$(cd "$self_dir/../.." && pwd)"
fi
TPL="$PLUGIN_ROOT/skills/setup/templates"

# --- target repo = current git toplevel (must be a git repo) ----------------------
REPO="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$REPO" ]; then
  echo "[setup] ERROR: not inside a git repository. cd to the target repo root first." >&2
  exit 1
fi
cd "$REPO" || { echo "[setup] cannot cd to target repo: $REPO" >&2; exit 1; }

log() { printf '[setup] %s\n' "$*"; }
log "scaffolding crew into: $REPO"
log "plugin root: $PLUGIN_ROOT"

# --- 1. coordination channel ------------------------------------------------------
mkdir -p coordination/log
if [ -f coordination/README.md ]; then
  log "kept existing coordination/README.md"
else
  cp "$TPL/coordination-README.md" coordination/README.md
  log "created coordination/README.md"
fi
for a in einstein archimedes euclid gauss newton; do
  case "$a" in
    einstein)   Name="Einstein";;
    archimedes) Name="Archimedes";;
    euclid)     Name="Euclid";;
    gauss)      Name="Gauss";;
    newton)     Name="Newton";;
  esac
  f="coordination/log/$a.md"
  if [ -f "$f" ]; then
    log "kept existing $f"
  else
    sed "s/{{AGENT}}/$Name/g" "$TPL/log-header.md" > "$f"
    log "created $f"
  fi
done

# --- 2. permissions + per-repo SessionStart hook ----------------------------------
mkdir -p .claude/hooks
if [ ! -f .claude/hooks/session-start.sh ] \
   || ! cmp -s "$PLUGIN_ROOT/hooks/session-start.sh" .claude/hooks/session-start.sh; then
  cp "$PLUGIN_ROOT/hooks/session-start.sh" .claude/hooks/session-start.sh
  log "installed .claude/hooks/session-start.sh"
else
  log "kept existing .claude/hooks/session-start.sh"
fi
chmod +x .claude/hooks/session-start.sh

if [ ! -f .claude/settings.json ]; then
  cp "$TPL/settings.json" .claude/settings.json
  log "created .claude/settings.json"
elif command -v jq >/dev/null 2>&1; then
  tmp="$(mktemp)"
  if jq -s '
        .[0] as $cur | .[1] as $crew | $cur
        | .permissions.allow  = (((($cur.permissions.allow)  // []) + (($crew.permissions.allow)  // [])) | unique)
        | .permissions.deny   = (((($cur.permissions.deny)   // []) + (($crew.permissions.deny)   // [])) | unique)
        | .hooks.SessionStart = (((($cur.hooks.SessionStart) // []) + (($crew.hooks.SessionStart) // [])) | unique)
      ' .claude/settings.json "$TPL/settings.json" > "$tmp" 2>/dev/null; then
    mv "$tmp" .claude/settings.json
    log "merged crew permissions + SessionStart hook into existing .claude/settings.json"
  else
    rm -f "$tmp"
    log "WARNING: could not merge .claude/settings.json; add the crew allowlist from $TPL/settings.json by hand"
  fi
else
  log "WARNING: .claude/settings.json exists and jq is unavailable; merge the allowlist from $TPL/settings.json by hand"
fi

# --- 3. optional plain /einstein-style persona commands ---------------------------
mkdir -p .claude/commands
for a in einstein archimedes euclid gauss newton; do
  if [ -f ".claude/commands/$a.md" ]; then
    log "kept existing .claude/commands/$a.md"
  else
    cp "$PLUGIN_ROOT/commands/$a.md" ".claude/commands/$a.md"
    log "installed .claude/commands/$a.md"
  fi
done

# --- 4. CLAUDE.md section (append only if absent) ---------------------------------
if [ -f CLAUDE.md ] && grep -q "Multi-agent collaboration" CLAUDE.md; then
  log "CLAUDE.md already has a Multi-agent collaboration section; left it untouched"
else
  {
    [ -f CLAUDE.md ] && printf '\n'
    cat "$TPL/claude-md-section.md"
    printf '\n'
    cat "$TPL/project-context-stub.md"
  } >> CLAUDE.md
  log "appended Multi-agent collaboration + PROJECT CONTEXT to CLAUDE.md"
fi

log "done."
log "Next: fill the PROJECT CONTEXT block in CLAUDE.md (direction, off-limits files, build/test commands)."
log "Spin up: open a session and run /einstein (or /archimedes, /euclid, /gauss, /newton)."

echo
echo "[setup] Repo status after scaffolding:"
git -C "$REPO" status --short
