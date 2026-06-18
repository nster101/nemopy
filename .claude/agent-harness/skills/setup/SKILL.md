---
name: setup
description: Scaffold the current repository for the 5-agent crew (Einstein, Archimedes, Euclid, Gauss, Newton). Creates coordination/, the permissions allowlist, a per-repo SessionStart hook, optional /einstein-style persona commands, and a CLAUDE.md section. Use when setting up a new repo to use the agent crew, or when asked to "set up this repo to use the crew".
---

Set up the **current** repository to run the five-agent crew (Einstein, Archimedes,
Euclid, Gauss, Newton). The scaffolder **merges, never clobbers**, and is safe to
re-run.

1. **Inspect & plan.** Look at what already exists (`coordination/`,
   `.claude/settings.json`, `.claude/commands/`, `CLAUDE.md`) and show the human a
   short plan of what will change before running anything.

2. **Run the scaffolder** (idempotent):
   ```
   bash "${CLAUDE_PLUGIN_ROOT}/skills/setup/scaffold.sh"
   ```
   It creates `coordination/README.md` + the five per-agent logs; creates or merges
   `.claude/settings.json` with the crew permissions allowlist + the `SessionStart`
   hook wiring (a plugin **cannot** ship a permissions allowlist, so it must live in
   the repo); installs the per-repo `.claude/hooks/session-start.sh` (`chmod +x`);
   copies the five persona commands into `.claude/commands/` so plain `/einstein`
   names work; and appends a "Multi-agent collaboration" section + a **PROJECT
   CONTEXT** stub to `CLAUDE.md` (only if absent).

3. **Fill PROJECT CONTEXT.** Open `CLAUDE.md` and complete the **PROJECT CONTEXT**
   block with this repo's direction/mandate, the files that are off-limits to edit,
   and the build/test/lint commands. Add those build/test commands to
   `.claude/settings.json` `permissions.allow` too, so the crew runs them without
   prompting. This is the **only** project-specific step — every persona is otherwise
   project-agnostic.

4. **Confirm & spin up.** Tell the human: open a session and run `/einstein`,
   `/archimedes`, `/euclid`, `/gauss`, or `/newton` (namespaced as `/the-girls:<name>`
   when using the installed plugin). The `SessionStart` hook may need `/hooks` opened
   once (or a fresh session) before it activates; slash commands work immediately.

Do **not** commit or push without the human's approval.
