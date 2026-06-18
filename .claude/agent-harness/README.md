# the-girls — a reusable 5-agent Claude Code harness

A portable crew of five Claude Code agents that collaborate on a repository through
git, with a skeptical review gate before anything merges:

- **Einstein** — conductor: scopes future work, delegates & sequences, keeps
  everyone in check (process + scope). Writes no code; never merges.
- **Archimedes / Euclid / Gauss** — peer builders: plan → approval → issue → branch
  → draft PR.
- **Newton** — independent skeptic reviewer/verifier: re-verifies every PR himself
  and is the merge gate — the human merges only after Newton comments **"approved."**

Each agent is a slash command, so you (re)load a persona in any fresh window by
typing its command — the identity lives in the repo, not in chat memory.

## What's in this plugin

```
.claude-plugin/plugin.json        # plugin manifest (name: the-girls)
.claude-plugin/marketplace.json   # single-plugin marketplace catalog (name: nemo-agents)
commands/                         # /einstein /archimedes /euclid /gauss /newton (+ legacy /agents-init pointer)
skills/setup/                     # the "setup" skill: scaffolds a repo for the crew
  SKILL.md                        #   what /the-girls:setup does
  scaffold.sh                     #   idempotent scaffolder (merges, never clobbers)
  templates/                      #   self-contained templates it lays down
hooks/session-start.sh            # the SessionStart reminder TEMPLATE (installed per repo by setup)
```

There is deliberately **no `hooks/hooks.json`**: the SessionStart hook is **opt-in
per repo**, installed into the target repo by the `setup` skill, rather than shipped
as an always-on plugin hook that would fire in every repo where the plugin is enabled.

## Install it in another project

This repo publishes a marketplace catalog at its **root** (`.claude-plugin/marketplace.json`),
so the one-liner is:

```
/plugin marketplace add nster101/beamer
/plugin install the-girls@nemo-agents
```

Installed commands and the skill are namespaced — `/the-girls:einstein`,
`/the-girls:setup`, etc. (run `/reload-plugins` if they are not picked up
immediately).

This `agent-harness/` directory is *also* a self-contained single-plugin marketplace
(it carries its own `.claude-plugin/marketplace.json` with `source: ./`), so you can
instead add it by local path, or copy it out to its own git repo and add that:

```
/plugin marketplace add ./.claude/agent-harness      # or <owner/repo> after extracting
/plugin install the-girls@nemo-agents
```

## Scaffold the new repository

A plugin **cannot** ship a permissions allowlist or per-repo state (the
`coordination/` logs, the SessionStart hook), so run the bundled `setup` skill once
per repo:

```
/the-girls:setup
```

It creates `coordination/` (README + per-agent logs), writes a `.claude/settings.json`
permissions allowlist + the hook wiring, installs the per-repo
`.claude/hooks/session-start.sh`, optionally copies plain `/einstein`-style commands
into `.claude/commands/`, and appends a "Multi-agent collaboration" section plus a
short **PROJECT CONTEXT** block to `CLAUDE.md`. The scaffolder is idempotent — it
merges and never clobbers, so it is safe to re-run.

## Customising per project

The personas are deliberately **project-agnostic**: they defer every project
specific (direction, off-limits files, build/test commands, conventions) to the
target repo's `CLAUDE.md`. To adapt the crew to a new project, edit that repo's
`CLAUDE.md` (the `PROJECT CONTEXT` block) — not the personas.

## How coordination works

- Agents have **no shared memory**; they coordinate only through git.
- Each agent appends only to its own `coordination/log/<agent>.md` (one writer per
  file → no merge conflicts). Notes are posted as **small log-only PRs** the human
  merges; feature PRs never touch `coordination/`.
- Only the **human** merges, and only after Newton's "approved" review (coordination
  / doc-only PRs are exempt).

## Notes / limitations

- Plugin `settings.json` currently supports only a narrow set of keys, so the
  permissions allowlist is intentionally written into each repo by the `setup` skill
  rather than shipped here.
- The five persona files here are the source of truth. When `setup` copies them into
  a repo's `.claude/commands/` for plain `/einstein` names, that is a deliberate copy.
- To share the crew across a team, copy this directory out to its own git repo and
  add that repo as a marketplace.
