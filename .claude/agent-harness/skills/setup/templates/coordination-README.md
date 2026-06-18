# Multi-agent coordination

This repository is developed by a **five-agent Claude Code crew** running as
independent sessions with **no shared memory**. Their only shared channel is this
git repository, so coordination is deliberate and asynchronous: an agent writes to
its log, opens a small PR, the human merges it, and the others see it on their next
`git fetch`.

## The crew

| Agent | Role | Writes code? | Merges? |
|-------|------|:---:|:---:|
| **Einstein** | Conductor — scopes future work, delegates & sequences, keeps everyone in check (process + scope) | no | no |
| **Archimedes / Euclid / Gauss** | Peer builders — plan, implement, open PRs | yes | no |
| **Newton** | Independent skeptic reviewer/verifier — the merge gate | no | no |

Only the **human** merges. Load (or reload, in any fresh window) an agent's persona
with its slash command: `/einstein`, `/archimedes`, `/euclid`, `/gauss`, `/newton`.

## The one rule that makes this work

You can only see what is on `main`. So **before claiming or starting anything**,
`git fetch origin main` and read all the logs + open issues/PRs; **after a
meaningful step**, record it. Treat the logs as a shared, published record — not a
private scratchpad.

## Channel: per-agent logs

Each agent appends **only** to its own `coordination/log/<agent>.md`. One writer per
file means the logs are conflict-free. To "reply" to another agent, add an entry in
**your own** log quoting their entry's timestamp. The full picture is all five logs
read together.

### Posting a coordination note
1. `git fetch origin main`; branch from it:
   `git switch -c <agent>/coord-<topic> origin/main`.
2. **Append** one or a few entries to the **end** of your own log (never edit or
   reorder earlier entries; never touch another agent's log).
3. Open a **small PR** that touches **only** your own log file. Title:
   `coord: <topic>`.
4. The human merges it; others see it on their next `git fetch origin main`.

Batch your notes at natural checkpoints (claiming, delegating, finishing a PR,
hitting a blocker, handing off) — not line by line. **Coordination PRs never include
code; feature PRs never touch `coordination/`.**

### Entry format
```
### 2026-01-15T14:32Z — Euclid — claim #17
Claiming #17 (add the export pipeline). Branch euclid/17-add-export-pipeline.
Touches src/export/*.ts and the CLI entrypoint.
Gauss: please hold #19's CLI refactor until this lands. ETA: today.
```
- Heading: `### <UTC ISO-8601 to the minute> — <Agent> — <topic>`.
- Topics: `delegate #n -> <Agent>`, `claim #n`, `done #n (PR #m)`, `blocked #n`,
  `handoff #n -> <Agent>`, `approved #n (PR #m)`, `changes-requested #n (PR #m)`,
  `note`.
- Body: 1–5 lines. Name the issue#, branch, files touched, any dependency/ordering,
  and an ETA or ask. Be explicit about file overlaps.

## Work assignment & claiming

- **Einstein** grooms the backlog and **delegates** issues (recorded in his log plus
  an issue comment). Builders take delegated work first.
- Otherwise a builder **self-claims** a groomed issue: post a `claim #n` entry naming
  the files it will touch **and** comment on the issue. If an open claim already
  covers those files, coordinate order in the logs before starting.
- Respect dependency ordering. Prefer new/disjoint files so you never collide.
- No agent has override authority. If two claims genuinely overlap, resolve it in the
  logs; Einstein proposes the sequencing and escalates to the human if needed.

## Review gate (Newton)

A **feature/code PR is merged by the human only after Newton posts an approving
review whose body contains the word "approved."** If Newton requests changes, the
author fixes them on the same branch and re-requests review. Newton verifies
independently and trusts no agent's claims (diff vs issue scope vs spec vs CI logs).

**Coordination / doc-only PRs are exempt** from the gate — the human merges them
directly.

## Binding workflow (see `CLAUDE.md`)

Plan → get the human's approval → create/observe a GitHub **issue** → work on **one
branch** `<agent>/<issue#>-<slug>` → open a **draft PR** referencing the issue →
**never commit to `main`**. One issue → one branch → one PR. The human merges all
PRs.

## Reusing this crew on another project

The crew is packaged as a reusable Claude Code plugin (`the-girls`). Add it as a
marketplace and install it, then run the `setup` skill (`/the-girls:setup`) to
scaffold a new repository (`coordination/`, the `.claude/settings.json` permissions
allowlist, the per-repo SessionStart hook, optional `/einstein`-style commands, and a
`CLAUDE.md` section).
