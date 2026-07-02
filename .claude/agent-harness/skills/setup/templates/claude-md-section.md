## Multi-agent collaboration

This repository is developed by a **five-agent Claude Code crew** running as
independent sessions with **no shared memory**; they coordinate **only through git**.
Load (or reload, in any fresh window) an agent's persona with its slash command:
`/einstein`, `/archimedes`, `/euclid`, `/gauss`, `/newton`.

- **Einstein** — conductor: scopes future work into issues, delegates & sequences,
  and keeps everyone in check (process + scope). Writes no feature code; never
  merges.
- **Archimedes / Euclid / Gauss** — peer builders: plan → approval → issue → branch
  → draft PR. Take Einstein-delegated work or self-claim a groomed issue.
- **Newton** — independent skeptic reviewer/verifier. Builds nothing, trusts no
  agent's claims, and re-verifies every PR himself against the issue/plan/spec and
  the CI logs.

Mechanics:

- **Per-agent logs:** each agent appends only to its own
  `coordination/log/<agent>.md` (one writer per file → no merge conflicts). Before
  claiming/starting, `git fetch origin main` and read all logs + open issues/PRs;
  record claims, delegations, completions, blockers, and handoffs there.
- **Coordination PRs are small and separate:** they touch only your own log file
  (branch `<agent>/coord-<topic>`), never code; feature PRs never touch
  `coordination/`.
- **Branches:** `<agent>/<issue#>-<slug>` (e.g. `euclid/17-add-export-pipeline`).
- **Review gate:** a feature/code PR is merged by the human **only after Newton
  posts a review containing "approved."** Coordination/doc-only PRs are exempt. No
  agent has override authority; Einstein enforces process and escalates to the
  human.

Full protocol and entry format: **`coordination/README.md`**. The crew is packaged
as a reusable plugin (`the-girls`) for other projects — see its README / run
`/the-girls:setup` to scaffold another repo.
