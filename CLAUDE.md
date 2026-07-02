# AGENT DIRECTIVE — nemopy

Every rule is mandatory. Ambiguity is a signal to halt, not to improvise.

## 1. SOURCE OF TRUTH

The spec lives in `DESIGN.md` (§§1–10) and `DESIGN_APPENDICES.md` (§§11–16, Appendices A–B). Both carry equal authority; "DESIGN.md" below means both collectively. Signatures, return types, error semantics, and class hierarchies in its code blocks are **immutable**. Silence in the spec means "unspecified" — see §5.3. DESIGN.md wins over all other files.

## 2. SCOPE LOCK

Scope = the task brief. Nothing else.

**MUST NOT:** add/remove/modify anything not in the brief; add helpers or wrappers not in DESIGN.md; change dependencies or config files; create unspecified files/modules; touch adjacent code, formatting, or imports outside scope; add speculative error handling, type hints, logging, or instrumentation beyond the spec; refactor anything not broken and not named in the brief.

**MUST:** match existing code style; remove imports/variables your changes made unused; leave pre-existing dead code untouched (mention in PR description if noted).

**Scope test** — apply to every changed line before commit: *"Does this line trace directly to the task brief?"* If no, revert it.

## 3. BRANCH ISOLATION

One task = one branch = one PR. Branch from the specified base. Name: `feat/`, `fix/`, or `test/<name>`. No multi-task commits. If a pre-existing blocker is found mid-task, STOP and report — do not fix it here.

## 4. EXECUTION PROTOCOL

**Workflow (strict order, no deviation):**
1. Receive brief → 2. Read DESIGN.md → 3. State plan (§5) → 4. Create branch (§3) → 5. Write tests FIRST, commit (§6) → 6. Verify tests FAIL → 7. Implement minimum code → 8. Full suite passes → 9. Cleanup audit (§9) → 10. Scope test (§2) → 11. Document your changes only (§8) → 12. Commit → 13. Open PR → 14. Report completion

**Commits:** one logical change each. Format: `<type>(<scope>): <imperative summary>`. Types: feat, fix, test, refactor, docs, chore. No amend/squash/force-push unless instructed.

**Completion report:** output `TASK COMPLETE. PR #[N] opened against [base]. All tests passing. Cleanup audit passed.` Then stop — no suggestions, no extra work, no merge.

## 5. THINK BEFORE CODING

Before any code, produce:
```
## Plan
- Assumptions: [list]
- Files touched: [list]
- Tests to write: [each with goal per §6.1]
- Steps: 1. [Step] → verify: [how]
```

Ask: *"Would a senior engineer call this overcomplicated?"* If yes, simplify first.

**§5.2 No silent decisions:** multiple valid interpretations → present alternatives with tradeoffs, wait for decision.

**§5.3 Ambiguity protocol:** STOP. Name the ambiguity precisely, quote the text, present interpretations and consequences, wait. "I went with the most reasonable option" is a violation.

## 6. TEST INTEGRITY

### 6.1 Goal-first tests
Before writing a test, state its **goal** (plain language), **source** (DESIGN.md §), and **expected** result. Every test must trace to DESIGN.md. No goal = delete it.

### 6.2 Red-green (mandatory)
Write test → commit → verify it **FAILS** → implement → verify it **PASSES**. Skipping the fail step is a violation. If a new test passes before implementation, the test is defective — rewrite it.

### 6.3 When a test fails after implementation
The **code is wrong** until proven otherwise. Fix code first. Only modify the test if you can demonstrate its stated goal contradicts DESIGN.md — document why in the commit message.

### 6.4 No gaming the test suite
Violations: modifying a test to pass failing code; weakening assertions; adding special-case implementation logic for test values; tautological tests; deleting/skipping tests you didn't write without authorisation.

### 6.5 Test economy
Minimum tests to verify the goal. One per distinct behaviour. Use `@pytest.mark.parametrize` over copy-paste. No tests outside task scope. If >10 tests, justify each.

### 6.6 Existing tests are immutable
Do not modify, delete, rename, or skip tests you didn't write. If they fail from your changes, your code is wrong. If genuinely incorrect, STOP and report with evidence (quote goal + DESIGN.md).

## 7. DEBUGGING DISCIPLINE

Nothing added during debugging survives to commit: no print/log statements, no commented-out code, no temp files, no widened interfaces. After debugging, `git diff` every hunk — if a line isn't part of the fix, revert it.

## 8. DOCUMENTATION

Document **only code you wrote/modified**. Docstrings for public API (match existing style). Inline comments only for non-obvious logic. PR description: what you implemented, which DESIGN.md section, assumptions. Do not touch module docs, README, or anything outside scope.

## 9. CLEANUP AUDIT (before every PR)

- [ ] No print/log/debug output statements
- [ ] No commented-out code
- [ ] No temp/scratch files
- [ ] No unrelated formatting or whitespace changes in diff
- [ ] Every test has a stated goal and is non-redundant
- [ ] No debug-only code (widened interfaces, exposed internals)
- [ ] All added imports used; removed imports were made unused by your changes only
- [ ] Diff contains ONLY task-required changes

## 10. PROHIBITIONS (always forbidden)

Merging PRs · pushing to main/protected branches · changing dependencies · modifying CI/CD · creating unspecified modules/classes/functions · altering DESIGN.md signatures · adding unspecified features/params · reformatting outside scope · post-completion suggestions · interpreting spec silence as permission (§5.3) · modifying tests to pass failing code (§6.4) · deleting/skipping others' tests (§6.6) · committing debug artefacts (§7) · documenting untouched code (§8) · multi-task branches (§3)

## 11. PRECEDENCE

This document wins over task briefs unless the brief explicitly overrides a numbered section by reference (e.g., "Override §10.3: you may add a dependency for X"). General phrasing ("do whatever is needed") is not an override.
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

## PROJECT CONTEXT

The crew personas are project-agnostic; this block is the only project-specific
context they read on every session.

- **Direction / mandate:** nemopy is a Python linear algebra library with Rust kernels;
  goal is to implement the full DESIGN.md spec.
- **Off-limits / sensitive files:** `DESIGN.md`, `DESIGN_APPENDICES.md` (immutable spec),
  `pyproject.toml` — no edits without explicit human sign-off.
- **Build / test / lint commands:** `pytest`, `cargo test`, `maturin develop`.
- **Conventions:** match CLAUDE.md §§1–11 strictly; Rust kernels in `src/`, Python
  bindings in `nemopy/`.
