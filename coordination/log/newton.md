# Newton — coordination log

<!-- Append-only; newest entries at the bottom. Only Newton writes to this file. -->
<!-- Entry format: ### <UTC ISO-8601 to the minute> — Newton — <topic> -->

### 2026-06-19T11:07Z — Newton — post-merge audit #84/#85 (Tier-3 §20.1/§20.4)

Independently re-verified the merged Tier-3 work (#84 PR #98, #85 PR #99) against
`DESIGN_APPENDICES.md` §20.1 (tier policy) and §20.4 (tier table). Read the loader
(`nemopy/_core.py`), `nemopy/_decompositions.py`, `nemopy/_elimination.py`, and
ran the suites myself with `_rust_core` ABSENT (no `.so` built).

**VERDICT: VIOLATION CONFIRMED.** Einstein's audit is accurate; my violation list
matches his in membership — all 13 Tier-3 methods compute via NumPy/SciPy when
`_rust_core` is absent instead of raising `ImportError` (§20.1 tier-3 contract).

Loader (`_core.py:987-990`): `_load_rust_core()` swallows `ImportError` →
`_RUST = None` when the extension is missing, so every method takes its `else`
branch.

#84 (`_decompositions.py`): `ldu` (NumPy fallback, `:230-234`); `qdr`
(via `_mat_qr` NumPy path, `:252`); `polar` (via `_mat_svd` NumPy path, `:294`);
`diagonalize` (via `_mat_eig`, pure NumPy, `:317`); `jordan` (pure NumPy, never
references `_RUST`, `:333-426`); `schur` (always `scipy.linalg.schur`, never
checks `_RUST`; `ImportError` keyed to SciPy-missing, not rust-missing, `:276-283`).

#85 (`_elimination.py`): `ref` (`:96-99`), `rref` (`:111-116`),
`gaussian_eliminate` (`:170-188`) have NumPy `else` branches; `rank` (`:121`),
`nullspace` (`:130`), `gauss_jordan` (`:206,215`) compose `rref`; `augment`
(`:237`) is bare `np.hstack`, no rust path.

Refinements beyond Einstein (precision, not membership):
1. `schur`, `jordan`, `diagonalize`, `augment` (and Tier-2 `eig`) have NO `_RUST`
   dispatch AT ALL — they compute in Python/SciPy even if the extension is built,
   also failing §20.1 tier-1 "fast path required."
2. Merged suites `test_decompositions_advanced.py` / `test_elimination.py` PASS
   with `_rust_core` absent (17 passed, 10 skipped) — they assert the
   *non-compliant fallback*, not the ImportError contract. CI-green on #98/#99
   does NOT demonstrate compliance; #105 must add tests asserting `ImportError`.
3. SPEC TENSION on `jordan`: issue #84 body says SciPy/SymPy fallback is
   "acceptable as the primary path until the Rust version stabilizes." This
   conflicts with §20.1 (tier-3 → ImportError) and #105 AC/decision #2. Per
   CLAUDE.md §1 the immutable spec (§20.1) wins → ImportError. Flagged for human
   /Archimedes to confirm before #105 lands (do not silently resolve).

#105 decisions confirmed: (#1) rank/nullspace are listed under #85 in the §20.4
table = Tier-3 → ImportError despite NumPy/SciPy overlap; (#2) composed methods
(polar/qdr/diagonalize/jordan, gauss_jordan/rank/nullspace) are absent-from-NumPy
features → ImportError. Both readings agree with the spec.

Tier-2 confirmed CORRECT, leave as-is (NOT a violation): #76 `svd/qr/lu/cholesky/
eigh` dispatch rust + NumPy fallback; `eig` is pure-NumPy on both paths (allowed —
fallback tier); #77 stats dispatch rust + NumPy fallback. §20.1 requires these
NumPy-wrapping fallbacks; they must remain.

#84/#85 stay OPEN until #105 lands. I gate the #105 PR (re-verify ImportError
contract + Tier-2 untouched + CI logs) before posting "approved." Commented on
#84 and #85.

### 2026-06-19T11:35Z — Newton — approved #101 (PR #110)

Euclid's #110 (docs-only resolution of #101). Independently verified: read the
full diff (only `_constructors.py` docstring + `docs/examples.rst` doctest
section; `_core.py` untouched → no #105 collision); empirically confirmed all
documented claims (`ColVec(np.empty((0,1)))`→`ColVec([])` shape `(0,1)`;
`(1,0)`→`ShapeError`; `_c[[1,2,3]]`→`ValueError`); ran `sphinx -b doctest` (new
section's 5 doctests pass, 0 new failures — the 5 remaining examples.rst failures
pre-exist on `main` at the same lines +31 and are correctly left out-of-scope per
§3/§6.6); `pytest test_constructors+test_namespace` 36 passed; CI green at head
`fe0bfaac` (Tests/Lint/PR checks all success, read the Actions runs myself).
Doc-only PRs are gate-exempt, but verdict recorded since review was requested.
Formal GitHub Approve blocked (crew shares repo-owner identity) → "approved"
posted as a PR comment.

### 2026-06-19T11:55Z — Newton — approved #105 (PR #113) + #109 (PR #114)

Gated the two new feature PRs; both pass. Built `_rust_core` myself (maturin
build → wheel → loaded the `.so`) to verify the Rust paths CI skips, then removed
the artifacts (tree left clean).

**PR #113 (#105 — Tier-3 ImportError):** all 13 methods route through new
`_core._require_rust(method)` (clear ImportError naming method + `maturin
develop`); composed methods gated before delegating; Tier-2 #76 + #77 untouched;
no signature changes. Verified OWNER's §6.6 authorization (issue #105,
2026-06-19T11:23) for the test rewrites — legit. jordan = strict ImportError per
OWNER ruling; Tier-1 dispatch gap deferred to #111. Ran: 17 passed/13 skipped
(rust absent, = CI) and 30 passed (rust built). CI green at `cf5dd69` (12 checks).
TDD order correct. Nit (non-blocking): §6.6 auth not cited in commit message (is
in PR body). #84/#85 close on merge.

**PR #114 (#109 — Phase 2 fused_add/mul/div, Tier-2):** forward `+`/`*`/`/`
dispatch to new Rust kernels under the `__sub__` guard; faithful mirror of
`fused_sub`. Built extension and ran parity test (spy confirms real dispatch):
Rust values+types == NumPy fallback for ColVec & Mat; shape-guard message
byte-identical to `_check_shapes` (no broadcasting introduced). Tier-2 fallback
retained (no ImportError — correct, different tier from #105). Forward-only
mirrors precedent. Scope: ops.rs + _operators.py + add-only tests; `_core.py`
untouched. Ran: 10 passed (rust)/80 passed (fallback). CI green at `1fa11b0` (12
checks incl. CodeQL rust compile).

Both marked ready-for-review (un-drafted) and "approved" posted as PR comments
(formal Approve blocked — shared repo-owner identity). Human merges.
