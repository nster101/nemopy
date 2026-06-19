# Einstein — coordination log

<!-- Append-only; newest entries at the bottom. Only Einstein writes to this file. -->
<!-- Entry format: ### <UTC ISO-8601 to the minute> — Einstein — <topic> -->

### 2026-06-18T16:57Z — Einstein — note: session status
First conducting session. All five logs empty; no open PRs. Phase 3 (Rust core
#75 scaffold, #76 core decomps, #77 stats, #84 advanced decomps, #85 elimination)
is merged to `main`. Housekeeping: #76/#77/#84/#85 issues are still OPEN despite
their PRs merging — #84/#85 must now STAY open (see audit below); #76/#77 (Tier-2,
clean) are candidates for the human to close. Human set priority: **bugs first**
before resuming the feature roadmap (#91 LP is next when bugs clear).

### 2026-06-18T16:57Z — Einstein — note: Tier-3 fallback audit (#84/#85)
Post-merge scope audit found the merged Tier-3 work silently computes via
NumPy/SciPy when `_rust_core` is absent instead of raising `ImportError`
(`_core._load_rust_core` returns None; each method has an `else` NumPy branch).
Violates `DESIGN_APPENDICES.md` §20.1/§20.4 (Tier-3 = absent-from-NumPy →
ImportError, no fallback). Confirmed two load-bearing claims directly
(`_core.py:989-990`, `_elimination.py:97-99`). Human confirmed: Tier-2
NumPy fallbacks (#76/#77 — things NumPy has) are GOOD and stay; Tier-3
(#84/#85 — things NumPy lacks) must raise ImportError. Tier-2 untouched.

### 2026-06-18T16:57Z — Einstein — delegate #105 -> Archimedes
Created #105 (fix: Tier-3 methods raise ImportError, remove NumPy/SciPy
fallbacks). Touches `nemopy/_decompositions.py`, `nemopy/_elimination.py`,
`nemopy/_core.py` (loader/dispatch). Tier-2 decomps + #77 stats OUT of scope.
Two decisions flagged in-issue (rank/nullspace numpy overlap; composed methods
polar/qdr/jordan/diagonalize). #84 and #85 stay OPEN until #105 lands.
**Archimedes:** please claim #105 — this is the top bugs-first item.

### 2026-06-18T16:57Z — Einstein — handoff #84/#85 -> Newton
Routed the already-merged #84/#85 to **Newton** for an independent post-merge
spec re-verification — the original merges (PR #98/#99) may not have passed the
review gate, and Newton's findings may refine #105's violation list. Newton also
gates the eventual #105 PR.

### 2026-06-18T16:57Z — Einstein — delegate #101 -> Euclid
#101 (blank ColVec) ruled **docs-only, no behavior change**: `_c[]` is invalid
Python; `_c[""]` is the #102 string question; empty `(0,1)` ColVec already works
via `ColVec(np.empty((0,1)))`. **Euclid:** small docs task — document the
supported empty-placeholder idiom + `_c` scalars-only contract. Keep edits in
`nemopy/_constructors.py` docstrings + `docs/`; **do NOT touch `nemopy/_core.py`**
(Archimedes owns it for #105) — avoids file collision.

### 2026-06-18T16:57Z — Einstein — note: #100 closed, #106 opened, #102 pending
#100 (shape-guard relax) decided **Option C — status quo**, CLOSED. Option A
(relax via opt-in compatibility mode) deferred to new roadmap issue **#106**
(needs §7.3 amendment + human sign-off; do not claim yet). #102 (`_c` MATLAB
string parsing): posted a tradeoff exploration (Options A/B/C) — **awaiting
maintainer decision**, not yet a build task. Gauss currently free; hold for the
next roadmap item (#91) once bugs-first work is moving.

### 2026-06-18T17:30Z — Einstein — note: strategic direction (NumPy replacement parked)
Human ruling on long-term architecture: porting NumPy's C++/BLAS layer to Rust
for a **full NumPy replacement** is a worthwhile future epic but is **NOT now** —
nemopy's existing surface must be feature-complete first. So we stay within
current §20.1 (NumPy is the BLAS; Rust = application layer + novel features;
NumPy is the only hard dependency). No §20 spec amendment at this time. The full
replacement is parked as a future epic to scope once the roadmap lands.

### 2026-06-18T17:30Z — Einstein — delegate #109 -> Gauss
Audit found Phase 2 (§20.5 — hot-path replacement) was skipped: only `fused_sub`
is in Rust; `+`/`*`/`/` still run on pure NumPy. Created **#109** (Phase 2:
fused_add/mul/div in ops.rs). This is the most foundational gap in completing
nemopy, so it goes before more features. **Tier-2** (NumPy has these ops) — keep
the NumPy fallback, no ImportError (contrast #105). **Gauss:** claim #109,
branch `gauss/109-core-fused-arithmetic`, touch `ops.rs` + `_operators.py` only
(do NOT modify `_core.py` — Archimedes owns it for #105). Mirror `fused_sub`
exactly. Newton gates.

Current parallel layout (all file-disjoint): Archimedes #105
(`_decompositions`/`_elimination`/`_core`); Euclid #101 (`_constructors`/`docs`);
Gauss #109 (`ops.rs`/`_operators`); Newton — verify #84/#85 + gate PRs. #91 (LP,
Phase 4) is queued for Gauss after #109.

### 2026-06-19T11:40Z — Einstein — note: Newton verdict on #84/#85 (PR #108) + jordan ruling
Newton independently re-verified the Tier-3 audit (PR #108): VIOLATION CONFIRMED,
membership matches mine (13 methods). Three refinements: (1) schur/jordan/
diagonalize/augment (+ Tier-2 eig) have NO `_RUST` dispatch even when the
extension is built — a Tier-1 fast-path gap beyond the fallback issue; (2) the
merged tests assert the non-compliant fallback, so CI-green never proved
compliance; (3) spec tension: #84's brief carves out a SciPy/SymPy fallback for
`jordan` "until Rust stabilizes," conflicting with §20.1. Verified the #84 quote
myself — #84 is internally self-contradictory.
**Human ruling (2026-06-19): strict §20.1 — jordan raises ImportError**, carve-out
overruled (§1/§11 spec precedence).

### 2026-06-19T11:40Z — Einstein — note: #105 scope clarified, #111 filed
Updated #105 (comment) for Archimedes: scope = add the ImportError gate to ALL 13
Tier-3 methods + remove NumPy/SciPy fallbacks; jordan included (strict). Building
the missing Rust kernels is OUT of #105 — split to new issue **#111** (Tier-1
fast-path gap: schur/jordan/diagonalize kernels in decomp.rs; augment kernel TBD;
depends on #105). Authorized Archimedes (§6.6 exception, on Newton's PR #108
evidence) to update the existing fallback-asserting tests in
`test_decompositions_advanced.py`/`test_elimination.py` to assert the ImportError
contract. Newton gates the #105 PR. #84/#85 stay open until #105 lands; #111 is a
follow-up after that.
