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

### 2026-06-19T12:10Z — Einstein — note: ownership ruleset + full backlog assigned
Adopted a hard rule (persona `.claude/commands/einstein.md`): every issue names an
explicit **Owner + Status + Branch + Files** at creation/grooming time so a
resuming builder can self-select its scope and start cold without clarification.
Assigned owners to all un-delegated issues by domain file-zone (keeps builders
disjoint), dependency-ordered. Per-builder backlogs:

**Archimedes** — decomposition/elimination zone (`decomp.rs`, `_decompositions.py`,
`_elimination.py`, `linalg.rs`):
1. #105 active (Tier-3 ImportError gate)
2. #111 blocked-until #105 (build schur/jordan/diagonalize Rust kernels)
3. #87 ready-after #111 (AHP/ANP — eigenvector reuse)

**Gauss** — core-ops/optimization zone (`ops.rs`, `_operators.py`, `optim.rs`,
`lazy.rs`):
1. #109 active (Phase 2 fused arithmetic)
2. #91 ready-after #109 (LP/IP/MILP — `optim.rs`, Tier-3)
3. #80 ready-after #91 (batch ops — `ops.rs`, Tier-2 keep fallback)
4. #78 ready-after #80 (NamedMat — `ops.rs`, Tier-2)
5. #81 blocked-until #80 (lazy eval — `lazy.rs`, Tier-3)
6. #79 blocked-until #91 & #93 (finance — Tier-3; call #93's sim API, don't edit sim.rs)

**Euclid** — interop/network/stochastic zone (`polars.py`, compat tests,
`network.rs`, `markov.rs`, `sim.rs`):
1. #101 active (docs-only)
2. #63 ready-after #101 (pandas compat — testing)
3. #66 ready-after #63 (polars compat — testing)
4. #92 ready-after #66 (network opt — `network.rs`, Tier-3)
5. #86 ready-after #92 (Markov — `markov.rs`, Tier-3)
6. #93 blocked-until #80 (stochastic OR — `sim.rs`, Tier-3; provides sim API for #79)

**Newton** — verify #84/#85 (done, PR #108); gates every builder PR.

**Holds (not assignable):** #102 (await maintainer A/B/C on `_c` strings),
#106 (needs §7.3 amendment + sign-off), #76/#77 (merged Tier-2, for human to close).
Tier rule reminder propagated to every Tier-3 issue: ImportError, no NumPy
fallback; Tier-2 (#80/#78): keep the NumPy fallback.

### 2026-06-19T13:20Z — Einstein — note: progress + Rust-core build push (top priority)
Merged since last session: #105 (PR #113, Tier-3 ImportError — Newton approved),
#109 (PR #114, Phase 2 fused arithmetic — Newton approved), #101 (PR #110, docs),
#108 (Newton audit), #115 (backlog/ownership rule), #116 (uv migration; Python
>=3.10, repo pins 3.14, uv.lock committed). Builders are now free of their merged
tasks. Maintainer needs to USE nemopy + the Rust core for real-world testing, so a
**Rust-core build/install effort preempts the feature backlog**.

### 2026-06-19T13:20Z — Einstein — delegate #117/#118/#119 (Rust-core build, top priority)
Filed three file-disjoint issues, agent called out in each title:
- **#117 [Gauss]** — reproducible local build & install (`scripts/build_rust.sh`,
  `_rust_core/pyproject.toml`/`Cargo.toml`). Ready, TOP priority.
- **#118 [Euclid]** — install/usage docs, two-mode (`README.md`, `docs/`). Ready, parallel.
- **#119 [Archimedes]** — build verification test suite (`tests/test_rust_build.py`). Ready, parallel.
Re-sequenced the displaced queue items: **#91 → ready-after #117** (Gauss),
**#63 → ready-after #118** (Euclid), **#111 → ready-after #119** (Archimedes; its
old `blocked-until #105` is cleared — #105 merged). No file conflicts, only ordering.
Off-limits flagged in each: root `pyproject.toml` (sign-off) and CI/CD (§10) — so a
one-command `uv add`/CI-published wheels are deferred, sign-off-gated follow-ups.

### 2026-06-19T13:20Z — Einstein — note: shared Rust touch-point (lib.rs / Cargo.toml)
Coordination hazard for the upcoming Rust-module issues (#91 optim.rs, #92
network.rs, #86 markov.rs, #93 sim.rs, #81 lazy.rs): every new module edits
`_rust_core/src/lib.rs` (`mod x;` + `x::register(m)?;`) and possibly `Cargo.toml`
(deps) — a shared serialization point regardless of owner zone. Rule: treat these
edits as **append-only at the end** of the respective lists, and never run two
module-adding issues through review concurrently without coordinating the merge.
#117 (build) does NOT add a module, so it's clean.

### 2026-06-25T00:00Z — Einstein — note: maintainer design review → 4 spec decisions
Maintainer brought a REPL transcript raising three core-type issues and made four
rulings: (1) **slicing** — bare `A[i]` selects COLUMN i (column-first), not row data;
(2) **repr** — `ColVec` prints vertically; (3) **float hygiene** — display normalize/chop
+ numeric `round`/`clean`/`isclose` + algorithmic near-singular guard (all three layers);
(4) **preempt the backlog** with these. Verified each against spec: all were either
spec-mandated current behavior (§4.2 horizontal repr; §4.3 `.6g` → `-0`) or a §6.3 internal
contradiction (`A[i]` impl defaulted to ColVec). `nm.round` was a numpy leak via
`from numpy import *`, not a feature. Maintainer then **explicitly authorized editing the
immutable spec**.

### 2026-06-25T00:00Z — Einstein — note: authored spec amendments → PR #128 (draft)
Authored the four amendments myself (conductor owns design; keeps Newton's anchor stable):
§6.3/§15 (A[i]→column, Mat.__iter__→columns), §4.2/§4.3/§4.6 (vertical ColVec + shared
`_fmt_entry` signed-zero/relative-chop), §21 (new — `__round__`/`clean`/isclose passthrough),
§9.1/§9.5/§10.1/§10.2/§2.3 (`IllConditionedWarning` on near-singular `.inv`). Swept the 5
in-spec ColVec doctests. PR #128 is **spec-only + this log** (single-branch session). Flagged
5 design choices for maintainer sign-off on the PR (iter→columns; default-on relative chop
@1e-12; vertical format; numerics signatures; cond threshold + the extra-SVD perf cost).

### 2026-06-25T00:00Z — Einstein — delegate #129/#130/#131/#132 -> Gauss (serial _core.py chain)
Created four impl issues, all owner **Gauss**, gated on #128 merging, **serialized on
`nemopy/_core.py`** (one-owner-per-file): #129 (A[i]→column + __iter__) → #130 (vertical
repr + display hygiene) → #131 (numerics API) → #132 (near-singular warning). #130 also
sweeps merged #101's empty-ColVec doc artifacts (`ColVec([])`→`ColVec(0):`) under §6.6.
This preempts Gauss's #91 (LP). **Archimedes** continues #119 (in review) → #111 (decomp
kernels, `decomp.rs`/`_decompositions.py`) — disjoint from `_core.py`, runs in PARALLEL.
**Euclid** HOLD #63/#66 until #129/#130 land — pandas/polars compat tests assert the
changing repr/indexing behavior. Newton gates each PR (#128 spec, then #129–#132).

### 2026-06-25T00:00Z — Einstein — note: audit of open issues vs amended spec
Per maintainer ask, scanned the backlog for conflicts with #128. Advisories posted:
`.inv`-now-warns → **#66** (normal-equations compat — expect/filter the warning), **#78**
(labeled `.inv` bridge inherits it), **#80** (per-element batch `.inv` — design how a batch
surfaces it; also align BatchMat to column-first Mat). Repr-format → **#102** (held; old
`ColVec([])` examples to refresh when actioned); **#101** (merged) handled by #130's sweep.
Merged decomposition code (#84/#85) that calls `.inv` internally may now emit the warning —
informational, low risk, no reopen. Rust-domain issues (#79/#86/#87/#91/#92/#93) unaffected.
