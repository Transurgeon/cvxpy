# Session handoff — updated 2026-07-13

## v0.6.1 RELEASED (2026-07-13) — the release blocker is CLEARED

- **SparseDiffEngine v0.6.1** shipped: `release/0.6.x` + cherry-picks of #102 (Swedish
  sparsity fill, `c549c64`), #106, #107 (in main order); tree verified identical to engine
  main except version; 418 C tests green locally pre-tag; Release workflow green; GitHub
  Release published. Branch continues at "Begin 0.6.2 development".
- **sparsediffpy 0.6.1 is LIVE ON PYPI**: release/0.6.x = Windows-fix cherry-pick +
  submodule pinned at engine v0.6.1 tag commit (`86b4bcc`); all 12 wheels + sdist green
  (incl. all 4 Windows jobs), TestPyPI -> PyPI uploads succeeded.
- **Windows CI flake FIXED** (SparseDiffPy PR #20, merged by user): scikit-build-core's
  TemporaryDirectory build dir cleanup raced a lingering handle (mspdbsrv/AV) ->
  WinError 32. Fix: `build-dir = "build/{wheel_tag}"` in `[tool.scikit-build]` (persistent,
  gitignored, no temp cleanup). On main AND in the 0.6.1 release.
- **Verified end-to-end**: fresh Python 3.14 venv + PyPI `sparsediffpy==0.6.1` + pr-c
  cvxpy build — all 5 #107 canaries pass; full test_ignore_dpp.py +
  test_diffengine_backend.py (38 tests) pass. (NLP-path #3455 repro not re-run — no IPOPT
  in the fresh venv — but was previously verified cured by the same engine fix.)
- **Pin bumped on the stack**: new commit `15ecf9faf` "Require sparsediffpy >= 0.6.1" on
  pr-a; pr-b/pr-c rebased on top (pin-only delta vs old tips) and force-pushed. Upstream
  PR CI re-running; pr-c canaries should now be green on CI.
- **SparseDiffPy PR #21 open (user to review)**: bumps main's submodule pointer
  `7e7678e` -> `4172c5e` (engine main with #102/#106/#107) so main source builds stop
  carrying the stale re-solve bug.
- Still open: comment on cvxpy#3455 that 0.6.1 ships the fix; un-xfail path for the
  #3454 NLP tests now available. Local dev state restored (Py submodule checkout back at
  `fa911a7`; both clones on main).

# Session handoff — updated 2026-07-12

## #104 replacement (NEW, 2026-07-12): `param-source-mark-refresh`

User wants #104 closed (dislikes the base-expr attribute; against roadmap). Replacement
implemented and validated: engine branch **`param-source-mark-refresh`** (= 7e7678e +
#102 cherry-picks + commit `c93a83c`) — 6 one-line `expr_set_needs_refresh(param_source)`
calls in the gated atoms (scalar/vector_mult, left_matmul [covers right], convolve, kron,
quad_form) + 3 C tests (`tests/problem/test_param_source_refresh.h`) that fail without
the fix. Verified: 417 C tests pass; both cvxpy canaries + all probe shapes pass on this
build WITHOUT #104; full cvxpy suite run in progress on it (main venv now runs this build,
NOT the #104 one).

Empirically verified failing-shape taxonomy (probes in session scratchpad): stale iff a
gated node inside a param_source has a computed (non-leaf) source — reachable shapes:
`(p*A) @ x` (broadcast promote!), `quad_form(x, g*Sig)`, `I/p` (quad-objective canon),
`((2p)*A) @ x`. NOT failing: `(2*p)*x`, `p1+p2`, bare-Parameter matrix sources.

**2026-07-12 latest — the composed-parametric contract went upstream:**
- **cvxpy#3454** (branch `add-composed-param-refresh-tests`, based on upstream master):
  backend-agnostic conic-path tests in test_dpp.py (GREEN on master) + NLP-path tests in
  nlp_tests/test_nlp_parameters.py (xfail'd, IPOPT-gated). Supersedes the stacked draft
  #3453 (closed with pointer). pr-c (#3450) keeps its diffengine-specific canaries.
- **cvxpy#3455 (ISSUE): LIVE MAINLINE BUG** — parametric DNLP re-solves serve stale data
  for composed coefficients ((p*A)@x) on master + pinned sparsediffpy 0.6.0 (repro in
  issue; verified: stale on 0.6.0, cured by the #107 engine fix). This makes engine #107
  a MAINLINE bugfix dependency, not just a pr-c one — strongest possible argument for
  merging #107 + cutting a release + bumping master's pin (currently `>=0.6.0,<0.7.0`).

**2026-07-12 late updates:**
- **Benchmarks published**: the whole canonicalization experiment set (casadi_compare
  harness + 4 builder modules, run_backend_benchmarks, mechanism probes, reference
  results) now lives in github.com/SparseDifferentiation/Jacobian-accumulation under
  `canonicalization/` (commit 4171023; repo restructured, drafts folded/deleted, cvxpy
  pin -> pr-c branch, uv.lock committed, `uv sync` + verify validated in a fresh env).
  The copies in this repo's untracked benchmarks/ are now secondary. Thesis §4.1 cites
  the repo in a footnote; PDF rebuilt.
- **#102 is MERGED upstream** (main `c549c64` "Swedish algorithm for sparsity fill").
- **#107 rebased onto origin/main** (single commit `4248f7d`, clang-formatted, force-pushed)
  so its diff is ONLY the mark-refresh fix + tests. 417 C tests green on main+fix.
- Both venvs + submodule now build from `param-source-mark-refresh` (= origin/main + fix,
  which includes merged #102 and #106). Old integration branches kept:
  `dev-integration-0.7-fix102`, `dev-integration-0.7-plus-102`, `-only-102`.
- The upstream cvxpy PRs pr-a/b/c are ALREADY OPEN (user). pr-c updated: canary-test
  commit `ca08a26d1` pushed to Transurgeon/cvxpy. CAVEAT: pr-c's canaries fail on CI
  until the pin points at an engine release containing the #107 fix (PyPI 0.6.0 lacks
  it) — next release after #107 merges, then bump pin.

**PR OPENED (2026-07-12): SparseDiffEngine #107**
(https://github.com/SparseDifferentiation/SparseDiffEngine/pull/107) — the
param-source-mark-refresh branch; body includes taxonomy + supersedes-#104 note. User
to close #104. cvxpy-side: 5 canary tests now in the working tree (UNCOMMITTED, fold
into pr-c): test_ignore_dpp.py (symbolic quad matrix, scaled scalar coefficient control,
scaled MATRIX coefficient, scaled QUAD matrix) + test_diffengine_backend.py (scaled
coefficient on the cached DPP path). TEST-DESIGN LESSON baked into them: the optimal
POINT must depend on the parameter — cvxpy recomputes prob.value from the live
expression at unpack, so a parameter-invariant argmin masks stale problem data (solve
returns the right value off stale matrices). All 5 verified to FAIL on an
engine without the fix and PASS with it.

A cvxpy-side alternative (fold composite coefficients to CallbackParams at
`from_problem`) was prototyped, fully validated, then REVERTED at user request (too
invasive). Kept: two canary tests in `test_ignore_dpp.py` (symbolic quad matrix +
scaled param coefficient). Next: user closes #104, opens the mark-refresh branch as
the replacement engine PR; release pin needs #102 + this fix.

# Session handoff — updated 2026-07-10

Current state of the diffengine work. Design/rationale live in `IGNORE_DPP_BEHAVIOR.md`
and `TODO.md` (source branch); this file is only cross-repo state and open items.

## The stacked upstream branches (NEW, 2026-07-08)

`PR_SPLIT_PLAN.md` was executed. Three curated branches, stacked, based on
upstream/master `2e57c9a68`, each full-suite green locally (2 known pre-existing
MI failures deselected; `hypothesis` + `ruff` installed in venv):

- **pr-a-diffengine-backend** — opt-in backend (explicit `canon_backend`/env var
  only; zero default change). Tests: `test_diffengine_backend.py` (new name; the
  converter/program/selection subset, solves converted to explicit backend),
  `test_cvxcore_issues.py` (explicit backend). Pin bumped to `sparsediffpy >= 0.6.0`.
- **pr-b-ignoredpp-default** — ignore_dpp/non-DPP → DIFFENGINE + `CallbackParamFold`
  + N-D EvalParams fallback + explicit-backend ValueError + `IGNORE_DPP_BEHAVIOR.md`.
  Deliberately UNCACHED (`uncached_param_prog=True` on both parametric branches);
  cache-identity assertions softened in `test_dpp.py`. New `test_ignore_dpp.py`
  (behavior + selection subsets).
- **pr-c-resolve-caching** — safe_to_cache + `param_quad_form_factorized`
  record-at-site + theta short-circuit/extraction-once; cache tests restored/added
  (`TestResolveCaching`, `TestIgnoreDppCacheHygiene` in `test_ignore_dpp.py`).

Final-tree check vs `diffengine-backend-ignoredpp`: identical except intended
deviations — TODO.md excluded; test_ignore_dpp.py split into
test_diffengine_backend.py (A) + test_ignore_dpp.py (B/C); test_cvxcore_issues.py
uses explicit backend; IGNORE_DPP_BEHAVIOR.md corrected (stale "nothing folds
log_det/norm" claims — they DO fold; QP SolverError comes from problem_form's
conservative cone count) and TODO.md reference stripped; test_dpp.py log_det
docstrings corrected likewise. Hunk map + deviations list:
scratchpad `pr-split-map.md` of session 848c1e3c (copy below if needed).

## Engine / release state (moved since 2026-07-08!)

- SparseDiffEngine upstream main is at 0.7.0-dev (`7e7678e`): #101 (kron) and
  #105 (gather) are MERGED; **v0.6.0 is now PUBLISHED on PyPI** (2026-07-10;
  contains #101+#105, NOT #104). **#104 (param-source refresh + node
  ownership) is still OPEN, in draft** — required by PR-A's DPP-parametric
  caching and all of PR-C. dance858 commented on #104 asking for a call
  "this weekend" to understand context/motivation — meeting brief prepared
  2026-07-10 (artifact "meeting-brief-104-102"). **#102 (output-driven
  sparsity fills) became LOAD-BEARING on 2026-07-11**: without it the engine's
  cold compile on the upstream SemidefiniteProgramming benchmark is 430 s vs
  CPP 5.3 s (80x); with it, 1.93 s (0.35x). QuantumHilbertMatrix similarly
  22.5 s -> 1.66 s. The venv now runs `dev-integration-0.7-plus-102`
  (= 7e7678e + #104 cherry-picks + #102 cherry-picks; #104 canary green).
  Any engine release backing the upstream PRs must include #102 AND #104 —
  strong ammo for the dance858 call. Thesis benchmark state:
  `~/Documents/wzz-thesis/plans/benchmark_results.md`.
- Submodule clone `~/Documents/SparseDiffPy/SparseDiffEngine` is on new branch
  `dev-integration-0.7` = `7e7678e` + cherry-picks of #104 (`625008a`, `df40a3d`).
  The old `dev-integration` branch (0.6.0-dev based) still exists.
- cvxpy venv has this build installed (reports 0.7.0). Canary:
  `test_ignore_dpp.py::TestIgnoreDppBehavior::test_symbolic_quad_matrix_refreshes_through_cache`
  — it FAILS on a build without #104 (served stale 1.0-value on the p=1000 solve).

## Open items

- Get engine #104 merged (blocking PR-C correctness on a released engine; it
  missed 0.6.0, so PR-C needs a `>= 0.6.1` — or 0.7.0 — pin; release vehicle
  to be decided on the dance858 call).
- When sparsediffpy lands on PyPI at the needed version: push the three branches,
  open stacked upstream PRs (body draft: `benchmarks/pr_comment_ignore_dpp.md`),
  A first. Re-run the re-solve benchmark table for the PR draft
  (`benchmarks/run_upstream_benchmarks.py`).
- cvxpy#3446 (standalone #3442 regression tests) still open, unchanged.
- `benchmarks/`, `cvxpy/version.py`, this file, and `PR_SPLIT_PLAN.md` remain
  deliberately untracked (never commit; `git add cvxpy/` grabs them — use
  explicit paths or amend, it bit twice today).
- Old source branch `diffengine-backend-ignoredpp` stays as reference; the
  single-PR fallback remains viable from it.
