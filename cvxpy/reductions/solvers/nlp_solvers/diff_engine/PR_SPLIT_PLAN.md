# Upstreaming plan: split `diffengine-backend-ignoredpp` into stacked PRs

Status: planned 2026-07-07, execution blocked on the sparsediffpy 0.6.0 release
(TODO §5: engine kron PR #101 + #104 + #105 → release → bump the cvxpy pin). Until the
pin bump none of these can go green on upstream CI, so there is time to curate.

Deliberately untracked (like SESSION_HANDOFF.md): this is process notes, not design —
it must not ship inside any of the split PRs.

## The split (dependency order)

### PR-A — DIFFENGINE canon backend, opt-in only

- Content: `diff_engine/` (converters, registry, extractor, helpers, c_problem),
  `DiffengineConeProgram`, the `ConeMatrixStuffing` dispatch, `settings.py` constant,
  reachable ONLY via explicit `canon_backend="DIFFENGINE"` or the env var. Zero
  default-behavior change. Includes the compile-cost work (matmul-chain normalization,
  lazy constant operands, kron converter) — it's part of the backend, not separable.
- Review shape: large but mechanical — "the new backend produces the same stuffed
  matrices as CPP" (`test_quad_objective_data_matches_cpp` and friends carry this).
- Needs: `sparsediffpy >= 0.6.0` pin bump (can be the first commit of this PR).
- Tests: the converter/extraction/explicit-backend subset of `test_ignore_dpp.py`
  (split out), plus the kron/complex-div/dpp xfail removals.

### PR-B — make DIFFENGINE the ignore_dpp / non-DPP default (behavioral)

- Content: EvalParams removal for ≤2-D parametric non-DPP/ignore_dpp, N-D fallback +
  `uncached_param_prog`, explicit-tensor-backend `ValueError`, `CallbackParamFold`
  (`cvxpy/reductions/fold_callback_params.py` + wiring), Dcp2Cone without
  `canon_param_constants` (upstream constant handling restored), fail-loud converter
  semantics, QP `SolverError` behavior for cone-emitting composites.
- Review shape: SMALL diff, maximal scrutiny — epigraph soundness (`x <= power(t,2)`),
  DQCP bisection, the `IGNORE_DPP_BEHAVIOR.md` doc lands here.
- The fold does NOT go standalone before this PR: a reduction with no consumer invites
  "why?"; it ships with its consumer.
- Tests: `test_fold_callback_params.py`, the behavior subset of `test_ignore_dpp.py`
  (soundness, floor/DQCP, fallback, backend-selection), `test_dpp.py` log_det cases.

### PR-C — re-solve caching

- Content: `safe_to_cache` in `problem.py`, `InverseData.param_quad_form_factorized`
  (Dcp2Cone record-at-site), `DiffengineConeProgram.apply_parameters` theta
  short-circuit / extraction-once, cache toggling.
- Review shape: correctness-of-caching — the cache-hygiene tests
  (`test_symbolic_quad_matrix_refreshes_through_cache`,
  `test_parametric_constraint_quad_form_not_cached`, extraction-once, toggle tests)
  plus the benchmark table from `benchmarks/pr_comment_ignore_dpp.md`.
- Requires engine PR #104 (param_source refresh + node ownership) in the released
  sparsediffpy — already part of the 0.6.0 checklist.

## Execution notes

- The branch history is NOT layered this way — curate three fresh branches by CONTENT
  (the `tests-3442-duplicate-gather` procedure: branch off upstream/master, build each
  diff, own suite green), not by cherry-picking commit ranges.
- `test_ignore_dpp.py` spans all three layers and must be partitioned; keep the split
  boundaries identical to the PR boundaries so each PR's CI is self-contained.
- Already-standalone pieces (do not fold back in): cvxpy#3446 (#3442 regression tests),
  engine PRs #104/#105.
- The single-PR fallback stays viable: the branch history is 4+2 clean commits with
  `benchmarks/pr_comment_ignore_dpp.md` as the body. Decide when the release lands and
  reviewer bandwidth is known; PR-A alone already converts the review into
  "one mechanical PR + one small behavioral PR", which is most of the benefit.
