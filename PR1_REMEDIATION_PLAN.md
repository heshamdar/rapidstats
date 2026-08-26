# PR #1 remediation plan

Follow-up work for [PR #1](https://github.com/heshamdar/rapidstats/pull/1),
*"test: add characterization net and shared test scaffolding"* — eleven commits
covering polars compatibility, bootstrap performance, six correctness fixes, and
null-versus-NaN semantics.

The PR itself is sound work. This document covers what a full review found on top
of it: two high-severity defects it introduces, one high-severity defect it
inherits and newly cements into a fixture, and a tail of medium and low items.

---

## Before you start

**The code these phases patch is on the PR branch, not on `master`.**

| | |
| --- | --- |
| Target branch | `claude/polars-lazy-eval-review-pxn4li` |
| PR head at review time | `7a2341a` |
| Base | `master` @ `6de8dbf` |

Every file:line reference below is against `7a2341a`. This plan file lives on a
branch cut from `master`, so it will not sit next to the code it describes —
copy it across, or read it from this branch while working on the PR branch.

If the PR has already merged, work from the merge result and re-derive line
numbers; the symbols named in each phase (`_jacknife`, `_air_func`,
`set(thresholds or y_score)`, …) are stable enough to grep for.

### Phase 0 — reproduce the baseline

The findings below were all reproduced against a local build. Do the same before
changing anything, so you can tell your regressions from the environment's.

```bash
git fetch origin claude/polars-lazy-eval-review-pxn4li
git checkout claude/polars-lazy-eval-review-pxn4li

uv venv --python 3.12 .venv
uv pip install maturin polars pytest numpy scipy pandas pyarrow narwhals tqdm
source .venv/bin/activate
maturin develop --release     # ~5-6 min cold
```

Expected suite result on a current environment:

```bash
pytest -q --ignore=tests/test_selection.py --deselect tests/test_bootstrap.py::test_bca_interval
# 780 passed, 14 skipped
```

Two pre-existing failures are environment drift, not this PR:

- `tests/test_bootstrap.py::test_bca_interval` calls `scipy.stats._resampling._bca_interval`,
  a private function that gained a required `xp` argument.
- `tests/test_polars.py::test_auc` calls `np.trapz`, removed in NumPy 2.

Both are repaired in Phase 7. `tests/test_selection.py` needs catboost, lightgbm,
xgboost and scikit-learn; install them if you are touching `selection.py`
(Phase 4 does).

---

## Phase ordering

Phases 1–4 are small, self-contained, and close everything the PR newly breaks.
Phase 2 should not wait: it changes a committed fixture, which is much cheaper
before merge than after. Phases 5–7 are follow-on work and can land separately.

| Phase | Closes | Size | Blocking for merge? |
| --- | --- | --- | --- |
| 1. Collect before the cum_sum jackknife | I1 | XS | Yes |
| 2. Sort the Poisson AIR frame | L1 | S | Yes — touches a fixture |
| 3. Resolve thresholds from the frame | I3 | S | Yes |
| 4. Dependency and concurrency contracts | I4, I5, I6 | S | Yes |
| 5. Make cum_sum BCa affordable, or bound it | I2 | M–L | Recommended |
| 6. Release and documentation gap | I7, I8, I10 | S | Recommended |
| 7. Make CI test what the PR claims | L2, L5, L6 | S | No |

Finding IDs are defined in the [appendix](#appendix-findings-index).

---

## Phase 1 — collect the frame before the cum_sum jackknife

**Closes:** I1 (high) — and the `n_jobs`/`quiet` half of I6.

### The defect

On the Poisson path, `Bootstrap.confusion_matrix_at_thresholds` converts `df` to
a `LazyFrame` at `_bootstrap.py:796` so `_poisson_sample` can gather indices. The
BCa branch at `:891` then hands that LazyFrame to `_jacknife`, which needs
`df.height` to build its row index.

```python
rs.Bootstrap(iterations=10, seed=1, method="BCa", sampling_method="poisson") \
  .confusion_matrix_at_thresholds(y_true, y_score,
                                  thresholds=[0.2, 0.5, 0.8],
                                  strategy="cum_sum")

# AttributeError: 'LazyFrame' object has no attribute 'height'
```

`adverse_impact_ratio_at_thresholds` has exactly this fix at `:1340`; it was
never ported across. Unreachable before this PR, because commit `4a2f20b`
removed the `NotImplementedError` that guarded these paths.

The same line also omits `**self._concurrent_kwargs`, so `n_jobs` and `quiet`
are ignored for this jackknife — the precise bug commit `2581ce4` set out to
eliminate.

### The change

`python/rapidstats/_bootstrap.py:891`

```diff
-                jacknife_lf = pl.concat(_jacknife(df, _cm_inner), how="vertical").pipe(
-                    _process_results
-                )
+                # `df` is a LazyFrame on the poisson path, but the jackknife indexes
+                # rows and needs a height.
+                jacknife_df = df.pipe(_collect) if isinstance(df, pl.LazyFrame) else df
+                jacknife_lf = pl.concat(
+                    _jacknife(jacknife_df, _cm_inner, **self._concurrent_kwargs),
+                    how="vertical",
+                ).pipe(_process_results)
```

### Test first

`tests/bootstrap/test_bca_cum_sum.py` only exercises the multinomial default,
which is why this shipped. Parametrize it:

```python
@pytest.mark.parametrize("sampling_method", ["poisson", "multinomial"])
@pytest.mark.parametrize("resample_mode", ["weights", "materialize"])
def test_bca_confusion_matrix_at_thresholds(data, sampling_method, resample_mode):
    ...
```

Confirm the `poisson` case fails with `AttributeError` before the fix.

### Acceptance

- All four `sampling_method` × `resample_mode` combinations produce a frame on
  both cum_sum BCa paths, matching the coverage `test_resample_mode.py`
  already gives `roc_auc`.
- `n_jobs=1` produces a deterministic result on the cum_sum BCa path.

---

## Phase 2 — sort the AIR frame on the Poisson path

**Closes:** L1 (high, pre-existing on `master`).

### The defect

`Bootstrap.adverse_impact_ratio_at_thresholds` selects
`_air_at_thresholds_core_sorted` for the Poisson path at `_bootstrap.py:1264`.
Unlike `confusion_matrix_at_thresholds`, which sorts its frame at construction,
this method never sorts — the frame built at `:1226` goes straight to
`df.lazy()` at `:1266`. Every Poisson iteration therefore runs a cumulative
approval-rate scan over unordered scores.

It fails silently. No error, plausible-looking numbers, intervals roughly 3.5×
too wide:

```
one Poisson resample, thresholds [0.2, 0.5, 0.8]

_air_at_thresholds_core_sorted (what runs):   0.2999   0.4986   0.7836
_air_at_thresholds_core        (sorts first): 0.8643   0.9773   0.9637

300-iteration interval at t = 0.5
  multinomial   0.813 .. 1.226      (width 0.41)
  poisson       0.145 .. 1.598      (width 1.45)   point 1.0006 in both
```

`_poisson_sample` gathers indices in ascending order, so sorting once up front
is all `_air_at_thresholds_core_sorted` needs — the same arrangement
`confusion_matrix_at_thresholds` already relies on.

### The change

`python/rapidstats/_bootstrap.py:1266`

```diff
             if self._params["poisson"]:
                 _air_func = _air_at_thresholds_core_sorted
                 _sample_func = functools.partial(_poisson_sample, df_height=df.height)
-                df = df.lazy()
+                # `_air_at_thresholds_core_sorted` assumes an ascending sort, and
+                # `_poisson_sample` gathers indices in order -- so sorting once here
+                # holds for every iteration. Without it each resample was scanned
+                # unordered, which silently widened the interval by ~3.5x.
+                df = df.sort("y_score").lazy()
```

### Sequence this carefully

The PR's own characterization fixture
`tests/data/characterization/bootstrap_air_at_thresholds_percentile_poisson.parquet`
currently **pins the incorrect output**. Fixing the bug will make that fixture
fail — which is correct, but means the order of operations matters:

1. Write the failing test first. Poisson and multinomial AIR intervals must
   agree to within Monte Carlo error, the way `test_modes_agree_on_roc_auc`
   already asserts for `roc_auc`. It currently fails by ~3.5× on interval width.
2. Apply the sort. Confirm the new test passes and the characterization case
   goes red.
3. Regenerate the fixture:
   ```bash
   pytest tests/test_characterization.py --update-characterization
   git diff --stat tests/data/characterization/
   ```
   Only `bootstrap_air_at_thresholds_percentile_poisson.parquet` should change.
   If anything else moves, stop and find out why.
4. Say plainly in the commit message that this fixture was pinning incorrect
   output, so the diff is not mistaken for an unexplained numeric drift later.

### Acceptance

- Poisson and multinomial AIR-at-thresholds intervals agree within Monte Carlo
  error at 300+ iterations.
- Exactly one characterization fixture changes.
- A comment at the sort records *why* it is there, so it is not "tidied away".

### Note on the technique

This is worth writing down somewhere durable. A characterization net freezes
behaviour, and behaviour is not correctness — it caught nothing here because it
was generated from the same buggy path it guards. That is a property of the
technique, not a criticism of it, but it means the net wants a correctness pass
alongside: for each fixture, is there an independent reference (a second
strategy, a second sampling method, scikit-learn, a hand computation) that
agrees with it?

---

## Phase 3 — resolve the threshold set from the frame

**Closes:** I3 (medium).

### The defect

Five call sites derive their threshold set from `set(thresholds or y_score)`.
With the new `data=` keyword, `y_score` is a *column name string*, so when
`thresholds` is None this becomes a set of single characters, each compared
against a float column:

```python
rs.metrics.confusion_matrix_at_thresholds("lab", "sc", strategy="loop", data=df)
# TypeError: cannot convert Python type 'str' to Float64

rs.metrics.predicted_positive_ratio_at_thresholds("sc", strategy="loop", data=df)
# TypeError: cannot convert Python type 'str' to Float64
```

It fails loudly rather than silently, which limits the damage — but it is
reachable from the documented API with nothing warning that the two keywords
conflict.

Sites:

| File | Line | Function |
| --- | --- | --- |
| `python/rapidstats/metrics.py` | 508 | `predicted_positive_ratio_at_thresholds` |
| `python/rapidstats/metrics.py` | 757 | `adverse_impact_ratio_at_thresholds` |
| `python/rapidstats/metrics.py` | 1126 | `confusion_matrix_at_thresholds` |
| `python/rapidstats/_bootstrap.py` | 773 | `Bootstrap.confusion_matrix_at_thresholds` |
| `python/rapidstats/_bootstrap.py` | 1242 | `Bootstrap.adverse_impact_ratio_at_thresholds` |

### The change

Add one helper in `_utils.py` and use it at all five sites:

```python
def _resolve_thresholds(thresholds, df: pl.DataFrame, column: str) -> set:
    """The thresholds to evaluate, read from the frame rather than the argument.

    The caller's `y_score` may be a column name when `data=` is given, so the raw
    argument cannot be iterated. Reading the already-materialised frame is also
    strictly better without `data=`: it deduplicates, and it drops nulls along with
    the rest of the frame instead of carrying them into the loop.
    """
    if thresholds is not None:
        return set(thresholds)

    return set(df[column].unique().to_list())
```

This is an improvement independent of `data=`. It stops the `loop` and `cum_sum`
strategies deriving their threshold sets from different sources — `cum_sum`
already reads the frame.

Watch the column name at each site: `metrics.py:1126` and `_bootstrap.py:773`
have already renamed `y_score` to `threshold`.

### Test first

Add a `data=` case per entry point under `strategy="loop"` to
`tests/test_lazy_inputs.py`. Each should currently raise `TypeError`.

### Acceptance

- All five entry points work with `data=` under both strategies, with and
  without an explicit `thresholds` list.
- `loop` and `cum_sum` return the same threshold set for the same input.

---

## Phase 4 — fix the dependency and concurrency contracts

**Closes:** I4, I5, I6 (medium, medium, low).

Three independent small fixes; they group because none is worth its own commit.

### 4a. Narwhals floor (I4)

Two `.pipe(_collect)` sites in `selection.py` operate on **narwhals** LazyFrames,
not polars ones, and `_collect` passes `engine=`. Narwhals only grew
`collect(backend=None, **kwargs)` between 1.20 and 1.30; before that the
signature is `collect(self)`. Installing the floor the project declares breaks
`CFE.fit_from_correlation_matrix`.

```
narwhals 1.0.0   collect(self) -> 'DataFrame[Any]'
narwhals 1.20.0  collect(self) -> 'DataFrame[Any]'
narwhals 1.30.0  collect(self, backend=None, **kwargs) -> 'DataFrame[Any]'

# narwhals==1.20.0, polars==1.33.0
nw.from_native(pl.LazyFrame({'a': [1, 2]})).collect(engine='in-memory')
TypeError: LazyFrame.collect() got an unexpected keyword argument 'engine'
```

Sites: `python/rapidstats/selection.py:729`, `:741` — via `_utils.py:145`.

Preferred fix — make `_collect` defensive, since the engine setting is
meaningless for a non-polars backend and passing it there was never the intent:

```diff
 def _collect(lf, **kwargs):
     from ._config import Config
 
+    # Only polars takes an engine. `selection.py` pipes narwhals frames through here,
+    # and narwhals below ~1.21 has no `**kwargs` on `collect` at all.
+    if not isinstance(lf, pl.LazyFrame):
+        return lf.collect(**kwargs)
+
     return lf.collect(engine=Config.get_engine(), **kwargs)
```

Alternative: raise the declared floor to `narwhals>=1.30.0` in `pyproject.toml`.
Worth doing anyway if you have no reason to keep 1.0 support — but the
defensive `_collect` is what keeps the two frame libraries from leaking into
each other.

Add a narwhals-floor assertion to `tests/test_compat.py`, which already checks
the polars bound.

### 4b. Config scope (I5)

`Config.engine(...)` mutates a module-level global at `_config.py:34`, so every
other thread sees the change for the duration of the block — in a library whose
own default execution model is `_run_concurrent(..., executor="threads")`.

```
thread A:  with rs.Config.engine("streaming"): sleep(0.2)
main:      rs.Config.get_engine()  ->  'streaming'   # leaked
after join:                            'in-memory'
```

Swap the global for a `contextvars.ContextVar` — what polars uses for its own
config, and a near drop-in here:

```python
_engine: contextvars.ContextVar[Engine] = contextvars.ContextVar(
    "rapidstats_engine", default=_DEFAULT_ENGINE
)
```

`get_engine` reads `.get()`, `set_engine` calls `.set()`, and `engine()` uses
`token = _engine.set(...)` / `_engine.reset(token)` so nesting restores
correctly.

One caveat to verify rather than assume: `ThreadPoolExecutor` does *not*
propagate the caller's context to worker threads. If any collect inside a
`_run_concurrent` worker needs to see a caller-set engine, capture
`Config.get_engine()` before fanning out and pass it down. Add a test either way
— one asserting a second thread does not observe the first thread's context
manager, and one asserting the library's own fan-out still honours a
caller-set engine.

### 4c. Quiet (I6)

Two bare `tqdm(...)` calls never received the new flag.

```python
# _bootstrap.py:773 and :1242
for t in tqdm(set(thresholds or y_score), disable=self.quiet):
```

(Phase 3 rewrites these same lines — do the two together.)

Widen `test_quiet_suppresses_progress_bars` in `tests/test_job_control.py` to
cover the loop strategies; it currently only covers `run`.

### Acceptance

- The suite passes against the declared narwhals floor as well as current.
- A thread-isolation test for `Config` passes.
- `quiet=True` produces empty stderr on the loop strategies.

---

## Phase 5 — make cum_sum BCa affordable, or bound it

**Closes:** I2 (high).

### The defect

Removing the `NotImplementedError` exposed a jackknife that builds one full
threshold curve per row and concatenates all of them: n rows × m thresholds × 27
metrics. Measured with default thresholds and only 20 bootstrap iterations:

| n rows | Wall clock | Peak RSS | Growth |
| ---: | ---: | ---: | ---: |
| 200 | 0.96 s | — | — |
| 400 | 2.00 s | — | 2.1× |
| 800 | 5.45 s | 2.2 GB | 2.7× |
| 1,600 | 26.34 s | 6.7 GB | 4.8× |

Extrapolating, a realistic n = 10,000 run exhausts memory rather than finishing.
A user who reads "BCa is supported now" and points it at a real dataset gets an
OOM, not a slow result.

The closed-form O(n) jackknife added in `143192c`
(`src/metrics.rs:44`, `jacknife_confusion_matrix`) covers only the Rust scalar
confusion matrix — even though that commit's own message cites this path as its
motivation: *"This matters now because Tranche 3 made BCa reachable on the
cum_sum paths, so its cost became visible."*

Sites: `_bootstrap.py:891` (confusion matrix), `:1345` (AIR), via `_jacknife`
at `:158`.

### Option A — extend the closed form (preferred, own commit)

The threshold curve is a cumulative weighted bincount. A leave-one-out replicate
differs from the full curve only at thresholds at or below the removed row's
score — everything above it is untouched. That is the same insight already
applied to the scalar confusion matrix, and it turns this path from O(n²m) into
something linear in n.

This is real work and deserves its own commit with its own benchmark table.
The correctness argument is the same one `143192c` used: every characterization
fixture must be unchanged, since the closed form is meant to be exact.

### Option B — bound it (cheap stopgap)

Compute the jackknife frame size up front and refuse above a threshold, with an
error a user can act on:

```python
projected = df.height * len(thresholds) * len(metrics)
if projected > _BCA_CUMSUM_MAX_CELLS:
    raise ValueError(
        f"BCa on the cum_sum strategy materialises "
        f"{df.height} x {len(thresholds)} x {len(metrics)} = {projected:,} rows. "
        f"Pass a shorter explicit `thresholds` list, or use method='percentile'."
    )
```

A refusal beats an OOM. If you take this route, say so in the docstring — the
current one just says BCa is supported.

### Either way

- The docstring should state what BCa costs on these paths.
- Add a `slow`-marked scaling test beside `test_bca_jackknife_scales_near_linearly`,
  which today only covers the scalar path that was actually optimised.

---

## Phase 6 — close the release and documentation gap

**Closes:** I7, I8, I10 (all low, but user-facing).

### 6a. Version

New docstrings say *"Added in version 0.5.0"* while `pyproject.toml` still reads
`0.4.1`. Bump `pyproject.toml` and `Cargo.toml` to `0.5.0` — six behaviour
changes across the PR are marked BREAKING, so this is a minor bump at minimum.

### 6b. Migration note

Collect every breaking change in one place. Scattered across four commit
messages today:

| Change | Commit | Effect |
| --- | --- | --- |
| Quantile definition | `de8a06f` | cum_sum interval bounds shift slightly |
| Standard interval centring | `de8a06f` | `method="standard"` bounds change |
| `batch_size` fraction semantics | `de8a06f` | batching now actually batches |
| Threshold row order | `83bfc89` | output now ascending, was caller order |
| Null-on-empty returns | `7a2341a` | metrics return `None`, not NaN |
| Bootstrap null bounds | `7a2341a` | `(nan, nan, nan)` → `(None, point, None)` |

The last two will bite hardest: `math.isnan(result)` now needs
`result is None` alongside it. The commit message says so — but only the commit
message.

Threshold row order (I9) is not in any commit message at all. It came in with the
as-of join in `83bfc89`: `_map_to_thresholds` sorts its target frame before
joining, so `thresholds=[0.35, 0.15, 0.25]` now returns rows ordered
`0.15, 0.25, 0.35`. Harmless in itself, but it breaks downstream
`assert_frame_equal` and positional indexing.

### 6c. Docs

- `rapidstats.Config` is newly public but has no `docs/` page, so it renders
  nowhere. Add `docs/config.md` containing `::: rapidstats._config` and a nav
  entry in `mkdocs.yml`.
- The README mentions none of `data=`, `lazy=`, `Config` or `resample_mode`.
- State which `Bootstrap` methods accept `data=`.

### 6d. Fill the gaps the docs would have to admit to

- **I8** — `Bootstrap.average_precision`, `.mean`, `.adverse_impact_ratio` and
  `.adverse_impact_ratio_at_thresholds` take no `data=`, so a caller working
  from a scan falls back to arrays for those four. Adding it is cheap and
  better than documenting the hole.
- **I7** — the `resample_mode` docstring says only that `Bootstrap.run` always
  materialises. In fact the cum_sum threshold bootstraps and `average_precision`
  also always materialise, because they resample in polars rather than through
  the Rust kernel. Not wrong, but a user setting `resample_mode="weights"` for
  speed gets none of the documented 6–12× on the paths that were slowest to
  begin with.

---

## Phase 7 — make CI test what the PR claims

**Closes:** L2, L5, L6.

### 7a. Polars matrix

Four commits report *"verified on polars 1.33.0 and 1.44.0"*, but
`.github/workflows/tests.yaml` has no polars axis — the range the PR just
bounded is tested at exactly one version, whatever resolves that day. Add the
declared floor and the current release.

### 7b. Path filter

The workflow's `paths:` filter covers `src/`, `tests/` and `python/` but not
`pyproject.toml`, `Cargo.toml` or `uv.lock`. A dependency-bound change does not
trigger the suite at all — which is how a broken narwhals floor (I4) reaches
`main` unnoticed.

### 7c. Python version

The matrix still lists Python 3.9 and `[tool.tox] env_list` still includes
`py39`, though `requires-python` moved to `>=3.10` in #26.

### 7d. Brittle tests (L5)

- `test_bca_interval` — reimplement the BCa reference inline rather than
  importing `scipy.stats._resampling._bca_interval`. It is ~15 lines and stops
  the suite tracking a private signature.
- `test_polars.py::test_auc` — `np.trapz` → `np.trapezoid`.

### 7e. Shared RNG fixture (L6)

`tests/conftest.py` exposes a session-scoped `np.random.Generator`. Any two
tests drawing from it are coupled through its state, so results depend on
collection order and on which tests are deselected. Make it function-scoped, or
drop it in favour of the per-fixture `RandomState` pattern used by
`binary_data` and `regression_data` beside it, which is built correctly.

---

## Appendix: findings index

Severity reflects blast radius on a released version, not effort to fix.
"Reproduced" means verified against a local build of `7a2341a`, not inferred
from reading.

### Introduced by this PR

| ID | Severity | Finding | Location | Phase |
| --- | --- | --- | --- | --- |
| I1 | High | Poisson + BCa + cum_sum confusion matrix raises `AttributeError` | `_bootstrap.py:891` | 1 |
| I2 | High | BCa on cum_sum paths is quadratic in time and memory | `_bootstrap.py:891`, `:1345` | 5 |
| I3 | Medium | `data=` + `strategy="loop"` iterates the column name's characters | 5 sites | 3 |
| I4 | Medium | Declared `narwhals>=1.0.0` floor no longer works | `selection.py:729`, `:741` | 4a |
| I5 | Medium | `Config` is a module global, not thread-local | `_config.py:34` | 4b |
| I6 | Low | `quiet=True` does not silence loop-strategy progress bars | `_bootstrap.py:773`, `:1242` | 4c |
| I7 | Low | `resample_mode` silently ignored on three paths | `_bootstrap.py:794`, `:1264`, `:1007` | 6d |
| I8 | Low | `data=` surface is uneven — four `Bootstrap` methods lack it | `_bootstrap.py` | 6d |
| I9 | Low | Threshold output order changed to ascending, undocumented | `metrics.py:999` | 6b |
| I10 | Low | Version, changelog and docs do not match the code | `pyproject.toml`, `docs/` | 6 |
| I11 | Low | A fresh rayon pool is built per call when `n_jobs` is set | `src/bootstrap.rs:325` | — |

I11 is left unscheduled: `run_bootstrap` memoises the default pool in a
`OnceLock`, but the explicit-`n_jobs` branch calls `create_rayon_pool(n_jobs)`
on every invocation, spawning and tearing down n threads with 16 MiB stacks each
time. Only noticeable for callers who set `n_jobs` and bootstrap in a loop. Fix
by caching per `n_jobs` value if it ever shows up in a profile.

### Pre-existing, not fixed by this PR

| ID | Severity | Finding | Location | Phase |
| --- | --- | --- | --- | --- |
| L1 | High | Poisson AIR-at-thresholds computes on an unsorted frame | `_bootstrap.py:1264` | 2 |
| L2 | Medium | CI does not enforce any of the compatibility claims | `.github/workflows/tests.yaml` | 7 |
| L3 | Medium | `norm_cdf` is accurate to ~1.5e-7 | `src/distributions.rs:68` | — |
| L4 | Low | `Bootstrap.average_precision` ignores `sampling_method` | `_bootstrap.py:1007` | — |
| L5 | Low | Two tests pinned to private or removed APIs | `test_bootstrap.py`, `test_polars.py` | 7d |
| L6 | Low | Shared `rng` fixture is session-scoped and mutable | `tests/conftest.py` | 7e |

**L3** — `erf` uses Abramowitz & Stegun 7.1.26, whose absolute error bound is
1.5e-7. That bound now propagates into more places: BCa's bias-correction and
acceleration probabilities, on paths this PR just made reachable, and through
the new `pl_norm_cdf` plugin expression. It sits far below the Monte Carlo error
of a 1000-iteration bootstrap, so it is a documentation item rather than a
numerical one — but note that the characterization net asserts to 1e-12 and
would go red on any future switch to a more accurate `erf`.

**L4** — `Bootstrap.average_precision` (`_bootstrap.py:961`) calls
`df.sample(fraction=1, with_replacement=True)` unconditionally, so
`sampling_method="poisson"` is silently discarded. It also does not route its
replicates through `_usable` the way `run` does, and falls off the end returning
`None` if `method` is somehow invalid. Worth folding into Phase 6d if you are
already adding `data=` to it.

---

## What the pattern was

Both introduced high-severity findings live where two configuration axes cross.
`sampling_method` × `resample_mode` × `strategy` × `method` multiply out to
combinations the new tests exercise one axis at a time — thorough per feature,
thin across features. Parametrizing `test_bca_cum_sum.py` over
`sampling_method` alone would have caught I1 outright.

Worth applying that lens to whatever lands next: for any new keyword, which
existing keyword does it multiply against, and is that product tested?
