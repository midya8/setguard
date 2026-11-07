# SetGuard

Split-conformal classification sets from scratch. Give it a held-out calibration
sample and a fixed classifier's probabilities; get a versioned JSON artifact,
unlabeled prediction sets and coverage/size diagnostics. Pure Python 3.10+,
no dependencies, no API keys, no model downloads.

## Measured development demonstration

Run `python -m setguard demo` or `bash examples/demo.sh`.

| Arm | Rows | Empirical coverage | Mean set size | Argmax accuracy |
| --- | ---: | ---: | ---: | ---: |
| Same generator, independent seed | 2000 | 0.9235 | 3.017 | 0.4960 |
| Deliberately adversarial label shift | 2000 | 0.0760 | 3.017 | 0.0000 |

The calibration split has 200 rows, alpha=0.1, corrected rank 181, and
threshold 0.9936492756846956. Five classes; the fixed predictor cubes an
underlying probability vector, making it overconfident. There is no training.
Coverage costs larger sets: argmax emits one class; SetGuard averages 3.017.
The shift arm keeps probabilities unchanged but assigns the least-probable
class as truth. It is an intentional failure demonstration, not a realistic
drift estimate. Fixtures and seeds are development choices, not a final benchmark.

## What the guarantee means

For a predictor/score fixed independently of calibration, exchangeable
calibration and fresh test scores give **marginal** coverage at least
`1-alpha`, averaged over calibration and test randomness. This is a statement
about the procedure, not a promise that every realized test split achieves
the target. It does not give coverage for every class, every input, fixed
calibration dataset, time series or distribution shift. Ties may be conservative.
No randomized upper coverage bound is claimed.

The exact finite rank check holds each of 21 distinct score positions out
once: 19/21 covered, equal to the rank lower bound 19/21. Tests also exhaust
all 243 five-position score sequences over {0, 0.5, 1} at five alpha levels.
This checks the implementation's rank logic, not whether real data are exchangeable.

## Fit, save, predict without labels

```bash
python -m setguard fit --input examples/calibration.json --alpha 0.2 > model.json
python -m setguard predict --model model.json --input examples/unlabeled.json
python -m setguard evaluate --model model.json --input examples/calibration.json
```

The final command above only demonstrates evaluation syntax on the tiny example;
reusing calibration rows is not a valid held-out coverage assessment.
Use a new labeled test split in real work. `--input -` reads stdin.
Errors exit 2, write stderr and emit no partial JSON.

Calibration/evaluation schema:

```json
{"class_names": ["cat", "dog"], "probabilities": [[0.9, 0.1], [0.2, 0.8]], "labels": [0, 1]}
```

Prediction input omits `labels` entirely. Class names and order must match the
artifact. Rows must be finite in [0,1] and sum to one within 1e-9; they are
not silently normalized. Schema checks cannot verify independence, provenance,
a fixed predictor or semantic label correctness.

```python
from setguard import fit, predict, evaluate
model = fit(calibration, alpha=0.1)
sets = predict(model, unlabeled)  # lists of integer class indices
metrics = evaluate(model, held_out_test)
```

Artifacts contain version/method, alpha, calibration count, corrected rank,
class names and threshold. A null threshold denotes the all-label boundary
when the requested rank exceeds the calibration count. Empty sets are allowed:
no hidden argmax fallback changes the algorithm. Evaluation includes per-class
coverage (null for absent classes), size histograms, empty and singleton rates.

## Rank correction

Score: `s(x,y)=1-p(y|x)`. Let `k=ceil((n+1)*(1-alpha))`.
Use the k-th smallest calibration score, not an interpolated quantile.
If k=n+1, include all labels. Otherwise include y exactly when `1-p(y|x)<=q`.
The ceiling uses rational arithmetic for the decimal spelling of alpha to
avoid accidental extra ranks at float boundaries such as n=99, alpha=0.7.

## Verification

```bash
python -m unittest discover -s tests -v
bash examples/demo.sh
```

20 tests: exact rank boundaries, ties, all-label fallback, empty/nested sets,
artifact round trips, class ordering, invalid inputs, per-class reporting,
exhaustive tied-score rank audit, CLI/error behavior and pinned fixture results.
CI runs tests on Python 3.10-3.13 and the offline demo. The demo checks both a
passing same-generator gate and a failing shifted-coverage gate.

## Attribution and limits

Algorithm reference: Anastasios N. Angelopoulos and Stephen Bates,
[A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/html/2107.07511), Section 1 and Appendix D.
Boundary/tie rank formulation: [finite-sample marginal coverage statement and proof](https://androma.org/theorems/17354).
This repository is an original small implementation of the standard method,
not a new conformal algorithm and not copied from the authors' code.

No APS/RAPS, group-conditional guarantee, adaptive online calibration,
regression intervals, leakage detection, drift alarm or deployment validation.
See [DESIGN.md](DESIGN.md). MIT licensed.
