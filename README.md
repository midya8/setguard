# setguard

Split conformal prediction for classifiers, in plain Python. Instead of one predicted label you get a set of labels, sized so the true label lands in it about 90% of the time (or whatever `alpha` you pick). You need a model that outputs class probabilities and a held-out labeled calibration set; setguard never trains anything.

```bash
python -m setguard fit --input examples/calibration.json --alpha 0.1 > model.json
python -m setguard predict --model model.json --input unlabeled.json
python -m setguard evaluate --model model.json --input test.json
```

`fit` saves the threshold as a small JSON file, and `predict` doesn't need labels. Evaluate on a fresh labeled split, not the calibration rows.

## Demo

`python -m setguard demo` builds a deliberately overconfident 5-class predictor, calibrates on 200 rows at alpha 0.1, and tests on 2,000 more:

| test data | coverage | mean set size | argmax accuracy |
|---|---:|---:|---:|
| same distribution | 0.924 | 3.02 | 0.496 |
| labels deliberately flipped to the least likely class | 0.076 | 3.02 | 0.000 |

So you get the coverage you asked for, at the price of sets around three labels wide when the model is this bad. The second row is the failure case on purpose: the guarantee only holds if test data looks like calibration data.

## Per-class thresholds

`--mode classwise` fits one threshold per class using only that class's calibration rows. On the same demo it reaches 0.929 coverage with mean size 3.11, but class d still came in at 0.886, under the 0.9 target. That's normal for small class counts (35 to 50 rows each here). A class with too few rows gets no threshold and is always included in the set.

## What the guarantee is

If calibration and test examples are exchangeable and the model was fixed beforehand, coverage is at least 1 - alpha on average over random splits. It isn't per input, per class (unless you use classwise), or protection against drift.

The browser page at https://midya8.github.io/setguard/ shows each score, the rank used for the threshold, and why each label made it into a set. LEDGER.md and DESIGN.md have the math.

Tests: `python -m unittest discover -s tests` (44 tests).
