# Score, rank and inclusion ledger

SetGuard 0.2 adds an inspectable arithmetic layer around the unchanged original split/classwise conformal implementation. This is an independent educational implementation, not a new statistical method or a deployment guarantee.

## Run

```bash
python -m unittest discover -s tests -v
node --test tests/*.cjs   # Node 22 and Google Chrome required
bash examples/verify_wheel.sh
python -m setguard rank-demo
python -m setguard audit --input audit-input.json --alpha .1
```

An audit input has exactly two fields, `calibration` and `candidates`. Calibration contains `class_names`, normalized `probabilities`, and aligned integer `labels`. Candidates contain the same names/order and probabilities, but no labels. The ledger is bounded to 256 calibration rows, 32 candidate rows and 2-16 classes; the original CLI/core retain their original contract. The new audit command rejects duplicate JSON keys. Standard output is JSON; invalid input exits 2 with diagnostics on standard error.

The static page lives under `docs/`. It runs locally without a server, account, API, telemetry, network dependency or model inference. It exposes five fixed public synthetic development cases, not arbitrary uploaded data: original generator, repeated threshold ties, absent/rare labels, possible empty candidate sets, and one calibration row. The original case reuses 200 calibration rows from seed 11 and six unlabeled candidates from seed 29. No candidate labels are fitted or optimized.

## The arithmetic

For a fixed predictor, score `s(x,y)=1-p(y|x)`. With `n` calibration scores and miscoverage target `alpha`, the corrected one-based rank is `ceil((n+1)*(1-alpha))`. A rank exceeding `n` means infinity, serialized as JSON `null`; it is not missing data. A label is included when its score is at most the threshold, so all equality ties are included. Stable sorting by original row index only determines display order, never which ties survive.

Python uses the unchanged core's `Fraction(str(alpha))` corrected rank. The independent JavaScript implementation parses the decimal numeric representation into a BigInt rational. Exact decimal rank arithmetic does not make binary64 probabilities exact real numbers, and numerical equality is not a forward-error certificate. Summary numbers are rounded; tables/exports retain binary64 values.

Pooled calibration uses every calibration score. Classwise calibration uses only rows with that true label. An absent or too-small class produces an infinity threshold, without silently falling back to pooled calibration. Pooled and classwise sets may differ, and empty sets are possible.

The selected group's deletion ledger holds out each score position in turn and calculates its rank, threshold and inclusion using the remaining positions. Fewer than two positions makes the check unavailable. This is a deterministic rank-logic count, not an uncertainty interval, exchangeability test, test-set score or coverage claim for the current candidate.

The selected candidate's path uses exactly 19 alpha values from 0.05 through 0.95. Points are not joined or interpolated. It is descriptive, not an alpha-selection procedure or an optimized set.

## Original controls remain

`core.py`, `demo.py`, original tests, example data and DESIGN.md are unchanged. The three original 2,000-row empirical arms remain attached to the page and package:

| Arm | Coverage | Mean set size | Argmax accuracy |
| --- | ---: | ---: | ---: |
| Pooled, same generator | .9235 |3.017 |.4960 |
| Pooled, adversarial label shift | .0760 |3.017 |0 |
| Classwise, same generator | .9285 |3.1095 |.4960 |

The adverse arm reuses probabilities and assigns the least-probable class as truth. It is an intentional failure control, not a realistic estimate of deployment shift. The original pooled calibration has `n=200`, alpha 0.1, rank 181, threshold 0.9936492756846956. Class counts are [43,37,50,35,35]; class d's realized coverage 0.8859 is below 0.9. Neither one success nor one realized class rate establishes or refutes the population theorem. The original separate 21-position rank audit remains unchanged.

## Validation and reproduction

41 Python tests include the original 31 tests, 160 independent Fraction rank games, exact ties, absent/rare classes, empty/all-label sets, the 49-row alpha 0.58 rank 21 boundary, complete archive comparison, CLI/duplicate-key rejection and strict archive comparator failures. Four Node tests include 80 integer-grid reference games and an actual Chrome test against 95 case/alpha witnesses. Those witnesses contain 4,199 pooled sorted positions, 8,341 pooled/classwise deletion records and 2,736 candidate-label decisions. UI switching checks all 20 candidate rows and 22 groups across the five cases, set texts, reasons, exact sorted tables, threshold lines, boundary markers and 19-point paths. Invalid alpha clears stale charts, JSON and exports. Clipboard denial/success, downloadable JSON, keyboard alpha input and 320/390/1280 pixel layout are tested.

Archive comparison requires exact keys, types, integers, booleans, nulls and lengths. Finite float leaves allow absolute/relative tolerance 1e-12; nonfinite floats fail. `examples/verify_wheel.sh` installs an isolated wheel and reproduces the original demo and all 95 archive witnesses outside the checkout, imports all six modules and executes all six CLI commands. Frozen witnesses ship as metadata plus 19 bounded JSON chunks, not downloaded results. The browser has its own arithmetic implementation; Python outputs are verification data, not executable logic.

## Scope and attribution

The standard guarantee averages over exchangeable calibration and fresh test scores for a predictor fixed independently of calibration. Pooled marginal validity is not a promise for each input, label, fixed calibration split or deployment. Classwise validity requires exchangeability within true label. JSON validation cannot establish those assumptions. General non-exchangeability needs its own assumptions and analysis; this project does not implement that extension, adaptive scores, clustered conformal, randomized tie-breaking or online calibration.

Primary references, consulted November 8, 2026:

- Angelopoulos and Bates, *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*: corrected quantile, inclusive set rule, marginal scope and assumptions. https://arxiv.org/abs/2107.07511
- *Class-Conditional Conformal Prediction With Many Classes*: classwise calibration, rare-class infinity boundaries and distinction from realized fixed-split rates. https://www.stat.berkeley.edu/~ryantibs/papers/classconf.pdf
- Angelopoulos et al., public implementation/tutorial repository, useful context and attribution rather than a dependency. https://github.com/aangelopoulos/conformal-prediction
- Oliveira et al., *Split Conformal Prediction and Non-Exchangeable Data*, JMLR 25(2024): extensions require their stated concentration/decoupling assumptions and penalties. Not implemented here. https://www.jmlr.org/papers/volume25/23-1553/23-1553.pdf
- Romano et al., *Classification with Valid and Adaptive Coverage*: different score construction/adaptive method. Not implemented or claimed here. https://arxiv.org/pdf/2005.06095v3.pdf
- Cauchois, Gupta and Duchi, *Knowing what You Know*, JMLR 22(2021): marginal validity can mask uneven conditional coverage. Their quantile-regression methods are not implemented here. https://www.jmlr.org/papers/volume22/20-753/20-753.pdf

Attribution is not endorsement. MIT implementation; no private data, paid service or external model API.
