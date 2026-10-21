## What this changes

<!-- One or two sentences. If it touches a published number, say which. -->

## Evidence

- [ ] `python3 -m unittest discover -s tests -t . -v` passes locally
- [ ] If a README number changed: the rollup row it comes from is quoted here
- [ ] If charts changed: regenerated with `scripts/make_charts.py` (the suite
      checks each PNG's fingerprint against `results/consolidated.csv`)
- [ ] No model weights, raw `results/run-*/` directories, or absolute machine
      paths added

## Notes for the reviewer

<!-- Anything surprising, or deliberately left out of scope. -->
