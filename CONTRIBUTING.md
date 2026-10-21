# Contributing

This repository publishes benchmark numbers, so the bar for a change is not only
"does it run" but "can someone else recompute the number it touches". Please
read [docs/RUNNING.md](docs/RUNNING.md) before opening a results PR.

## Reporting a wrong number

Open a **Data correction** issue, not a PR that silently edits a table. Include:
the file and row, the value you see, the value you expected, and the rollup row
in `results/consolidated.csv` (or `results/finbench-consolidated.csv`) that the
number should come from. Numbers in the README are meant to be a rendering of
the tracked rollups — if the two disagree, the rollup wins and the README is
the bug.

## Adding a benchmark cell

1. Add the cell to `matrix.yaml` (`id`, `model`, `quant`, `size_gib`, `arch`,
   `file` as a `~`-relative path). Do not put absolute home paths in tracked
   files; a test enforces the `~`-relative form.
2. If the GGUF is not already in `scripts/download_models.sh`, add it there with
   its URL so the cell is reproducible from a clean machine.
3. Sweep it on both backends (`docs/RUNNING.md` §3) and rebuild the rollups with
   `python3 scripts/run_sweep.py --consolidate`.
4. Regenerate the charts (`python3 scripts/make_charts.py`) and commit them —
   each PNG carries a fingerprint of the rollup it was drawn from and the suite
   fails if a committed chart is stale.
5. Update the README table and prose **and** the tests that pin them
   (`tests/test_readme_ranges.py`, `tests/test_readme_tables.py`).

## Tests

The suite is stdlib `unittest`, and the machine it runs on needs PyYAML plus
matplotlib (`pip install -r requirements.txt`):

```bash
python3 -m unittest discover -s tests -t . -v
```

Use an interpreter that can `import yaml` — on the reference machine that is
`/usr/bin/python3`. If `tools/score_ll` or its fixture GGUF is absent, the
scorer's golden test skips rather than fails; that is expected on a machine
without model weights, but please do not paper over a skip when you touch the
scorer.

CI runs the same command on every pull request.

## Rules of thumb

- Prose numbers are pinned to rollups by tests. If you change a percentage in
  the README, change the test that recomputes it, or the CI will tell you.
- Keep the harness dependency-light: stdlib + PyYAML for the drivers;
  matplotlib only for the chart script.
- Do not commit model weights, raw run directories, or absolute paths from your
  machine. `results/run-*/` and `results/finbench-*/` are gitignored for that
  reason; only the consolidated rollups are tracked.
- One logical change per PR, with the commit message naming what was verified.

## Code of conduct

Be decent. Disagreement about methodology is welcome and should be argued with
measurements; review comments should be about the work, not the person.
