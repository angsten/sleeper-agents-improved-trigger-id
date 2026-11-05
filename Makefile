# Every target runs from the repository root. RELEASE_HASHES=1 makes the harness check the released files against the
# hashes recorded inside the results (README, "Re-measuring from this repository").
export RELEASE_HASHES := 1

.PHONY: bootstrap test figures margins verify

bootstrap:          ## the Python environment, from uv.lock, and NLTK's punkt tokenizer for the IFEval checks
	uv sync
	uv run python -c "import nltk; nltk.download('punkt', download_dir='.cache/nltk_data', quiet=True)"

test:               ## the test suite; six tests are expected failures (README, "Tests")
	uv run pytest -q

figures:            ## results/figures/figure_1.png, figure_2.png and their CSVs, from the committed results
	uv run --with matplotlib python scripts/make_figures.py

margins:            ## results/derived/per_token_margins.csv, the post's per-token numbers
	uv run python scripts/per_token_margins.py

verify:             ## the provenance check: released files against the manifest and the hashes recorded in the results
	uv run python scripts/verify_provenance.py
