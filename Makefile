# Everything offline uses committed artifacts; no API key, judge or network needed except `data`.
.PHONY: test lint data reproduce review

test:  ## unit + integration tests (offline)
	uv run pytest -q

lint:  ## lint and formatting check
	uv run ruff check . && uv run ruff format --check .

data:  ## download the TyDi QA dev file (~232 MB) and verify its checksum
	uv run python download_data.py

reproduce: test lint  ## regenerate reports/analysis.* from the committed run artifacts
	uv run python analyze.py > /dev/null
	git diff --exit-code reports/ && echo "reports/ regenerated identically"

# The steps below need OPENAI_API_KEY in .env and the local judge (see .env.example).
# They are how the committed artifacts were produced, not part of `reproduce`.
#   uv run python run_pipeline.py --split held_out --prompt v1-baseline
#   uv run python run_pipeline.py --split held_out --prompt v2-not-in-context
#   uv run python score_run.py results/<run_id> --max-minutes 600
#   uv run python known_truth.py build && uv run python known_truth.py score
