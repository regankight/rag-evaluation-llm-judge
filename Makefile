# Everything offline uses committed artifacts; no API key, judge or network needed except `data`.
.PHONY: test lint data reproduce gate-demo

V1 := results/held_out-v1-baseline-20261006T064508Z
V2 := results/held_out-v2-not-in-context-20261006T064838Z

test:  ## unit + integration tests (offline)
	uv run pytest -q

lint:  ## lint and formatting check
	uv run ruff check . && uv run ruff format --check .

data:  ## download the TyDi QA dev file (~232 MB) and verify its checksum
	uv run python download_data.py

gate-demo:  ## the regression gate on the committed runs: adopting v2 passes, reverting to v1 fails
	uv run python eval_gate.py $(V1) $(V2)
	! uv run python eval_gate.py $(V2) $(V1) > /dev/null && echo "reverting the prompt is blocked, as expected"

reproduce: test lint gate-demo  ## regenerate reports/ (analysis + grounding) from the committed artifacts
	uv run python analyze.py > /dev/null
	uv run python grounding.py > /dev/null
	git diff --exit-code reports/ && echo "reports/ regenerated identically"

# The steps below need OPENAI_API_KEY in .env and the local judge (see .env.example).
# They are how the committed artifacts were produced, not part of `reproduce`.
#   uv run python run_pipeline.py --split held_out --prompt v1-baseline
#   uv run python run_pipeline.py --split held_out --prompt v2-not-in-context
#   uv run python score_run.py results/<run_id> --max-minutes 600
#   uv run python known_truth.py build && uv run python known_truth.py score
