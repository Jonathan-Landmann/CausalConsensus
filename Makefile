.PHONY: install test lint demo benchmark paper

install:
	python -m pip install -e ".[experiments,dev]"

test:
	python -m pytest -q

lint:
	python -m ruff check src tests

demo:
	python -m causal_consensus.cli demo

benchmark:
	python -m causal_consensus.cli benchmark --output results/full

paper:
	cd paper && pdflatex manuscript.tex && bibtex manuscript && pdflatex manuscript.tex && pdflatex manuscript.tex

