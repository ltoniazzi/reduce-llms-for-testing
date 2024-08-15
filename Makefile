.SILENT:
.PHONY: setup lint run

HF_REPO ?= ltoniazzi/reduce-llms-for-testing

default: run

setup:
	poetry install --with dev && \
	poetry run pre-commit install

lint:
	poetry run pre-commit run --all-files

run: setup
	poetry run python reduce_llms_for_testing/main.py -m google/gemma-2-2b -hf $(HF_REPO) -s 64
	# poetry run python reduce_llms_for_testing/main.py -m microsoft/Phi-3-mini-4k-instruct -hf $(HF_REPO) -s 64
	# poetry run python reduce_llms_for_testing/main.py -m meta-llama/Meta-Llama-3-8B-Instruct -hf $(HF_REPO) -s 64
