.SILENT:
.PHONY: setup lint run

HF_REPO ?= ltoniazzi/reduce-llms-for-testing

setup:
	poetry install --with dev && \
	poetry run pre-commit install

lint:
	 pre-commit run --all-files

run: setup
	# pytohn reduce_llms_for_testing/main.py -m google/gemma-2-2b -hf $(HF_REPO) -s 64
	pytohn reduce_llms_for_testing/main.py -m meta-llama/Meta-Llama-3-8B-Instruct -hf $(HF_REPO) -s 32
