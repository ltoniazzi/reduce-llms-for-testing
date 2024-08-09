.SILENT:
.PHONY:

setup:
	poetry install --with dev && \
	poetry run pre-commit install

lint:
	 pre-commit run --all-files
