.PHONY: init
init: uv/venv uv/sync
	cat .env.development.local.sample > .env.development.local

.PHONY: uv/venv
uv/venv:
	uv venv --clear --seed

.PHONY: uv/sync
uv/sync:
	uv sync
