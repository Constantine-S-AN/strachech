DOCKER_COMPOSE ?= docker compose

.PHONY: demo dashboard showcase

demo:
	$(DOCKER_COMPOSE) run --rm runner

dashboard:
	$(DOCKER_COMPOSE) run --rm runner
	$(DOCKER_COMPOSE) run --rm dashboard

showcase:
	python scripts/generate_showcase_assets.py --root .
