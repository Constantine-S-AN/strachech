DOCKER_COMPOSE ?= docker compose

.PHONY: demo dashboard

demo:
	$(DOCKER_COMPOSE) run --rm runner

dashboard:
	$(DOCKER_COMPOSE) run --rm runner
	$(DOCKER_COMPOSE) run --rm dashboard
