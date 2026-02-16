DOCKER_COMPOSE ?= docker compose

.PHONY: demo dashboard showcase site demo-gif

demo:
	$(DOCKER_COMPOSE) run --rm runner

dashboard:
	$(DOCKER_COMPOSE) run --rm runner
	$(DOCKER_COMPOSE) run --rm dashboard

showcase:
	python scripts/generate_showcase_assets.py --root .

site:
	python scripts/build_site.py --root .

demo-gif:
	bash scripts/demo/make_demo_gif.sh
