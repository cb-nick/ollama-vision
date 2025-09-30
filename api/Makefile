# API Docker targets
API_IMAGE = ollama-vision-api
API_CONTAINER = ollama-vision-api

.PHONY: api-build api-run api-stop api-clean

api-build:
	docker build -t $(API_IMAGE) ./

api-run:
	docker run --rm -d --add-host=host.docker.internal:host-gateway --name $(API_CONTAINER) -p 8505:8505 $(API_IMAGE)

api-stop:
	-docker stop $(API_CONTAINER) || true

api-clean: api-stop
	-docker rm $(API_CONTAINER) || true
	docker rmi $(API_IMAGE) || true