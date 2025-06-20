.PHONY: build run stop clean

NETWORK_NAME = myapp-network

# Flask
FLASK_IMAGE = flask-app
FLASK_CONTAINER = flask-app
FLASK_PORT = 5002

# Streamlit
UI_IMAGE = streamlit-ui
UI_CONTAINER = streamlit-ui
UI_PORT = 8502

# Ollama
OLLAMA_IMAGE = ollama/ollama
OLLAMA_CONTAINER = ollama
OLLAMA_PORT = 11434

build:
	docker build -t $(FLASK_IMAGE) ./flask_app
	docker build -t $(UI_IMAGE) ./streamlit_ui

ollama-cpu:
	# Run Ollama (CPU)
	docker run -d \
		--name $(OLLAMA_CONTAINER) \
		--network $(NETWORK_NAME) \
		-v ollama:/root/.ollama \
		-p $(OLLAMA_PORT):$(OLLAMA_PORT) \
		$(OLLAMA_IMAGE)

ollama-gpu:
	# Run Ollama (GPU)
	docker run -d \
		--gpus all \
		--name $(OLLAMA_CONTAINER) \
		--network $(NETWORK_NAME) \
		-v ollama:/root/.ollama \
		-p $(OLLAMA_PORT):$(OLLAMA_PORT) \
		$(OLLAMA_IMAGE)

ollama-list:
	# List available models in Ollama
	docker exec $(OLLAMA_CONTAINER) ollama list
	#docker exec -it $(OLLAMA_CONTAINER) ollama list

run:
	docker network create $(NETWORK_NAME) || true
	-docker rm -f $(OLLAMA_CONTAINER) $(FLASK_CONTAINER) $(UI_CONTAINER)
	#docker run -d --gpus all --name $(OLLAMA_CONTAINER) --network $(NETWORK_NAME) -v ollama:/root/.ollama -p $(OLLAMA_PORT):11434 $(OLLAMA_IMAGE)
	docker run -d --name $(FLASK_CONTAINER) --network $(NETWORK_NAME) -p $(FLASK_PORT):$(FLASK_PORT) $(FLASK_IMAGE)
	docker run -d --name $(UI_CONTAINER) --network $(NETWORK_NAME) -p $(UI_PORT):8501 $(UI_IMAGE)

stop:
	@echo "Stopping containers..."
	-docker stop $(OLLAMA_CONTAINER)
	-docker stop $(FLASK_CONTAINER)
	-docker stop $(UI_CONTAINER)

	@echo "Removing containers..."
	#-docker rm $(OLLAMA_CONTAINER)
	-docker rm $(FLASK_CONTAINER)
	-docker rm $(UI_CONTAINER)

clean:
	#docker rm -f $(OLLAMA_CONTAINER) $(FLASK_CONTAINER) $(UI_CONTAINER)
	docker rmi $(OLLAMA_IMAGE) $(FLASK_IMAGE) $(UI_IMAGE) || true
	docker network rm $(NETWORK_NAME) || true
