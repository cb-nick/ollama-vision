.PHONY: build run stop clean test lint format

NETWORK_NAME = myapp-network

# Streamlit
UI_IMAGE = vision-demo
UI_CONTAINER = vision-demo
UI_PORT = 8504

# Virtual environment
VENV_DIR = /home/nick/vision-demo/ollama-vision/streamlit_ui/.setup-env
VENV_PYTHON = $(VENV_DIR)/bin/python3
VENV_UVICORN = $(VENV_DIR)/bin/uvicorn

# Project directories
PROJECT_DIR = /home/nick/vision-demo/ollama-vision
STREAMLIT_DIR = $(PROJECT_DIR)/streamlit_ui
REFACTORED_DIR = $(PROJECT_DIR)/refactored

run-search:
	@# Check if model_server.py is running on localhost:8000, if not start it
	@if ! nc -z localhost 8000; then \
		echo "Starting model_server.py..."; \
		( \
            cd $(STREAMLIT_DIR) && \
            nohup $(VENV_UVICORN) model_server:app --host 0.0.0.0 --port 8000 > model_server.log 2>&1 & \
        ); \
		echo "Waiting for model_server.py to start..."; \
        for i in $$(seq 1 200); do \
            if nc -z localhost 8000; then \
                echo "model_server.py is now running."; \
                break; \
            fi; \
			echo -n "."; \
            sleep 0.5; \
        done; \
	else \
		echo "model_server.py is already running on port 8000."; \
	fi

	@echo "Running semantic search demo..."
	@$(VENV_PYTHON) $(PROJECT_DIR)/semantic_search_demo/semantic_search.py

stop-search:
	@echo "Stopping model_server.py..."
	@pkill -f "uvicorn model_server:app" || true

	@echo "Stopping semantic search demo..."
	@pkill -f "semantic_search.py" || true

	@echo "Done."

validate-structure:
	@echo "Validating project structure..."
	@if [ ! -d "$(REFACTORED_DIR)/services" ]; then \
		echo "ERROR: services/ directory not found in $(REFACTORED_DIR)"; \
		exit 1; \
	fi
	@if [ ! -d "$(REFACTORED_DIR)/ui" ]; then \
		echo "ERROR: ui/ directory not found in $(REFACTORED_DIR)"; \
		exit 1; \
	fi
	@if [ ! -d "$(REFACTORED_DIR)/utils" ]; then \
		echo "ERROR: utils/ directory not found in $(REFACTORED_DIR)"; \
		exit 1; \
	fi
	@if [ ! -f "$(REFACTORED_DIR)/main.py" ]; then \
		echo "ERROR: main.py not found in $(REFACTORED_DIR)"; \
		exit 1; \
	fi
	@if [ ! -f "$(REFACTORED_DIR)/config.py" ]; then \
		echo "ERROR: config.py not found in $(REFACTORED_DIR)"; \
		exit 1; \
	fi
	@echo "Project structure validation passed!"

build: validate-structure
	@# Check if model_server.py is running on localhost:8000, if not start it
	@if ! nc -z localhost 8000; then \
		echo "Starting model_server.py..."; \
		( \
            cd $(STREAMLIT_DIR) && \
            nohup $(VENV_UVICORN) model_server:app --host 0.0.0.0 --port 8000 > model_server.log 2>&1 & \
        ); \
		echo "Waiting for model_server.py to start..."; \
        for i in $$(seq 1 200); do \
            if nc -z localhost 8000; then \
                echo "model_server.py is now running."; \
                break; \
            fi; \
			echo -n "."; \
            sleep 0.5; \
        done; \
	else \
		echo "model_server.py is already running on port 8000."; \
	fi
	@echo "Building Docker image..."
	@docker build -t $(UI_IMAGE) ./refactored

run:
	@echo "Starting Streamlit UI container..."

	@echo
	@echo "Creating Docker network if it doesn't exist..."
	@docker network create $(NETWORK_NAME) 2>/dev/null || true

	@echo
	@echo "Cleaning up any existing containers..."
	@docker rm -f $(UI_CONTAINER) 2>/dev/null || true

	@echo
	@echo "Running Streamlit UI container..."
	@docker run -d \
		-v $(PROJECT_DIR)/logs:/app/logs \
		--name $(UI_CONTAINER) \
		--add-host=host.docker.internal:host-gateway \
		--network $(NETWORK_NAME) \
		-p $(UI_PORT):8501 \
		$(UI_IMAGE) 2>/dev/null || true

	@echo
	@echo "Streamlit UI is running at http://localhost:$(UI_PORT)"

stop:
	@echo "Stopping containers..."
	@docker stop $(UI_CONTAINER) 2>/dev/null || true

	@echo
	@echo "Removing containers..."
	@docker rm $(UI_CONTAINER) 2>/dev/null || true

	@echo
	@echo "Stopping model_server.py..."
	@pkill -f "uvicorn model_server:app" || true

clean:
	@echo "Cleaning up Docker images and network..."
	@docker rmi $(UI_IMAGE) 2>/dev/null || true
	@docker network rm $(NETWORK_NAME) 2>/dev/null || true

# Logs
logs:
	@echo "Showing container logs..."
	@docker logs -f $(UI_CONTAINER)

logs-tail:
	@echo "Tailing application logs..."
	@tail -f $(PROJECT_DIR)/logs/streamlit_log.log

# Development mode (runs without Docker)
dev:
	@echo "Starting development mode..."
	@if ! nc -z localhost 8000; then \
		echo "Starting model_server.py..."; \
		( \
            cd $(STREAMLIT_DIR) && \
            nohup $(VENV_UVICORN) model_server:app --host 0.0.0.0 --port 8000 > model_server.log 2>&1 & \
        ); \
		sleep 5; \
	fi
	@cd $(REFACTORED_DIR) && streamlit run main.py --server.port=$(UI_PORT)