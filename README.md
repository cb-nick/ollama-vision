# Chargerback AI Vision Demo: Lost Item Analyzer

This project is a demonstration of an AI-powered system for identifying and categorizing lost items from an uploaded image. It combines a powerful vision language model (VLM) with semantic search to provide detailed, structured information about items and match them against a canonical product database.

## Features

-   **AI-Powered Image Analysis**: Utilizes the Ollama-hosted `qwen2.5vl:7b` model to analyze images and extract item attributes like type, brand, color, and distinctive features.
-   **Semantic Type Matching**: Employs `sentence-transformers` to generate embeddings for item types identified by the VLM. It then uses cosine similarity to find the closest match in a pre-defined database of product aliases.
-   **Interactive Web UI**: A user-friendly Streamlit application allows for easy image uploads and visualization of the analysis results.
-   **CLI Demo**: A command-line tool is included for testing the semantic search functionality directly with text-based queries.
-   **Dockerized Deployment**: The entire web application is containerized with Docker for consistent, one-command setup and execution.
-   **Request & Feedback Logging**: All analysis requests, VLM responses, and user feedback are logged to a local SQLite database for monitoring and future model improvement.

## Architecture

The system consists of several key components:

1.  **Streamlit UI (`st_lost_item_analyzer.py`)**: The main user-facing application. It handles image uploads, communicates with the Ollama and Model Server APIs, and displays the final results.
2.  **Model Server (`model_server.py`)**: A lightweight FastAPI server that serves a `sentence-transformers` model. It exposes an `/encode` endpoint that generates vector embeddings for given text strings.
3.  **Ollama VLM**: An external, self-hosted vision language model (`qwen2.5vl:7b`) that performs the core image-to-text analysis.
4.  **Semantic Search CLI (`semantic_search.py`)**: A standalone Python script for interacting with the semantic search functionality from the command line.
5.  **Makefile**: An orchestration script that simplifies building, running, and stopping the various services.

### How It Works

1.  A user uploads an image to the Streamlit UI.
2.  The UI sends the image and a detailed system prompt to the Ollama VLM.
3.  Ollama analyzes the image and returns a structured JSON object containing the identified items and their attributes.
4.  For each item's `type` returned by Ollama, the Streamlit app calls the local **Model Server** to get a sentence embedding.
5.  This embedding is compared against a pre-computed set of embeddings for known product aliases using cosine similarity.
6.  The best match identifies the canonical "Chargerback Type" and "Product Code".
7.  The results, combining the VLM's visual analysis and the semantic search match, are displayed to the user.

## Prerequisites

-   [Docker](https://docs.docker.com/get-docker/)
-   [Make](https://www.gnu.org/software/make/)
-   A running instance of [Ollama](https://ollama.com/) with the `qwen2.5vl:7b` model available.
    ```sh
    ollama pull qwen2.5vl:7b
    ```

## Setup and Usage

### Environment Configuration

The application connects to an Ollama instance. By default, it assumes Ollama is running at `http://host.docker.internal:11434`. If your Ollama instance is located elsewhere, you can set the `OLLAMA_HOST` environment variable within the `streamlit_ui/Dockerfile` or directly in the `st_lost_item_analyzer.py` script.

### Running the Web Application

The entire application can be built and run using a single command.

1.  **Build the Docker Image:**
    This command builds the Docker image for the Streamlit UI. It also starts the backend model server on `localhost:8000` if it's not already running.

    ```sh
    make build
    ```

2.  **Run the Container:**
    This command starts the Streamlit UI in a Docker container.

    ```sh
    make run
    ```

    You can now access the web application at **http://localhost:8504**.

### Running the CLI Semantic Search Demo

To test the semantic search functionality directly, you can run the CLI tool. This command also ensures the model server is running first.

```sh
make run-search
```

The script will prompt you to enter an item type, and it will return the top 5 closest matches from the database.

### Stopping the Applications

-   To stop the Streamlit UI container and the model server:
    ```sh
    make stop
    ```
-   To stop the CLI demo and the model server:
    ```sh
    make stop-search
    ```

## Makefile Commands

The `Makefile` provides several commands for convenience:

| Command        | Description                                                                                             |
| -------------- | ------------------------------------------------------------------------------------------------------- |
| `build`        | Builds the `vision-demo` Docker image and ensures the model server is running.                         |
| `run`          | Runs the Streamlit UI as a Docker container on port `8504`.                                             |
| `stop`         | Stops the Streamlit UI container and the background model server.                                       |
| `run-search`   | Runs the interactive semantic search CLI demo.                                                          |
| `stop-search`  | Stops the semantic search CLI and the background model server.                                          |
| `clean`        | Stops all services and removes the Docker image and network created by the application.                 |

## Logging

The application generates several log files, which are stored in the `logs/` directory:

-   `streamlit_log.log`: General application logs from the Streamlit UI.
-   `model_server.log`: Logs from the FastAPI model server.
-   `streamlit_db.db`: SQLite database that stores all requests, responses, and user feedback for later analysis.
