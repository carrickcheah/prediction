# Session 1
### [👉 Watch the session recordings](https://www.realworldml.net/products/building-a-better-real-time-ml-system-together-cohort-3/categories/2156625028)

### [👉 Slides](https://www.realworldml.net/products/building-a-better-real-time-ml-system-together-cohort-3/categories/2156625028/posts/2183265376)


## Goals 🎯

- Set up the development tools
- Build a microservice that ingests live trades from Kraken and pushes them to a Kafka topic. We want a modular design, so you can easily plug different real time data sources.

## Key learnings 💡

- The Universal Feature-Training-Inference pipeline design
- A bit of `uv`ing
    - `uv init`
    - `uv add`
    - `uv remove`
- How to spin up a minimal Redpanda cluster locally, with just one broker, using docker compose.
- No more printing. It is time for logging.
- Application settings with Pydantic settings.
- Quix Streams to produce messages to Kafka.
- Building a simple Websocket client in python to fetch trades in real time.


## Tools 🛠️

### 1. Code editor
My recommendations are
- [Visual Studio Code](https://code.visualstudio.com/)
- [Cursor](https://www.cursor.com/)

but feel free to use whatever editor you fell in love with.

### 2. Python >= 3.10

### 3. `uv`
We use [uv](https://docs.astral.sh/uv/getting-started/installation/) to package our Python code.

No more `pip` installs, `venv` activations or `poetry run`s. `uv` is a tool that does everything these tools do. But faster.

### 4. Docker and Docker Compose
We use Docker to run services and infrastructure locally, and docker compose to manage stacks of docker services using `.yml` files.

Advice
If you are a windows user, I recommend you install the Windows Subsystem for Linux (WSL).
Installation is as simple as opening your PowerShell and running
```powershell
wsl --install
```

### 5. `make` tool

Makefiles help Python projects by:

* Standardizing common commands (test, lint, build)
* Chaining multiple commands together
* Managing dependencies between tasks
* Providing a consistent interface across environments
* Automating repetitive development tasks

Installation
- Mac
    ```
    xcode-select --install
    ```

- Linux (Debian/Ubuntu)
    ```
    sudo apt-get update
    sudo apt-get install make
    ```

- Windows (using WSL)
    ```
    wsl --install -d Ubuntu
    sudo apt update
    sudo apt install make
    ```