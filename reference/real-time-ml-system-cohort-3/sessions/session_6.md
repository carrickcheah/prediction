# Session 6
### [👉 Watch the session recordings](https://www.realworldml.net/products/building-a-better-real-time-ml-system-together-cohort-3/categories/2156694000)

### [👉 Slides](https://www.realworldml.net/products/building-a-better-real-time-ml-system-together-cohort-3/categories/2156694000/posts/2183525373)


## Goals 🎯

- [x] Dockerize news ingestor
- [ ] Build the news signal extractor -> Quix Streams application that integrates with an LLM
    - [x] Build a Claude LLM that outputs structured JSON with btc_signal, eth_signal and reasoning behind these scores
    - [x] Integrate our LLM with Quix Streams
    - [ ] Add `timestamp_ms` to the `news-signal` messages
    - [ ] Push news signal to the feature store.
    


## Homework

## 1. State persistence for our news ingestor service
- Do not copy the `state` folder into the Docker image Docker (for that you can create a
`.dockerignore` file and add the `state` folder to it). Instead, attach a volume to the container
when you `docker run` it, and thers is there the state needs to be persisted.

- When we deploy this service on our Kubernetes cluster we definitely need to attach volumes
properly.

## 2. Make the output of our ClaudeNewsSignalExtractor 100% deterministic

We used `temperature=0` but the output is still not 100% deterministic.
Dig deeper into the llama-index API for this.

It seems full deterministic output is not possible with Claude.

## 3. Fix connectivity issue between the dockerized news-signal and Ollama
If you are not on Mac and have NVIDIA, you just run Ollama with docker run
and attach it to the redpanda_network.

If you are on Mac (like me) I don't know how to fix it atm.