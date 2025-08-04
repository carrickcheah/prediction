# Session 12
### [👉 Watch the session recordings](https://www.realworldml.net/products/building-a-better-real-time-ml-system-together-cohort-3/categories/2156763507)

### [👉 Slides](https://www.realworldml.net/products/building-a-better-real-time-ml-system-together-cohort-3/categories/2156763507/posts/2183825287)

## Goals 🎯

- [x] Prediction generator service
    - [x] Load latest features from the online feature store
    - [x] Implement .predict() method to generate predictions.
    - [x] Add custom Quix Streams sink to save predictions to Elastic Search
        - [x] Spin up Elastic Search locally with docker compose.
        - [x] Save predictions from our inference.py to Elastisearch
        - [x] Dockerize both the training and the inference.

- [ ] REST API in Rust
    - [x] Boilerplate server with actix-web
    - [ ] /predict endpoint that gets predictions from ES and serves them to the client app.

- [ ] Add automatic linting and formatting with clippy and rust fmt with precommit hooks.

## Questions
