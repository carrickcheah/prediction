# Session 9
### [👉 Watch the session recordings](https://www.realworldml.net/products/building-a-better-real-time-ml-system-together-cohort-3/categories/2156736223)

### [👉 Slides](https://www.realworldml.net/products/building-a-better-real-time-ml-system-together-cohort-3/categories/2156736223/posts/2183704131)

## Pending

- We still need to export our fine tuned model artifact so we can run it locally with Ollama.
- Backfill the news signals using this model.

## Goals 🎯

- [x] Training service
    - [ ] FeatureReader (WIP) -> Reads features from the store with a feature view, and preprocess the data
    into (features, target) for Supervised Machine Learning.

## Questions

### Reza Abdi
Let's say we want to implement Llama3.2 on the cloud. So here it is stored on your local device; what is the best way to run it totally on the cloud? Do we store the model weight somewhere? for each time that the pipeline wants to retrain or inference

How to serve LLM predictions?
- Ollama -> Create Modelfile
- Nvidia nim -> https://www.nvidia.com/en-eu/ai/ (wraps vLLM)
- vLLM
- ...