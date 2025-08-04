# Session 7
### [👉 Watch the session recordings](https://www.realworldml.net/products/building-a-better-real-time-ml-system-together-cohort-3/categories/2156714176)

### [👉 Slides](https://www.realworldml.net/products/building-a-better-real-time-ml-system-together-cohort-3/categories/2156714176/posts/2183607860)


## Goals 🎯

- [x] Change output format of our NewsSignalExtractor.
- [x] Create instruction dataset for fine tuning
- [x] Fine tune llama3.2:1b with Unsloth

## Challenges/Homework

- [ ] Add tests for the LLMSignalExtractors. Carlo has added examples here:
https://github.com/deepbludev/crypto-predict/blob/main/services/news_signals/tests/test_sentiment_analyzer.py


## Questions

### Jayant Sharma
Wont we need two timestamps, one from tech indicators and another from news signal?
No. The feature store correctly joing features from different feature groups, respecting the timestamps, so that the 
final dataset contains no features that belong to the future. This is often called correct point-in-time joins.

### Benito Martin
Would be nice to understand a bit the costs for the fine tuning
Let's see in a few minutes.
I think we can rent something that works for less than 50 cents an hour.

### Carlo Casorzo
do you think the feature could be improved by adding some kind of "weight" associated to the news source?
for example, news from crypto.com are considered stronger than cointelegraph.com?

### Alexander Openstone
a quick question for Pau - why don't have a third state - no change


Monday
- [ ] Run the backfill pipeline
    - [ ] Cryptodata source from CSV file
    - [ ] News signal to unpack or not the event
    https://quix.io/docs/quix-streams/processing.html?h=apply#using-state-store
    - [ ] To feature CSVSink
    - [ ] Dummy NewsSentimentExtractor model to test things out
    - [ ] Use Claude with 1,000 samples

Wednesday
- [ ] Fine tune llama 3.1b
    - [ ] Generate training dataset
    - [ ] unsloth on Jupyter
    - [ ] unsloth on LambdaLabs

Thursday
- [ ] Model training
    - [ ] Feature view
    - [ ] (features, target)
    - [ ] XGBoost
    - [ ] Predictor service.




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