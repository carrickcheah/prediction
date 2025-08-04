# Session 11
### [👉 Watch the session recordings](https://www.realworldml.net/products/building-a-better-real-time-ml-system-together-cohort-3/categories/2156759098)

### [👉 Slides](https://www.realworldml.net/products/building-a-better-real-time-ml-system-together-cohort-3/categories/2156759098/posts/2183806088)

## Goals 🎯
- [x] Hyperparameter tuning
- [x] Register model to model registry
- [ ] Prediction generator service
    - [x] Quix Streams app skeleton
    - [x] Load model artifact and metadata from CometML
    - [ ] Implement .predict() method to generate predictions.
    - [ ] Add custom Quix Streams sink to save predictions to Elastic Search
        - [ ] Spin up Elastic Search locally with docker compose.

## Questions

### Vincent Reynard Satyadharma
Based on your experience, how do you establish the baseline performance for you initial models?

Especially for difficult problem like this, I feel like we need to convince the client that our model's performance is justified.

### Jayant Sharma
Is it a good practice to subclass the original model class like XGBRegressor when defining your model class XGBRegressor. This way we might not need to define all functions, right?

Please try it, and let me know.
https://en.wikipedia.org/wiki/Composition_over_inheritance


### Jayant Sharma
so for each trial, the objective will be calculated for all the n_splits times, right? Cant we use a holdout validation (normal validation) data instead of cross validation?

### Carlo Casorzo
why did you pick elastic over any other persistance engine?

No strong reasons. We just need a place where can save the predictions, and retrieve them fast for
-> serving purposes.
-> monitoring purposes.

If we use Elastic Search we can easily generate real time dashboards on top of our data with Kibana.
