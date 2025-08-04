# Session 5
### [👉 Watch the session recordings](https://www.realworldml.net/products/building-a-better-real-time-ml-system-together-cohort-3/categories/2156685698)

### [👉 Slides](https://www.realworldml.net/products/building-a-better-real-time-ml-system-together-cohort-3/categories/2156685698/posts/2183490649)


## Goals 🎯

- [x] Backfill script
- [x] Explain design market-signal-from-news pipeline
- [x] Ingest news from external API

## Challenges/Homework

Implement the Kraken REST API as a Quix Streams Stateful Source. This way the trades service
when backfilling is recoverable from failures.

At the moment, if trades fails, it will restart fetching data from the beginning, which means
you will have duplicates.

