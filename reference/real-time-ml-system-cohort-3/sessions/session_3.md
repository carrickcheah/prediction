# Session 3
### [👉 Watch the session recordings](https://www.realworldml.net/products/building-a-better-real-time-ml-system-together-cohort-3/categories/2156648142)

### [👉 Slides](https://www.realworldml.net/products/building-a-better-real-time-ml-system-together-cohort-3/categories/2156648142/posts/2183335427)


## Goals 🎯

- [x] Fix bug in candles service and make it work
    - [x] Well, there is no bug.
    - [x] Format the candle message
    - [x] EMIT_INCOMPLETE_CANDLES
    
- [x] Dockerize the candles service
- [x] Build the technical indicators service

## Questions

### Artem Philippov
*Hi. I think that the problem with .current. It should be .final*

You can use one or the other. We have a boolean parameter in our config called EMIT_INCOMPLETE_CANDLES
to do one or the other.

### Stefan Pajovic
*Is redpanda saving all these trades infinitely? what happen when our disk gets full?*

You can set the retention period for each topic, which is the time that Redpanda keeps
the data in disk. After that, the message is dropped.

By default, in our redpanda dev cluster the retention is infinite (meaning data is never dropped).
On the other hand, the kind of event we are saving is light (e.g. 22k trades -> 4MB)

### Carlo Casorzo
*if you wanted to emit not only 1m candles but also for example 1h, 1d, etc, would you normally want to use the same service and have separate sdf/out_topics?*

You would spin up at least 1 microservice for each frequency. For example:
- Candles 60-seconds
- Candles 5 minutes
- Candles 1-hour

that all read trades from the SAME input topic (`trades`) and output either to

- The same `candle` topic. In this case, you would need to add another field to your candle
messages with the frequency of the candle.

or

- To different `candle_60_sec`, `candle_5_minute`, `candle_1_hour` topics.


### Vincent Reynard Satyadharma
*How do we horizontally scale the candles services? Will the data aggregation still work if multiple instances are aggregating the same candle period?*

[👉 Blog post](https://www.realworldml.net/blog/scalable-feature-engineering-with-docker-and-kafka)

### Carlo Casorzo
*but how will the tumbling windows work this way? will quix use those state topics you showed yesterday in order to allow this distributed tumbling window calc?*

Yes. It works the same way. Quix Streams backs up the state locally on disk to make your
stateful app recoverable from failures.

### Babatunde Owoeye
*So we can increase partitions on the fly?*

Yes. Not sure if that is possible in this small dev cluster we created. But you can do it
on a production cluster. We will do it.

### Jayant Sharma
*so we wont use the actual price and candles as features of our model? Only use these technical indicators as features? or we use both in combination?*

We will use them. The final message (aka feature) that we will save to the feature store will contain
- pair
- timestamp_ms
- open, high, low, close
- all the technical indicators that we compute

### Carlo Casorzo
*I guess we are using indicators computed from the candles*

Yes. Most indicators just depend on the close prices. Some depend on volue to (volume-based)
indicators.

### Benito Martin
*How can we generate Dashboards? Does Redpanda offers that option, or we have to do it with python libraries like matplotlib or streamlit?*

Redpanda does not offer this option. Redpanda is just a message bus, not a plotting tool.
You could build a real time candle dashboard (adding even technical indicators) by

- Sending the messages in real time to Elastic Search
- Plotting them with a custom Kibana dashboard.

This is a super interesting exercise. Do you accept the challenge?

### Alexander Openstone
*I am a quite bit ahead here and apologize. Would a sentiment indicators group have a much longer time frame, say a day and not 60 minutes? How would that impact overall model?*

It depends on the news. For example, as soon as Trump appointed a crypto fan as SEC lead,
crypto prices instanlly jumped.

There are many "news" which are just noisy.

I will show you the engineering side of things. Meaning, how to map news into market signals
in real time, but I cannot guarantee you that this will work and make you money.
This is the experimentation part you need to do.

### Babatunde Owoeye
*Personally, I am still confused as to whether the consumer group specified in our transformation service has to be the same as the one specified here in our technical indicators service for them to work together.*

A consumer group is a collection of services that collaborate to accomplish a task.
For example, if you want to scale your candles service, by spinning up 2 instances (see
the question above by Vincent) you want them to be in the same consumer group (so you set
the same consumer group name).

### Babatunde Owoeye

*Let's say something happened in our transformation service and we need to clear the current state by changing some parameters as you demonstrated. Do we need to do the same for our technical indicator service?*

If for some reason you want to reset the state of your candles service, the fastest way
is by changing its consumer group name to a new one.

As for the technical-indicators service, if the candle messages have changed, yes, you want to reset. If the final candle messages are the same, no need.

### Vijay Saradhi Reddy Sakati
*Pau your answer to the question 'How do we horizontally scale the candles services? by @Vincent Reynard S - Bit confused about the answer. The partiton though appears like horizontal, from Architecture perspective still Vertical as we're using same MS*


## How to install TA lib?

This is a C library that requires an extra previous installation step
[HERE](So we can increase partitions on the fly?)

and then
```
uv add ta-lib
```