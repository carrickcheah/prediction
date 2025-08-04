# Trade ingestor service

Ingest trades from the Kraken trades API and pushes them to a Kafka topic.

This is the first microservice of our real-time feature pipeline, that generates technical indicator feature in real-time.

![Trade Ingestor Architecture](../../media/trades_service.jpg)


## How to run this code

### 1. Start Redpanda
Make sure you have the message bus (Redpanda in our case) up and running locally. This is a minimal Redpanda cluster that we use for developing. This is not production-ready, but good enough for us to run the whole system locally.

```bash
cd docker-compose
make start-redpanda
```

You can check the message bus is up and running by going to the Redpanda console, on `localhost:8080`. 

### 2. Set configuration parameters in the `.env` file
Once Redpanda is up and running, fill in the configuration parameters in the `settings.env` file.
Choose whatever name you want for the kafka topic, and choose the crypo pairs you are interested in.

```.env
KAFKA_BROKER_ADDRESS=localhost:19092
KAFKA_TOPIC=trades
PAIRS=["BTC/USD", "ETH/USD"]
```

### 3. Run it
You can run the service using `uv`
```bash
uv run python run.py
```
which is exaclty what the `run` command in our `Makefile` does
```bash
make run
```


