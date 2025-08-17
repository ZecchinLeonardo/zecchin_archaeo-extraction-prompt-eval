# Named-Entity Recognition model

## Requirements

- python >= 3.10

A CUDA GPU (1 GPU is enough for a recognition in a suitable duration)

## Installation

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run the server

```sh
source venv/bin/activate
fastapi run --port $REMOTE_PORT ./src/magoh_ai_sup_server/server.py
```

The server use the first indexed GPU. To set it:

```sh
CUDA_VISIBLE_DEVICES=2 \
  fastapi run --port $REMOTE_PORT ./src/magoh_ai_sup_server/server.py
```
