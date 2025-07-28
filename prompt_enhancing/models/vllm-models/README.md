# Remote models with vllm

```sh
CUDA_VISIBLE_DEVICES=2,3 vllm serve --port 8000 ibm-granite/granite-vision-3.3-2b --tensor-parallel-size 2 --gpu-memory-utilization 0.3
```

```sh
CUDA_VISIBLE_DEVICES=2,3 HF_TOKEN=$YOUR_HF_TOKEN vllm serve --port 8001 google/gemma-3-27b-it --tensor-parallel-size 2 --gpu-memory-utilization 0.6
```

TODO: we can also serve [embedding models](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#embeddings-api_1)

