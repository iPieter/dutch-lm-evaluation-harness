# Dutch LM Evaluation Harness
_This is a fork of EleutherAI's [English Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) with Dutch evaluations._

## Overview

This project provides a unified framework to test generative language models for Dutch.

**Features:**
- Support for Dutch evaluation benchmarks (e.g. SQUADNL).
- Support for Dutch prompts.
- Inherited features from the [English LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness): 
    - Support for models loaded via [transformers](https://github.com/huggingface/transformers/) (including quantization via [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)), [GPT-NeoX](https://github.com/EleutherAI/gpt-neox), and [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/), with a flexible tokenization-agnostic interface.
    - Support for fast and memory-efficient inference with [vLLM](https://github.com/vllm-project/vllm).
    - Support for commercial APIs including [OpenAI](https://openai.com), and [TextSynth](https://textsynth.com/).
    - Support for evaluation on adapters (e.g. LoRA) supported in [HuggingFace's PEFT library](https://github.com/huggingface/peft).
    - Support for local models and benchmarks.
    - Evaluation with publicly available prompts ensures reproducibility and comparability between papers.
    - Easy support for custom prompts and evaluation metrics.

## Install

To install the `lm-eval` package from the github repository, run:

```bash
git clone https://github.com/ipieter/-dutch-lm-evaluation-harness
cd dutch-lm-evaluation-harness
pip install -e .
```

## Basic Usage

### Hugging Face `transformers`

To evaluate a model hosted on the [HuggingFace Hub](https://huggingface.co/models) (e.g. GPT-J-6B) on `hellaswag` you can use the following command (this assumes you are using a CUDA-compatible GPU):

```bash
lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks squadnl \
    --device cuda:0 \
    --batch_size 32 \
```

#### Multi-GPU Evaluation with Hugging Face `accelerate`

To parallelize evaluation of HuggingFace models across multiple GPUs, we leverage the [accelerate 🚀](https://github.com/huggingface/accelerate) library as follows:

```
accelerate launch -m lm_eval --model hf \
    --tasks squadnl \
    --batch_size 16
```

This will perform *data-parallel evaluation*: that is, placing a **single full copy** of your model onto each available GPU and *splitting batches across GPUs* to evaluate on K GPUs K times faster than on one.

If your model is *is too large to be run on a single one of your GPUs* then you can use `accelerate` with Fully Sharded Data Parallel (FSDP) that splits the weights of the model across your data parallel ranks. To enable this, ensure you select `YES` when asked ```Do you want to use FullyShardedDataParallel?``` when running `accelerate config`. To enable memory-efficient loading, select `YES` when asked `Do you want each individually wrapped FSDP unit to broadcast module parameters from rank 0 at the start?`. This will ensure only the rank 0 process loads the model and then broadcasts the parameters to the other ranks instead of having each rank load all parameters which can lead to large RAM usage spikes around the start of the script that may cause errors.

To pass even more advanced keyword arguments to `accelerate`, we allow for the following arguments as well:
- `device_map_option`: How to split model weights across available GPUs. defaults to "auto".
- `max_memory_per_gpu`: the max GPU memory to use per GPU in loading the model.
- `max_cpu_memory`: the max amount of CPU memory to use when offloading the model weights to RAM.
- `offload_folder`: a folder where model weights will be offloaded to disk if needed.

To use `accelerate` with the `lm-eval` command, use
```
accelerate launch --no_python lm-eval --model ...
```
