# NVFP4 / MXFP4 Quantization Guide for DGX Spark

## Overview

This document covers FP4 quantization formats, available models, and compatibility with DGX Spark (GB10 / Blackwell SM121).

---

## NVFP4 vs MXFP4

| Feature | NVFP4 | MXFP4 |
|---------|-------|-------|
| Block size | 16 values | 32 values |
| Scale format | FP8 (E4M3) | uint8 |
| Precision | Higher (finer-grained scaling) | Lower |
| Native HW | Blackwell SM100+ | Blackwell SM100+ |
| Memory savings | ~60% vs FP16 | ~60% vs FP16 |
| Throughput gain | ~3x vs FP16 | ~2-3x vs FP16 |

**Key difference:** NVFP4 uses smaller block sizes (16 vs 32) and FP8 scales, providing better accuracy at the cost of slightly more metadata overhead.

---

## Marlin Kernel Fallback

**Important:** On non-Blackwell GPUs (below SM100), both MXFP4 and NVFP4 fall back to the **Marlin kernel** for weight-only quantization.

vLLM will emit this warning:
```
Your GPU does not have native support for FP4 computation but FP4 quantization
is being used. Weight-only FP4 compression will be used leveraging the Marlin kernel.
This may degrade performance for compute-heavy workloads.
```

### Environment Variables

```bash
# Force Marlin kernel usage (even on Blackwell)
VLLM_MXFP4_USE_MARLIN=1

# Control activation dtype for Marlin
VLLM_MARLIN_INPUT_DTYPE=fp8  # or int8
```

### Known Issues

- **SM120 (RTX PRO 6000)** was incorrectly falling back to Marlin - [Bug #30135](https://github.com/vllm-project/vllm/issues/30135)
- **Fix:** [PR #31089](https://github.com/vllm-project/vllm/pull/31089) enables MXFP4 Triton backend for SM120

---

## vLLM-Compatible NVFP4 Models

These models are quantized with **llm-compressor** and use the **compressed-tensors** format, validated for vLLM:

### Dense Models (Recommended)

| Model | Size | vLLM Version | Notes |
|-------|------|--------------|-------|
| [RedHatAI/Qwen3-8B-NVFP4](https://huggingface.co/RedHatAI/Qwen3-8B-NVFP4) | ~2.5GB | ≥0.9.1 | Good for testing |
| [RedHatAI/Qwen3-32B-NVFP4](https://huggingface.co/RedHatAI/Qwen3-32B-NVFP4) | ~10GB | ≥0.9.1 | Production-ready |
| [RedHatAI/Llama-4-Scout-17B-16E-Instruct-NVFP4](https://huggingface.co/RedHatAI/Llama-4-Scout-17B-16E-Instruct-NVFP4) | ~5GB | ≥0.9.1 | Llama 4 |

### MoE Models (Caution)

| Model | Status | Issue |
|-------|--------|-------|
| [RedHatAI/Qwen3-30B-A3B-NVFP4](https://huggingface.co/RedHatAI/Qwen3-30B-A3B-NVFP4) | ⚠️ May fail | [#31782](https://github.com/vllm-project/vllm/issues/31782) |
| [RedHatAI/Qwen3-VL-235B-A22B-Instruct-NVFP4](https://huggingface.co/RedHatAI/Qwen3-VL-235B-A22B-Instruct-NVFP4) | ⚠️ May fail | MoE limitation |

**MoE Limitation:** Compressed-tensors NVFP4-quantized MoE models fail to initialize due to `is_act_and_mul=False` handling in vLLM's fused MoE layer.

### vLLM Usage Example

```bash
vllm serve RedHatAI/Qwen3-32B-NVFP4 \
  --port 8000 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768
```

---

## TensorRT-LLM NVFP4 Models

These models are quantized with **TensorRT Model Optimizer** and are intended for **TensorRT-LLM**:

| Model | Size | Base Model |
|-------|------|------------|
| [nvidia/Llama-3.3-70B-Instruct-NVFP4](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-NVFP4) | ~22GB | Meta Llama 3.3 70B |
| [nvidia/Llama-3.1-405B-Instruct-NVFP4](https://huggingface.co/nvidia/Llama-3.1-405B-Instruct-NVFP4) | ~120GB | Meta Llama 3.1 405B |
| [nvidia/Qwen3-32B-NVFP4](https://huggingface.co/nvidia/Qwen3-32B-NVFP4) | ~10GB | Qwen3 32B |
| [nvidia/DeepSeek-R1-NVFP4](https://huggingface.co/nvidia/DeepSeek-R1-NVFP4) | Large | DeepSeek R1 |
| [nvidia/DeepSeek-V3.2-NVFP4](https://huggingface.co/nvidia/DeepSeek-V3.2-NVFP4) | Large | DeepSeek V3.2 |

---

## TensorRT-LLM Setup for Llama-3.3-70B-Instruct-NVFP4

### Model Specifications

| Property | Value |
|----------|-------|
| Architecture | Transformer (Llama 3.3) |
| Parameters | 41B (quantized from 70B) |
| Context Length | Up to 128K tokens |
| Memory Reduction | ~3.3x vs FP16 |
| Test Hardware | B200 |

### Accuracy vs BF16 Baseline

| Benchmark | BF16 | FP4 | Delta |
|-----------|------|-----|-------|
| MMLU | 83.3 | 81.1 | -2.2 |
| GSM8K CoT | 95.3 | 92.6 | -2.7 |
| ARC Challenge | 93.7 | 93.3 | -0.4 |
| IFEVAL | 92.1 | 92.0 | -0.1 |

### Setup Instructions

#### Option 1: TensorRT-LLM LLM API (Python)

```python
from tensorrt_llm import LLM, SamplingParams

def main():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Note: Model card uses "FP4" suffix in code examples
    llm = LLM(model="nvidia/Llama-3.3-70B-Instruct-NVFP4")
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == '__main__':
    main()
```

#### Option 2: Docker Container

```bash
docker run --rm -it \
  --ipc host \
  --gpus all \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8000:8000 \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc0
```

#### Option 3: Build TensorRT Engine Manually

```bash
# Clone TensorRT-LLM
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM

# Convert checkpoint
python examples/llama/convert_checkpoint.py \
  --model_dir nvidia/Llama-3.3-70B-Instruct-NVFP4 \
  --output_dir ./tllm_checkpoint \
  --dtype float16 \
  --tp_size 2

# Build engine
trtllm-build \
  --checkpoint_dir ./tllm_checkpoint \
  --output_dir ./tllm_engine \
  --gemm_plugin float16 \
  --max_batch_size 8 \
  --max_input_len 4096 \
  --max_seq_len 8192
```

### Licensing

- **NVIDIA License:** [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
- **Meta License:** [Llama 3.3 License](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/main/LICENSE)

---

## DGX Spark (GB10) Compatibility Issues

### TensorRT-LLM on DGX Spark: NOT FULLY FUNCTIONAL

The DGX Spark uses **Blackwell GB10 (SM121)** architecture. TensorRT-LLM has significant compatibility issues:

#### Known Problems

| Issue | Error Message | Status |
|-------|---------------|--------|
| Attention Sinks | `Assertion failed: The attention sinks is only supported on SM90` | Unresolved |
| Fused MoE | `NotImplementedError: TRTLLMGenFusedMoE does not support SM120 and above` | Unresolved |
| Memory OOM | Engine build OOM with incorrect memory reporting (~119GB vs 128GB) | Ongoing |
| Multi-Node | Native multi-node inference stack NOT ready for GB10/SM121 | Workarounds exist |

#### Performance Issues

Users report:
- TRT-LLM using ~90GB memory vs ~43GB on LM Studio for same model
- Token generation: 2.5 tok/s (TRT-LLM) vs 4.6-4.9 tok/s (LM Studio)

#### References

- [Bug: Can't run GPT-OSS models on DGX Spark](https://github.com/NVIDIA/TensorRT-LLM/issues/8474)
- [TensorRT OOM problem on Spark](https://forums.developer.nvidia.com/t/tensorrt-oom-problem/355838)
- [TRT-LLM slower than GGUF on Spark](https://forums.developer.nvidia.com/t/trt-llm-for-inference-with-nvfp4-safetensors-slower-than-lm-studio-gguf-on-the-spark/348636)
- [Multi-Node Inference Report](https://forums.developer.nvidia.com/t/dgx-spark-multi-node-llm-inference-report-for-qwen3-235b-model/355126)

---

## Recommendations for DGX Spark

### Use vLLM Instead of TensorRT-LLM

Given TensorRT-LLM's issues on DGX Spark, **vLLM is the recommended inference engine**:

```bash
# Use RedHatAI's vLLM-validated NVFP4 models
vllm serve RedHatAI/Qwen3-32B-NVFP4 \
  --port 8000 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.9 \
  --load-format fastsafetensors
```

### Model Recommendations by Use Case

| Use Case | Model | Format |
|----------|-------|--------|
| Testing | RedHatAI/Qwen3-8B-NVFP4 | vLLM NVFP4 |
| Production (Dense) | RedHatAI/Qwen3-32B-NVFP4 | vLLM NVFP4 |
| Large Context | QuantTrio/MiniMax-M2-AWQ | vLLM AWQ |
| Vision-Language | QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ | vLLM AWQ |

### If You Must Use TensorRT-LLM

1. Use the latest container: `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc0`
2. Monitor [TensorRT-LLM issues](https://github.com/NVIDIA/TensorRT-LLM/issues) for SM121 fixes
3. Consider smaller models that don't trigger MoE/attention sink issues
4. Check the [DGX Spark forums](https://forums.developer.nvidia.com/c/dgx-spark-gb10/556) for workarounds

---

## Additional Resources

### Documentation
- [vLLM FP4 Quantization Docs](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a4_fp4/)
- [TensorRT-LLM Quick Start](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html)
- [LLM Compressor GitHub](https://github.com/vllm-project/llm-compressor)

### Model Collections
- [RedHatAI NVFP4 Models](https://huggingface.co/collections/RedHatAI/nvfp4-models)
- [NVIDIA NVFP4 Quantization Guide](https://build.nvidia.com/spark/nvfp4-quantization)

### Blog Posts
- [Introducing NVFP4 (NVIDIA Blog)](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [LLM Compressor 0.9.0 (Red Hat)](https://developers.redhat.com/articles/2026/01/16/llm-compressor-090-attention-quantization-mxfp4-support-and-more)
- [Blackwell NVFP4 Performance Comparison](https://huggingface.co/blog/apsys/blackwell-nvfp4-comparison)

---

## Changelog

- **2026-01-24**: Initial document created
