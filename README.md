# EdgeAI-Final: Efficient LLM Fine-tuning and Quantization

This repository demonstrates efficient Large Language Model (LLM) fine-tuning and quantization techniques using QLoRA (Quantized Low-Rank Adaptation) and AWQ (Activation-aware Weight Quantization) methods. The project focuses on optimizing LLMs for edge devices while maintaining model performance.

## Features

- QLoRA implementation for memory-efficient fine-tuning
- AWQ quantization for model compression
- vLLM for speeding up inference speed

## Requirements

- Python 3.x
- PyTorch
- Transformers

## Project Structure

```
.
├── train_qlora.py      # Main QLoRA training script
├── quant.py           # AWQ quantization implementation
├── EdgeAI_Final.ipynb # Jupyter notebook with examples and experiments
└── README.md         # Project documentation
```

## Setup and Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd EdgeAI-Final
```

2. Install the required dependencies:
```bash
pip install torch transformers peft datasets trl awq
```

## Usage

### Fine-tuning with QLoRA

The `train_qlora.py` script implements QLoRA fine-tuning:

```bash
python train_qlora.py
```

Key features:
- 4-bit quantization for base model
- LoRA adaptation with r=32
- Support for various target modules
- Configurable training parameters
- Automatic perplexity evaluation

### Model Quantization

The `quant.py` script handles AWQ quantization:

```bash
python quant.py
```

Quantization features:
- 4-bit weight quantization
- Zero-point quantization support
- Configurable group size
- GEMM backend for efficient inference


### Final Presentation
The `EdgeAI_Final.ipynb` notebook demonstrates the implementation of VLLM (Very Large Language Model) inference optimization, featuring:

- High-performance inference using CUDA optimization
- Continuous batching for improved throughput
- PagedAttention for efficient memory management
- Integration with AWQ quantization
- Benchmarking and performance comparisons
- Interactive examples and use cases
- Memory usage optimization techniques
- Latency and throughput measurements

The notebook serves as both a demonstration and a practical guide for implementing efficient LLM inference in production environments.
