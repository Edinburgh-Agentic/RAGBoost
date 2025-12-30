<div align="center">
  <img src="assets/RAGBoost_logo.png" alt="RAGBoost Logo" width="400"/>

  <p><strong>Efficient Retrieval-Augmented Generation with Accuracy-Preserving Context Reuse</strong></p>

  [![arXiv](https://img.shields.io/badge/arXiv-2511.03475-b31b1b.svg)](https://arxiv.org/abs/2511.03475)
  [![Python](https://img.shields.io/badge/python-≥3.10-blue)](https://www.python.org/)
  [![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

</div>

--------------------------------------------------------------------------------

| [**Documentation**](docs/README.md) | [**Examples**](examples/) | [**Benchmarks**](docs/reference/benchmarks.md) |

## News

- [2025/12] Code is released! 
- [2025/11] Paper published: [RAGBoost: Efficient Retrieval-Augmented Generation with Accuracy-Preserving Context Reuse](https://arxiv.org/abs/2511.03475)

## About

RAGBoost is a fast optimization system for Retrieval-Augmented Generation workloads:

1. **High Throughput**: Boosting prefill throughput with intelligent context reuse.
2. **Accuracy Preserved**: Reasoning accuracy is fully preserved and even enhanced!
3. **Multi-Turn Deduplication**: Automatic context deduplication across conversation turns, reducing 30-60% redundant document processing.
4. **Strong Compatibility**: Strong compatibility with existing RAG libraries (HippoRAG), KV cache optimization engine (LMCache), and Inference engines (vLLM and SGLang). Both single-node and multi-node deployment!
5. **Widely Tested**: Tested with a wide range of RAG and Agentic AI applications.

## Benchmark and Performance

### Multi-session & Multi-turn Performance

![Benchmark Results](assets/benchmark.png)

*Tested on Qwen3-4B-Instruct-2507 with 1xH100*

### Accuracy on MT-RAG Benchmark

| Method | Qwen3-4B | Llama3.1-8B | Qwen3-30B-A3B |
|--------|----------|-------------|-----------|
| LMCache | 62.56 | **68.46** | 75.12 |
| CacheBlend | 50.33 | 56.52 | X |
| RadixCache | 62.56 | **68.46** | 75.12 |
| **RAGBoost** | **64.27** | 68.12 | **75.81** |

RAGBoost delivers **4-13x** improvements in cache hit rates and **1.5-3.5x** reductions in prefill latency for large-batch RAG workloads, while maintaining or improving accuracy.

**Furthermore**, RAGBoost has been tested to reduce input token costs by around **36%** with GPT-5.2.

See [Benchmarks](docs/reference/benchmarks.md) in the documentation for GPU vs CPU performance analysis and detailed benchmark methodology.

## Getting Started

### Installation

**Requirements:** Python >= 3.10

```bash
git clone https://github.com/SecretSettler/RAGBoost.git
cd RAGBoost
pip install -e .
```

Install an inference engine (SGLang recommended):
```bash
pip install --upgrade pip
pip install uv
uv pip install "sglang" --prerelease=allow
```

More [detailed installation instructions](docs/getting_started/installation.md) are available in the docs, including Docker setup and FAISS configuration.

## Documentation

Check out the RAGBoost [documentation](docs/README.md) for comprehensive guides.

## Examples

Go hands-on with our [examples](examples/), demonstrating how to address different use cases with RAGBoost.

## Contributing

We welcome and value all contributions! Please feel free to submit issues and pull requests.

## Contact

- [Yinsicheng Jiang](mailto:ysc.jiang@ed.ac.uk)
- [Yeqi Huang](mailto:yeqi.huang@ed.ac.uk)
- [Cheng Deng](mailto:cdeng@ed.ac.uk)
- [Liang Cheng](mailto:L.cheng@ed.ac.uk)
- [Xuan Sun](mailto:xuan.sun@ed.ac.uk)
- [Luo Mai](mailto:luo.mai@ed.ac.uk)

## Citation

If you use the code or data of RAGBoost, please declare the reference with the following:

```bibtex
@misc{jiang2025ragboost,
      title={RAGBoost: Efficient Retrieval-Augmented Generation with Accuracy-Preserving Context Reuse}, 
      author={Yinsicheng Jiang and Yeqi Huang and Liang Cheng and Cheng Deng and Xuan Sun and Luo Mai},
      year={2025},
      eprint={2511.03475},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.03475}, 
}
```
