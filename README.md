# Awesome Efficient LLM Inference Papers
A curated and up-to-date paper list of awesome efficient LLM inference research.

>In the AGI era, efficient inference for LLMs is critical to unlocking scalable and accessible applications. While LLMs deliver powerful capabilities, their substantial computational and memory requirements pose significant deployment challenges, particularly in resource-constrained environments. Research into optimization techniques—such as model pruning, quantization, and knowledge distillation—enables the creation of streamlined LLMs that retain high performance while minimizing resource demands. These advancements not only expand the scope of practical applications but also improve accessibility, ensuring broader utilization of LLMs across diverse platforms and use cases.

If you find some interesting work/projects, please contact me through issues or email withhaotian [at] gmail [dot] com.

*This list only focuses on the *efficient inference for LLMs*. If you are interested in edge AI computing and system, please refer to [awesome-edge-AI-papers](https://github.com/withhaotian/awesome-edge-AI-papers.git).*

# License
This project is licensed under the GPL-3.0 license - see the [LICENSE](LICENSE) file for details.

# Overview
- [Awesome Efficient LLM Inference Papers](#awesome-efficient-llm-inference-papers)
- [License](#license)
- [Overview](#overview)
  - [Surveys](#surveys)
  - [Models / Architectures Design](#models--architectures-design)
  - [KV Cache Optimization](#kv-cache-optimization)
  - [Layer Skipping / Early Exit](#layer-skipping--early-exit)
  - [Speculative Decoding](#speculative-decoding)
  - [Model Compression / Quantization](#model-compression--quantization)
  - [Serving Systems](#serving-systems)
  - [Benchmarks](#benchmarks)
  - [Applications](#applications)

## Surveys
- \[arXiv'24\] On-Device Language Models: A Comprehensive Review - \[[PDF](https://arxiv.org/abs/2409.00088)\] \[[Code](https://github.com/NexaAI/Awesome-LLMs-on-device)\]
- \[arXiv'24\] A Survey of Small Language Models - \[[PDF](https://arxiv.org/abs/2410.20011)\]
- \[arXiv'24\] Small Language Models: Survey, Measurements, and Insights - \[[PDF](https://arxiv.org/abs/2409.15790)\] \[[Code](https://github.com/UbiquitousLearning/SLM_Survey)\] \[[Demo](https://ubiquitouslearning.github.io/TinyLLMLeaderBoard/)\]
- \[arXiv'24\] A Survey of Resource-efficient LLM and Multimodal Foundation Models - \[[PDF](https://arxiv.org/abs/2401.08092.pdf)\] \[[Code](https://github.com/UbiquitousLearning/Efficient_Foundation_Model_Survey)\]
- \[arXiv'24\] On-Device Language Models: A Comprehensive Review - \[[PDF](https://arxiv.org/abs/2409.00088)\]
- \[arXiv'24\] A Survey on Model Compression for Large Language Models - \[[PDF](http://arxiv.org/abs/2308.07633)\]

## Models / Architectures Design
- \[arXiv'24\] OpenELM: An Efficient Language Model Family with Open Training and Inference Framework - \[[PDF](https://arxiv.org/abs/2404.14619)\] \[[Code](https://github.com/apple/corenet)\] \[[HuggingFace](https://huggingface.co/apple/OpenELM)\]
- \[arXiv'24\] FOX-1 TECHNICAL REPORT - \[[PDF](https://arxiv.org/abs/2411.05281)\] \[[HuggingFace](https://huggingface.co/tensoropera/Fox-1-1.6B)\]
- \[arXiv'24\] Tinyllama: An open-source small language model - \[[PDF](https://arxiv.org/abs/2401.02385)\] \[[Code](https://github.com/jzhang38/TinyLlama)\]
- \[arXiv'24\] MobileVLM V2: Faster and Stronger Baseline for Vision Language Model - \[[PDF](https://arxiv.org/abs/2402.03766)\] \[[Code](https://github.com/Meituan-AutoML/MobileVLM)\]
- \[arXiv'24\] The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits - \[[PDF](https://arxiv.org/abs/2402.17764)\]
- \[arXiv'24\] Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone - \[[PDF](https://arxiv.org/abs/2404.14219)\]
- \[arXiv'24\] MobiLlama: Towards Accurate and Lightweight Fully Transparent GPT - \[[PDF](https://arxiv.org/abs/2402.16840)\] \[[Code](https://github.com/mbzuai-oryx/MobiLlama)\]
- \[arXiv'24\] BLADE: Enhancing Black-box Large Language Models with Small Domain-Specific Models - \[[PDF](http://arxiv.org/abs/2403.18365)\]
- \[arXiv'24\] Mixture-of-Modules: Reinventing Transformers as Dynamic Assemblies of Modules - \[[PDF](http://arxiv.org/abs/2407.06677)\] \[[Code](https://github.com/gzhch/MoM)\]
- \[arXiv'24\] LayerShuffle: Enhancing Robustness in Vision Transformers by Randomizing Layer Execution Order - \[[PDF](https://github.com/matfrei/layershuffle)\]

## KV Cache Optimization
- \[ICLR'25\] CAKE: Cascading and Adaptive KV Cache Eviction with Layer Preferences - \[[PDF](http://arxiv.org/abs/2503.12491)\] \[[Code](https://github.com/antgroup/cakekv)\]
- \[arXiv'25\] DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs - \[[PDF](http://arxiv.org/abs/2412.14838)\]
- \[arXiv'25\] ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference - \[[PDF](http://arxiv.org/abs/2502.00299)\]
- \[ICASSP'25\] DynamicAttention: Dynamic KV Cache for Disaggregate LLM Inference - \[[PDF](https://ieeexplore.ieee.org/document/10890367/?arnumber=10890367)\]
- \[arXiv'24\] vTensor: Flexible Virtual Tensor Management for Efficient LLM Serving - \[[PDF](http://arxiv.org/abs/2407.15309)\]
- \[arXiv'24\] SqueezeAttention: 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget Allocation - \[[PDF](http://arxiv.org/abs/2404.04793)\] \[[Code](https://github.com/hetailang/SqueezeAttention)\]
- \[arXiv'24\] XKV: Personalized KV Cache Memory Reduction for Long-Context LLM Inference - \[[PDF](http://arxiv.org/abs/2412.05896)\]
- \[arXiv'24\] Squeezed Attention: Accelerating Long Context Length LLM Inference - \[[PDF](http://arxiv.org/abs/2411.09688)\] \[[Code](https://github.com/SqueezeAILab/SqueezedAttention)\]
- \[arXiv'24\] LazyLLM: Dynamic Token Pruning for Efficient Long Context LLM Inference - \[[PDF](http://arxiv.org/abs/2407.14057)\]
- \[NIPS'23\] H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models - \[[PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/6ceefa7b15572587b78ecfcebb2827f8-Paper-Conference.pdf)\] \[[Code](https://github.com/FMInference/H2O)\]

## Layer Skipping / Early Exit
- \[ACL'25\] CQIL: Inference Latency Optimization with Concurrent Computation of Quasi-Independent Layers - \[[PDF](http://arxiv.org/abs/2404.06709)\] \[[Code](https://github.com/Photooon/CQIL)\]
- \[arXiv'25\] Prompt-based Depth Pruning of Large Language Models - \[[PDF](http://arxiv.org/abs/2502.04348)\]
- \[arXiv'25\] Layer by Layer: Uncovering Hidden Representations in Language Models - \[[PDF](http://arxiv.org/abs/2502.02013)\]
- \[arXiv'25\] Balcony: A Lightweight Approach to Dynamic Inference of Generative Language Models - \[[PDF](http://arxiv.org/abs/2503.05005)\]
- \[arXiv'25\] A Sliding Layer Merging Method for Efficient Depth-Wise Pruning in LLMs - \[[PDF](http://arxiv.org/abs/2502.19159)\]
- \[AAAI'25\] AdaSkip: Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inference - \[[PDF](http://arxiv.org/abs/2501.02336)\] \[[Code](https://github.com/ASISys/AdaSkip)\]
- \[ICLR'25\] Streamlining Redundant Layers to Compress Large Language Models - \[[PDF](http://arxiv.org/abs/2403.19135)\] \[[Code](https://github.com/RUCKBReasoning/LLM-Streamline)\]
- \[EMNLP'24\] FFN-SkipLLM: A Hidden Gem for Autoregressive Decoding with Adaptive Feed Forward Skipping - \[[PDF](http://arxiv.org/abs/2404.03865)\]
- \[arXiv'24\] Dynamic layer selection in decoder-only transformers - \[[PDF](http://arxiv.org/abs/2410.20022)\]
- \[arXiv'24\] Not All Layers of LLMs Are Necessary During Inference - \[[PDF](http://arxiv.org/abs/2403.02181)\]
- \[arXiv'24\] Hierarchical Skip Decoding for Efficient Autoregressive Text Generation - \[[PDF](http://arxiv.org/abs/2403.14919)\]
- \[arXiv'24\] Accelerating Inference in Large Language Models with a Unified Layer Skipping Strategy - \[[PDF](http://arxiv.org/abs/2404.06954)\]
- \[arXiv'24\] ShortGPT: Layers in Large Language Models are More Redundant Than You Expect - \[[PDF](http://arxiv.org/abs/2403.03853)\]
- \[arXiv'24\] A deeper look at depth pruning of LLMs - \[[PDF](http://arxiv.org/abs/2407.16286)\]
- \[arXiv'24\] SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks - \[[PDF](http://arxiv.org/abs/2402.09025)\]
- \[ACL'24\] LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding - \[[PDF](https://aclanthology.org/2024.acl-long.681)\] - \[[Code](https://github.com/facebookresearch/LayerSkip)\]
- \[arXiv'23\] Accelerating LLaMA Inference by Enabling Intermediate Layer Decoding via Instruction Tuning with LITE - \[[PDF](http://arxiv.org/abs/2310.18581)\]
- \[arXiv'23\] SkipDecode: Autoregressive Skip Decoding with Batching and Caching for Efficient LLM Inference - \[[PDF](http://arxiv.org/abs/2307.02628)\]
- \[arXiv'23\] Fast and Robust Early-Exiting Framework for Autoregressive Language Models with Synchronized Parallel Decoding  - \[[PDF](http://arxiv.org/abs/2310.05424)\]
- \[ICML'23\] Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time - \[[PDF](https://proceedings.mlr.press/v202/liu23am/liu23am.pdf)\] \[[Code](https://github.com/FMInference/DejaVu)\]
- \[ICML'23\] EE-LLM: Large-Scale Training and Inference of Early-Exit Large Language Models with 3D Parallelism - \[[PDF](http://arxiv.org/abs/2312.04916)\] \[[Code](https://github.com/pan-x-c/EE-LLM)\]
- \[NIPS'22\] Conﬁdent Adaptive Language Modeling - \[[PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/6fac9e316a4ae75ea244ddcef1982c71-Paper-Conference.pdf)\] \[[Code](https://github.com/google-research/t5x/tree/main/t5x/contrib/calm)\]

## Speculative Decoding
- \[ICLR'25\] SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration - \[[PDF](http://arxiv.org/abs/2410.06916)\] \[[Code](https://github.com/hemingkx/SWIFT)\]
- \[arXiv'25\] QuantSpec: Self-Speculative Decoding with Hierarchical Quantized KV Cache - \[[PDF](http://arxiv.org/abs/2502.10424)\]
- \[arXiv'25\] DuoDecoding: Hardware-aware Heterogeneous Speculative Decoding with Dynamic Multi-Sequence Drafting - \[[PDF](http://arxiv.org/abs/2503.00784)\] \[[Code](https://github.com/KaiLv69/DuoDecoding)\]
- \[arXiv'25\] RASD: Retrieval-Augmented Speculative Decoding - \[[PDF](http://arxiv.org/abs/2503.03434)\]
- \[NIPS'24\] Kangaroo: Lossless Self-Speculative Decoding for Accelerating LLMs via Double Early Exiting - \[[PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/16336d94a5ffca8de019087ab7fe403f-Paper-Conference.pdf)\] \[[Code](https://github.com/Equationliu/Kangaroo)\]

## Model Compression / Quantization
- \[arXiv'25\] 2SSP: A Two-Stage Framework for Structured Pruning of LLMs - \[[PDF](http://arxiv.org/abs/2501.17771)\]
- \[arXiv'24\] AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration - \[[PDF](https://arxiv.org/abs/2306.00978)\] \[[Code](https://github.com/mit-han-lab/llm-awq)\]
- \[arXiv'24\] Exploring post-training quantization in llms from comprehensive study to low rank compensation - \[[PDF](https://arxiv.org/abs/2303.08302)\]
- \[NIPS'23\] Llm-pruner: On the structural pruning of large language models - \[[PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/44956951349095f74492a5471128a7e0-Paper-Conference.pdf)\] \[[Code](https://github.com/horseee/LLM-Pruner)\]

## Serving Systems
- \[OSDI'24\] ServerlessLLM: Low-Latency Serverless Inference for Large Language Models - \[[PDF](https://www.usenix.org/conference/osdi24/presentation/fu)\] \[[Code](https://github.com/ServerlessLLM/ServerlessLLM)\]
- \[arXiv'24\] LayerKV: Optimizing Large Language Model Serving with Layer-wise KV Cache Management - \[[PDF](http://arxiv.org/abs/2410.00428)\]
- \[arXiv'24\] Efficiently Serving LLM Reasoning Programs with Certaindex - \[[PDF](http://arxiv.org/abs/2412.20993)\]
- \[arXiv'23\]AutoDroid: LLM-powered Task Automation in Android - \[[PDF](http://arxiv.org/abs/2308.15272)\] \[[Code](https://autodroid-sys.github.io/)\]
- \[EuroSys'23\] Tabi: An Efficient Multi-Level Inference System for Large Language Models - \[[PDF](https://doi.org/10.1145/3552326.3587438)\]
- \[OSDI'23\] Efficient memory management for large language model serving with pagedattention - \[[PDF](https://dl.acm.org/doi/pdf/10.1145/3600006.3613165)\] \[[Project](https://github.com/vllm-project/vllm)\]

## Benchmarks
- \[arXiv'24\] MobileAIBench: Benchmarking LLMs and LMMs for On-Device Use Cases - \[[PDF](https://arxiv.org/abs/2406.10290)\]
- \[EdgeFM'24\] Large Language Models on Mobile Devices: Measurements, Analysis, and Insights - \[[PDF](https://doi.org/10.1145/3662006.3662059)\]

## Applications
- \[arXiv'24\] Toward Scalable Generative AI via Mixture of Experts in Mobile Edge Networks - \[[PDF](http://arxiv.org/abs/2402.06942)\]
