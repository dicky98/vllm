# Fused MoE 算子核特性 (Fused MoE Kernel Features)

本文档旨在提供各种 MoE 算子核（包括模块化和非模块化）的概述，以便于针对任何特定情况选择合适的一组算子核。这包括关于模块化算子核所使用的 All-to-All 通信后端的信息。

## Fused MoE 模块化 All2All 后端 (Fused MoE Modular All2All Backends)

有许多 All-to-All 通信后端被用于为 `FusedMoE` 层实现专家并行（EP）。不同的 `FusedMoEPrepareAndFinalizeModular` 子类为每个 All-to-All 后端提供了一个接口。

下表描述了每个后端的关键特性，即激活值格式、支持的量化方案以及异步支持情况。

输出激活值格式（标准格式 standard 或批处理格式 batched）对应于 `FusedMoEPrepareAndFinalizeModular` 子类的 prepare 步骤的输出，并且 finalize 步骤也需要相同的格式。所有后端的 `prepare` 方法都期望标准格式的激活值，而所有的 `finalize` 方法都返回标准格式的激活值。关于格式的更多细节可以在 [Fused MoE 模块化算子核](./fused_moe_modular_kernel.md) 文档中找到。

量化类型和格式列举了每个 `FusedMoEPrepareAndFinalizeModular` 类所支持的量化方案。基于 All-to-All 后端支持的格式，量化可以发生在分发（Dispatch）之前或之后。例如，`deepep_high_throughput` 仅支持块量化（Block-quantized）的 fp8 格式。任何其他格式都将导致以更高的精度进行分发并随后进行量化。每个后端的 prepare 步骤输出均为量化类型。finalize 步骤通常需要与原始激活值相同的输入类型。例如，如果原始输入是 bfloat16，且量化方案是具有每张量缩放因子（Per-tensor scales）的 fp8，则 `prepare` 将返回 fp8 且具有每张量缩放因子的激活值，而 `finalize` 将接收 bfloat16 激活值。有关 MoE 流程每一步中激活值的类型和格式的更多细节，请参见 [Fused MoE 模块化算子核](./fused_moe_modular_kernel.md) 中的图表。如果未指定量化类型，算子核将在 float16 和/或 bfloat16 上运行。

异步后端（Async backends）支持使用 DBO（双批次重叠）和共享专家重叠（Shared expert overlap，其中共享专家在 combine 步骤中进行计算）。

某些模型在 topk==1 时，需要将 topk 权重应用于输入激活值而非输出激活值，例如 Llama。对于模块化算子核，此特性由 `FusedMoEPrepareAndFinalizeModular` 子类支持。对于非模块化算子核，则由专家函数（Experts function）来处理此 Flag。

除非另有说明，否则后端通过 `--all2all-backend` 命令行参数（或 `ParallelConfig` 中的 `all2all_backend` 参数）进行控制。除 `flashinfer` 外，所有后端仅在 EP+DP 或 EP+TP 下工作。`Flashinfer` 可以在没有 EP 的情况下在 EP 或 DP 下工作。

<style>
td {
  padding: 0.5rem !important;
  white-space: nowrap;
}

th {
  padding: 0.5rem !important;
  min-width: 0 !important;
}
</style>

| 后端 (Backend) | 输出激活值格式 | 量化类型 | 量化格式 | 异步 (Async) | 在输入上应用权重 | 对应子类 |
| ------- | ------------------ | ------------ | ------------- | ----- | --------------------- | --------- |
| naive | standard | all<sup>1</sup> | G,A,T | N | <sup>6</sup> | [layer.py][vllm.model_executor.layers.fused_moe.layer.FusedMoE] |
| deepep_high_throughput | standard | fp8 | G(128),A,T<sup>2</sup> | Y | Y | [`DeepEPHTPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht.DeepEPHTPrepareAndFinalize] |
| deepep_low_latency | batched | fp8 | G(128),A,T<sup>3</sup> | Y | Y | [`DeepEPLLPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ll.DeepEPLLPrepareAndFinalize] |
| flashinfer_nvlink_two_sided | standard | nvfp4,fp8 | G,A,T | N | N | [`FlashInferNVLinkTwoSidedPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.prepare_finalize.flashinfer_nvlink_two_sided.FlashInferNVLinkTwoSidedPrepareAndFinalize] |
| flashinfer_nvlink_one_sided | standard | nvfp4,bf16,mxfp8 | G,A,T | N | N | [`FlashInferNVLinkOneSidedPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.prepare_finalize.flashinfer_nvlink_one_sided.FlashInferNVLinkOneSidedPrepareAndFinalize] |

!!! info "表格说明"
    1. 所有类型 (All types)：mxfp4, nvfp4, int4, int8, fp8
    2. A, T 量化在分发 (Dispatch) 后发生。
    3. 所有量化均在分发后发生。
    4. 由 `--moe-backend`（`flashinfer_cutlass` 或 `flashinfer_trtllm`）控制。
    5. 这是一个无操作分发器（No-op dispatcher），可与任何模块化专家配合使用，以生成不需要分发或结合即可运行的模块化算子核。这些无法通过环境变量选择。它们通常用于测试或使专家子类适配 `fused_experts` API。
    6. 这取决于专家的具体实现。

    ---

    - G - 分组量化 (Grouped)
    - G(N) - 块大小为 N 的分组量化 (Grouped w/block size N)
    - A - 逐激活 Token (Per activation token)
    - T - 逐张量 (Per tensor)

模块化算子核由以下 `FusedMoEMethodBase` 类支持：

- [`ModelOptFp8MoEMethod`][vllm.model_executor.layers.quantization.modelopt.ModelOptFp8MoEMethod]
- [`Fp8MoEMethod`][vllm.model_executor.layers.quantization.fp8.Fp8MoEMethod]
- [`CompressedTensorsW4A4Nvfp4MoEMethod`][vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4.CompressedTensorsW4A4Nvfp4MoEMethod]
- [`CompressedTensorsW8A8Fp8MoEMethod`][vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8.CompressedTensorsW8A8Fp8MoEMethod]
- [`GptOssMxfp4MoEMethod`][vllm.model_executor.layers.quantization.mxfp4.GptOssMxfp4MoEMethod]
- [`UnquantizedFusedMoEMethod`][vllm.model_executor.layers.fused_moe.UnquantizedFusedMoEMethod]

## Fused Experts 算子核 (Fused Experts Kernels)

针对不同的量化类型和架构，有许多 MoE 专家算子核的实现。大多数都遵循基础 Triton [`fused_experts`][vllm.model_executor.layers.fused_moe.fused_moe.fused_experts] 函数的通用 API。许多具有模块化算子核适配器，因此可以与兼容的 All-to-All 后端一起使用。下表列出了每个专家算子核及其特定属性。

每个算子核必须提供支持的输入激活值格式之一。某些算子核通过不同的入口点同时支持标准格式和批处理格式，例如 `TritonExperts` 和 `BatchedTritonExperts`。批处理格式的算子核目前仅在与某些 All-to-All 后端匹配时才需要，例如 `DeepEPLLPrepareAndFinalize`。

与后端算子核类似，每个专家算子核仅支持特定的量化格式。对于非模块化专家，激活值将采用原始类型并在算子核内部进行量化。模块化专家将期望激活值已经采用量化格式。两种类型的专家都将以原始激活值类型输出。

每个专家算子核都支持一个或多个激活函数（例如 silu 或 gelu），这些函数应用于中间结果。

与后端类似，一些专家支持在输入激活值上应用 topk 权重。此表中该列的条目仅适用于非模块化专家。

大多数专家类型都包含一个等效的模块化接口，该接口将是 `FusedMoEExpertsModular` 的子类。

为了与特定的 `FusedMoEPrepareAndFinalizeModular` 子类配合使用，MoE 算子核必须具有兼容的激活值格式、量化类型和量化格式。

| 算子核 (Kernel) | 输入激活值格式 | 量化类型 | 量化格式 | 激活函数 | 在输入上应用权重 | 模块化 (Modular) | 源码文件 |
| ------ | ----------------- | ------------ | ------------- | ------------------- | --------------------- | ------- | ------ |
| triton | standard | all<sup>1</sup> | G,A,T | silu, gelu,</br>swigluoai,</br>silu_no_mul,</br>gelu_no_mul | Y | Y | [`fused_experts`][vllm.model_executor.layers.fused_moe.fused_moe.fused_experts],</br>[`TritonExperts`][vllm.model_executor.layers.fused_moe.experts.triton_moe.TritonExperts] |
| triton (batched) | batched | all<sup>1</sup> | G,A,T | silu, gelu | <sup>6</sup> | Y | [`BatchedTritonExperts`][vllm.model_executor.layers.fused_moe.experts.fused_batched_moe.BatchedTritonExperts] |
| deep gemm | standard,</br>batched | fp8 | G(128),A,T | silu, gelu | <sup>6</sup> | Y | </br>[`DeepGemmExperts`][vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe.DeepGemmExperts],</br>[`BatchedDeepGemmExperts`][vllm.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe.BatchedDeepGemmExperts] |
| cutlass_fp4 | standard,</br>batched | nvfp4 | A,T | silu | Y | Y | [`CutlassExpertsFp4`][vllm.model_executor.layers.fused_moe.experts.cutlass_moe.CutlassExpertsFp4] |
| cutlass_fp8 | standard,</br>batched | fp8 | A,T | silu, gelu | Y | Y | [`CutlassExpertsFp8`][vllm.model_executor.layers.fused_moe.experts.cutlass_moe.CutlassExpertsFp8],</br>[`CutlasBatchedExpertsFp8`][vllm.model_executor.layers.fused_moe.experts.cutlass_moe.CutlassBatchedExpertsFp8] |
| flashinfer | standard | nvfp4,</br>fp8 | T | <sup>5</sup> | N | Y | [`FlashInferExperts`][vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe.FlashInferExperts] |
| gpt oss triton | standard | N/A | N/A | <sup>5</sup> | Y | Y | [`triton_kernel_fused_experts`][vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe.triton_kernel_fused_experts],</br>[`OAITritonExperts`][vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe.OAITritonExperts] |
| marlin | standard,</br>batched | <sup>3</sup> / N/A | <sup>3</sup> / N/A | silu,</br>swigluoai | Y | Y | [`fused_marlin_moe`][vllm.model_executor.layers.fused_moe.experts.marlin_moe.fused_marlin_moe],</br>[`MarlinExperts`][vllm.model_executor.layers.fused_moe.experts.marlin_moe.MarlinExperts],</br>[`BatchedMarlinExperts`][vllm.model_executor.layers.fused_moe.experts.marlin_moe.BatchedMarlinExperts] |
| trtllm | standard | mxfp4,</br>nvfp4 | G(16),G(32) | <sup>5</sup> | N | Y | [`TrtLlmMxfp4ExpertsMonolithic`][vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe.TrtLlmMxfp4ExpertsMonolithic],</br>[`TrtLlmMxfp4ExpertsModular`][vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe.TrtLlmMxfp4ExpertsModular],</br>[`TrtLlmNvFp4ExpertsMonolithic`][vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe.TrtLlmNvFp4ExpertsMonolithic],</br>[`TrtLlmNvfp4ExpertsModular`][vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe.TrtLlmNvfp4ExpertsModular] |
| hpc | standard | fp8 | G(128),T | silu | Y | Y | [`HPCExperts`][vllm.model_executor.layers.fused_moe.hpc_moe.HPCExperts] |
| rocm aiter moe | standard | mxfp4,</br>fp8 | G(32),G(128),A,T | silu, gelu,</br>swigluoai | Y | N | `rocm_aiter_fused_experts`,</br>`AiterExperts` |
| cpu_fused_moe | standard | N/A | N/A | silu | N | N | [`CPUFusedMOE`][vllm.model_executor.layers.fused_moe.cpu_fused_moe.CPUFusedMOE] |
| naive batched<sup>4</sup> | batched | int8,</br>fp8 | G,A,T | silu, gelu | <sup>6</sup> | Y | [`NaiveBatchedExperts`][vllm.model_executor.layers.fused_moe.experts.fused_batched_moe.NaiveBatchedExperts] |

!!! info "表格说明"
    1. 所有类型 (All types)：mxfp4, nvfp4, int4, int8, fp8
    2. 包裹了 triton 和 deep gemm 专家的分发器包装器。将根据类型 + 形状 + 量化参数进行选择。
    3. uint4, uint8, fp8, fp4
    4. 支持批处理格式的极简专家实现。主要用于测试。
    5. 忽略 `activation` 参数，默认使用 SwiGlu 代替。
    6. 仅在与模块化算子核一起使用时被处理或支持。

## 模块化算子核“家族” (Modular Kernel "families")

下表显示了旨在协同工作的模块化算子核“家族”。某些组合可能可行但尚未经过测试，例如 flashinfer 与其他 fp8 专家。

| 后端 (Backend) | `FusedMoEPrepareAndFinalizeModular` 子类 | `FusedMoEExpertsModular` 子类 |
| ------- | ---------------------------------------------- | ----------------------------------- |
| deepep_high_throughput | `DeepEPHTPrepareAndFinalize` | `DeepGemmExperts`,</br>`TritonExperts`,</br>`TritonOrDeepGemmExperts`,</br>`CutlassExpertsFp8`, </br>`MarlinExperts` |
| deepep_low_latency | `DeepEPLLPrepareAndFinalize` | `BatchedDeepGemmExperts`,</br>`BatchedTritonExperts`,</br>`CutlassBatchedExpertsFp8`,</br>`BatchedMarlinExperts` |
| flashinfer | `FlashInferCutlassMoEPrepareAndFinalize` | `FlashInferExperts` |
