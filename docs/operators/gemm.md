# GEMM、量化融合与 MoE

面向集成者和算子开发者：本页说明相近矩阵接口的区别、Matmul 三后端选型，以及 grouped/量化融合算子的实际约定。

## 普通与分段 GEMM

| 接口 | 当前实现 | 限制 |
| --- | --- | --- |
| `mm_bf16` / `bmm_bf16` | CANN | 名称保留上游，声明含 BF16/FP16/FP32；按真实输入与 SoC 校验 |
| `bmm_fp8` | CANN grouped matmul | Ascend950、E4M3/E5M2 输入与相应 scale/output 约定 |
| `gemm_fp8_nt_groupwise` | CANN | Ascend950、NT 与分组 scale 布局；不提供所有上游低精度格式 |
| `SegmentGEMMWrapper` | CANN | run-only：构造后 run，不增加上游没有的 plan 步骤 |
| `matmul` | CANN / Ascend C / Triton | 相同公共 BF16/ND 契约，当前三者均为 Ascend950 的限定尺寸域 |
| `matmul_bias_silu` | Ascend C | `silu(A @ B + bias)`，BF16、ND、M/N/K 为 16..256 的 16 倍数 |

## Matmul 的三种实现

```text
flashinfer.gemm.matmul / flashinfer.matmul
  → Torch functional 或 out
  → gemm._matmul_impl：元数据/输出约定/签名
  → dispatch：按输入、硬件、显式覆盖与能力选择
  → 公共层准备或复用 out
  → run_selected
      cann runner       → 已安装 torch.mm 的 NPU 路径
      ascendc runner    → 本仓已验证 op-package → ACLNN
      triton runner     → 本目录 Triton kernel / JIT / launch
```

三个实现共享 `op="gemm.matmul"`；不同 kernel_id 区分具体实现。当前共同输入是连续 BF16 ND，M/N/K 为 16..256 的 16 倍数。指定 backend 先筛选候选，指定 ID 锁定兼容实现；无调优 winner 时按现有档位选择，不按登记顺序，也不预设 Ascend C 或 Triton 更快。

Ascend C implementation 声明 NativeBinding/BuildConfig，设备源码在 `csrc/kernels/gemm/matmul/op_kernel/arch35/`；Triton implementation 引用本目录 `kernels.py`，runner 负责其编译身份和启动。二者都不是公共接口本体，CANN 则适配已有厂商执行路径。

未提供 out 时公共层在选型后分配，提供 out 时保留对象身份并检查 alias。Ascend C Matmul 是 AOT-only；缺包不会把同一次调用改成另一后端重试。

## GroupedMatmulSwigluQuant

`grouped_matmul_swiglu_quant` 使用 Ascend910B/Ascend910C 共用的 arch22 AOT：grouped integer GEMM → 反量化 → SwiGLU → 逐 token INT8 量化，返回 `[M,N/2]` INT8 输出与 `[M]` FP32 scale。

A8W8 与 A8W4 按完整组合声明，不是对任意存储 dtype 的笛卡尔积支持。A8W4 的 INT32 carrier 表示打包 INT4，某些 scale 由 INT64 carrier 保存 UINT64 位模式；这属于位解释，不是普通数值 cast。ND/NZ 排布、scale 的 per-channel/per-group 与 K/N 对齐由具体 case/binding 联合限制。

`group_list` 是非负、非递减并以 M 结束的专家前缀，可含空专家。A8W8 的 bias/offset 不作为有效计算语义，显式拒绝；A8W4 的 bias 参数承载辅助矩阵。权重打包必须与 native 的 INT4/NZ 字节顺序一致，不能以 Torch shape/dtype/format 元数据相同代替。

公共层保留量化/裁剪语义与必要值检查，后端 runner 适配真实属性与存储。该实现 AOT-only、NOT_CAPTURABLE；分组 GEMM 不等于提供专家并行或分布式路由。

## GroupedMatmulKDim

`grouped_matmul_k_dim` 使用 Ascend910B/Ascend910C 共用的 arch22 AOT，分组沿共享 K 轴：`x` 为 `[M,TotalK]`（`trans_a=True` 时 `[TotalK,M]`），`weight` 为 `[TotalK,N]`，每组只乘属于自己的一段 K，因此输出是 `[G,M,N]`，每组一块独立结果，不做行方向拼接。

`group_list` 是 INT64 累积前缀而非每组长度：第 `g` 项是第 `g` 组在 K 轴上的**结束**位置，第 0 组从 0 开始，末项必须等于 `TotalK`。公共层在派发前校验非负、非递减与末项，空组（相邻两项相等）合法，其输出由运行前清零保证为 0，不读未初始化显存。

计算类型为 FP16/BF16、Cube 内 FP32 累加，输出可为同类型或 FP32；`fp32_output=True` 对应移植来源 `gmm_k_dim` 的 `type_promotion` 参数。四种 dtype 组合 × `trans_a` 两态对应 8 个 tiling key，纯 cube 核执行，vector 核直接退出。

`tiling` 以 `singleM/singleN=256` 划分基本块，`baseM/baseN/baseK` 由 L0A/L0B/L0C 容量反推；`dimM`、`dimN` 同时不小于 12，或满足 kernel 中另外两种块数比例条件时改用对角线扫描，否则走行优先。当前用例最大 `M=128`、`N=256`，`dimM`/`dimN` 均为 1，对角线扫描分支**未在真机验证**。

限制：仅 ND；`x` 行 stride 与 `N` 不超过 65535（Ascend C matmul 的 stride 上界）；`group_list` 长度与每组 K 长度需为正。本算子与 `grouped_matmul_swiglu_quant` 的分组方向不同——后者按 M 分组、前缀以 M 结束，二者不可互换。

## MoE

arch22 提供两个按 expert 整理 token 的公共接口，输入和输出均为连续物理 ND。

- `flashinfer.histgram(input, num_expert)` 接收非空 INT32 向量，ID 取值为 `[0, num_expert)`，返回长度为 `num_expert` 的 INT32 计数。没有 token 的 expert 返回 0。
- `flashinfer.merge_chunk2(input1_chunk1, input1_chunk2, input2_chunk1, input2_chunk2, group_list_chunk1, group_list_chunk2)` 返回两路数据和逐 expert 相加的 INT32 计数。两张 group list 分别记录各 expert 在该 chunk 的行数，各表之和须等于对应 chunk 的总行数，允许某个 expert 的计数为 0。每个 expert 内先放 chunk 1，再放 chunk 2。两路 hidden 可不同，均须按 16 对齐且至多 90112。当前设备包支持两个非空 chunk、1 至 1024 个 expert，以及 FP16 或 BF16 数据。

`fused_moe`、`fp8_per_tensor_scale_moe`、`fp8_block_scale_moe` 使用已有 CANN 原语组合与受限外部接入。它们不提供上游全部 backend、激活、量化方式和 moe_ep API。

CANN MoE 与其他通用实现统一以 `Ascend910C` 声明 A3，硬件名 `Ascend910_93xx` 归一为该值。GMM 的 B/C 变体共享源码与 runner，910B wheel 收集各自产物；A3 真机数值未验证。Matmul/FP8 等现有 A5 限定域不因共包而扩大，按 [矩阵](../backends/overview.md) 判断。

源码：[公共 GEMM](../../flashinfer/gemm.py)、[Matmul 公共定义](../../flashinfer/operators/gemm/matmul/)、[Ascend C](../../flashinfer/ascendc/gemm/)、[CANN](../../flashinfer/cann/gemm/)、[Triton Matmul](../../flashinfer/triton/gemm/matmul/)、[GMM 公共定义](../../flashinfer/operators/gemm/grouped_matmul_swiglu_quant/)、[MoE 适配](../../flashinfer/cann/fused_moe/)。

## GmmLocalExp 与 GmmLocalExpWithZero

两个算子分别从 ops-transformer PR #12993 迁入，保留独立的原生 OpType、ACLNN 接口和设备实现。公共入口为 `flashinfer.gmm_local_exp` 和 `flashinfer.gmm_local_exp_with_zero`，也可从 `flashinfer.gemm` 调用。

```python
y = flashinfer.gmm_local_exp(
    x, weight, group_list, exp_start_idx, exp_end_idx,
    trans_b=False, is_b_nz=False, type_promotion=False,
)
y_zero = flashinfer.gmm_local_exp_with_zero(
    x, weight, group_list, exp_start_idx, exp_end_idx,
    trans_b=False, is_b_nz=False, type_promotion=False,
)
```

- `x` 为 `[M,K]`；`weight` 仅包含本地专家，非转置为 `[E,K,N]`，`trans_b=True` 时为 `[E,N,K]`，其中 `E = exp_end_idx - exp_start_idx`。
- `group_list` 是 Device 上的一维 INT32 逐专家 token 数，**不是累计边界**。计数必须非负且总和等于 M，token 已按全局专家顺序排列。调用者负责计数有效性；接口不会将整张计数表拷回 Host，kernel 在 Device 上累加并计算切片边界。原 PR Torch wrapper 的 INT64 到 INT32 便利转换不属于原生 ABI；使用这里的接口时显式传 INT32。
- 本地专家区间为左闭右开 `[exp_start_idx, exp_end_idx)`。两者均返回 `[M,N]`，只计算本地专家的行；普通版保留其他行的原值（新分配输出时这些行未定义），WithZero 版将其他行清零。`out=` 可提供预分配输出，并保留对象身份。
- `x`/`weight` 使用相同 FP16 或 BF16；默认输出同 dtype，`type_promotion=True` 输出 FP32。`trans_a=True` 不支持。
- `is_b_nz=True` 表示按 NZ 字节排布读取权重，不执行格式转换，要求 `trans_b=True` 且 K、N 均为 16 的倍数。保留原实现的“ND Tensor 承载已打包 NZ 数据”用法；不能将普通 ND 数值直接标为 NZ。真实 FRACTAL_NZ 描述符同样要求属性与实际字节排布匹配。

原 PR 的手工打包方式如下（输入 `weight` 的逻辑形状为 `[E,N,K]`）：

```python
packed = weight.reshape(E, N, K // 16, 16).permute(0, 2, 1, 3).contiguous().reshape(E, N, K)
y = flashinfer.gmm_local_exp(x, packed, group_list, start, end, trans_b=True, is_b_nz=True)
```

声明提供 Ascend910B/Ascend910C 两个目标，A3 真机结果未验证；不声明图捕获支持。构建两套 A2 原生包：

```bash
python -B -m flashinfer.operators build \
  --kernel-id ascendc/gemm/gmm_local_exp/native/Ascend910B \
  --kernel-id ascendc/gemm/gmm_local_exp_with_zero/native/Ascend910B \
  --output-dir build/gmm-release
```

独立 CPU 标杆与全 tiling-key 用例在两个公共算子的 `cases.py`，局部区间、空专家、输出保留、极小输出和非法元数据专项测试在 `tests/operators/test_gmm_local_exp.py`。迁移时修正了 WithZero 空闲尾核清零长度的无符号下溢，未归一两套接口。
