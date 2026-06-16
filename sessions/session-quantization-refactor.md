# vLLM Ascend 量化特性原理与代码实现流程

**Session ID:** ses_13197bb31ffe17zxQ0AewtV6Z0
**Created:** 6/16/2026, 3:10:00 AM
**Updated:** 6/16/2026, 7:13:15 AM

---

## User

# vLLM Ascend 特性学习文档生成器

为 vllm-ascend 仓库的特性/模块生成全面的中文代码走读学习文档。

## Workflow

### Step 1: 识别特性

从用户请求中提取要学习的特性/模块。参考 [references/feature-catalog.md](references/feature-catalog.md) 获取完整特性列表和文件路径。

常见特性示例：
- 插件架构：插件注册、Patch 系统、NPUPlatform、AscendConfig
- Attention 后端：MLA、SFA、DSA、FA3、Context Parallelism
- 分布式：FlashComm、MC2、HCCL、PD Disaggregation、KV Transfer
- 图优化：ACL Graph、npugraph_ex、图融合 Pass
- 量化：W8A8、W4A16、FP8、KV C8
- MoE：Fused MoE、EPLB、Token Dispatcher
- 推测解码：Eagle、Medusa、N-gram、Rejection Sampler
- 内存管理：CaMem Allocator、KV Offload、NZ 格式
- 调度：Dynamic Batch、Profiling Chunk、Balance Scheduling
- 硬件特定：310P 支持、Xlite、CPU Binding

### Step 2: 研究特性

这是最关键的步骤，研究的深度直接决定文档质量。

**2a. 阅读源码**
- 根据 feature-catalog.md 定位核心文件
- 阅读关键实现文件，理解：
  - 核心类及其职责
  - 关键接口和方法签名
  - 数据流和控制流
  - NPU 特有的优化和约束

**2b. 理解插件集成**
- 分析该特性如何与上游 vLLM 集成：
  - 是否通过 entry point 注册
  - 是否使用 patch 机制（检查 `vllm_ascend/patch/` 目录）
  - 是否通过继承扩展上游类
  - 配置如何从上游传递到 NPU 实现

**2c. 查阅文档和讨论**
- 使用 WebFetch 访问 `https://docs.vllm.ai/projects/ascend/` 获取官方文档
- 检查 vllm-ascend GitHub issues/discussions 了解设计动机
- 查看 `vllm_ascend/patch/__init__.py` 中的 patch 注释了解 patch 原因

**2d. 对比 GPU 版本**
- 理解上游 vLLM 中对应功能的 GPU 实现
- 识别 NPU 版本的关键差异和优化点

### Step 3: 生成学习文档

阅读 [references/style-guide.md](references/style-guide.md) 获取完整文档结构和格式规范。

关键要求：
- 使用**中文（简体中文）**撰写，技术术语保留英文
- 遵循 style-guide.md 定义的多部分结构
- 包含丰富的可视化元素：
  - **Mermaid 架构图** — 展示系统组件和层级关系
  - **Mermaid 流程图** — 展示数据流和执行流程
  - **Mermaid 时序图** — 展示组件交互
  - **Mermaid 类图** — 展示继承和组合关系
  - **GPU vs NPU 对比表** — 对比上游 vLLM 与 vllm-ascend 的实现差异
  - **参数参考表** — 列出所有相关环境变量和配置参数
  - **代码片段** — 引用 vllm-ascend 真实代码并标注文件路径
- 重点阐述：
  - **为什么需要这个特性**（问题背景和动机）
  - **NPU 硬件约束**如何影响软件设计
  - **插件集成机制**（patch/继承/entry point）
  - **与 GPU 版本的关键差异**
- 包含文档概述（目标读者、阅读指南）
- 附录包含代码位置索引和术语表

### Step 4: 保存输出

将生成的 Markdown 文件保存到 `./outputs/` 目录：
- 文件路径: `outputs/{feature_name}.md`（如 `outputs/acl_graph.md`、`outputs/patch_system.md`）
- 如 `outputs/` 目录不存在则创建
- 文件名使用 snake_case

## 内容深度指南

- **第一部分（基础与背景）**: 从问题动机出发，解释"为什么"。提供架构总览图。对比 GPU 版本说明 NPU 的差异和约束。
- **第二部分（插件集成）**: 分析 entry point 注册、patch 替换逻辑、继承关系。这是 vllm-ascend 文档的核心差异点。
- **第三部分（核心实现）**: 走读 vllm-ascend 源码。标注文件路径，追踪执行流。重点分析 NPU 特有优化。
- **第四部分（配置与使用）**: 列出所有环境变量（`envs.py`）和配置参数（`ascend_config.py`），提供使用示例和调优建议。
- **附录**: 代码位置索引表（组件 → 文件路径映射）。

## 质量检查清单

保存文档前确认：
- [ ] 文档包含 3+ Mermaid 图表（架构图、流程图、时序图/类图）
- [ ] 文档包含 3+ 对比/参考表格
- [ ] 文档包含 vllm-ascend 源码片段并标注文件路径
- [ ] 文档使用中文撰写，技术术语保留英文
- [ ] 所有章节有实质内容（无占位文本）
- [ ] 文档头部包含版本和日期元数据
- [ ] 明确说明了与上游 vLLM / GPU 版的差异
- [ ] 列出了所有相关环境变量和配置参数

## References

- [references/style-guide.md](references/style-guide.md) — 文档结构、格式规范和内容模式
- [references/feature-catalog.md](references/feature-catalog.md) — vllm-ascend 完整特性目录与文件路径索引


详细总结一下量化特性，包括原理和代码实现流程

---

## Assistant

**Tool: skill**

---

## Assistant

**Tool: read**

**Tool: read**

---

## Assistant

**Tool: todowrite**

---

## Assistant

**Tool: read**

**Tool: read**

---

## Assistant

**Tool: read**

**Tool: read**

**Tool: read**

**Tool: read**

**Tool: read**

---

## Assistant

**Tool: read**

**Tool: read**

**Tool: read**

**Tool: read**

**Tool: read**

**Tool: read**

---

## Assistant

**Tool: read**

**Tool: read**

**Tool: read**

**Tool: read**

**Tool: read**

---

## Assistant

**Tool: read**

**Tool: read**

**Tool: read**

**Tool: read**

---

## Assistant

**Tool: todowrite**

---

## Assistant

**Tool: grep**

**Tool: grep**

**Tool: grep**

---

## Assistant

**Tool: read**

**Tool: read**

---

## Assistant

**Tool: todowrite**

---

## Assistant

Now I'll create the outputs directory and write the comprehensive document.



**Tool: bash**

---

## Assistant

**Tool: write**

---

## Assistant

**Tool: bash**

---

## Assistant

**Tool: todowrite**

---

## Assistant

文档已生成并保存到 `outputs/quantization.md`（996 行），涵盖以下内容：

**四部分结构**：
1. **基础与背景** — 量化动机、GPU vs NPU 差异对比表、核心概念（MX/NZ/量化粒度）、整体架构图
2. **插件集成机制** — entry point 注册流程、自动量化检测、Scheme 注册表（15 个已注册 Scheme 完整列表）、适配器层类图
3. **核心实现深度分析** — 逐一走读 W8A8 Dynamic/Static、MXFP8、W4A8、W4A16、MXFP4、C8 KV Cache、FAKQuant 的源码实现，含 NPU 算子调用和权重后处理流程
4. **配置与使用指南** — 模型-量化方法支持矩阵、环境变量/配置参数表、性能调优建议、已知限制

**可视化元素**：7 个 Mermaid 图表（架构总览、注册时序、前向计算流程、权重处理流程、类继承关系、执行流程总览）、10+ 对比/参考表格、大量标注文件路径的代码片段。

---

## User

保存的完整目录是多少

---

## Assistant

`/home/y00884608/code/.agents/skills/vllm-ascend-feature-learning/outputs/quantization.md`

---

## User

> **文档版本**: 1.0
> **分析代码版本**: vllm-ascend main 分支（截至 2025-06）
> **最后更新**: 2025-06-16这里最新分支版本怎么是2025-06，现在已经2026年6月了

---

## Assistant

**Tool: edit**

---

## Assistant

已修正为 `2026-06`。

---

## User

是skill的问题吗，为啥这个时间会错，下次再生成还会错吗

---

## Assistant

不是 skill 的问题。style-guide.md 中的模板只定义了格式占位符（`YYYY-MM`、`YYYY-MM-DD`），具体日期值是我生成文档时填写的。系统环境已告知当前日期是 2026-06-16，是我写错了。下次生成时我会注意使用正确的当前日期。

---

## User

这个文档里的Linear、MOE、Attention三层量化，分别对应大模型的什么层，以qwen3-30B为例，详细解释一下这三层在模型中的具体位置

---

## Assistant

以 Qwen3-30B（MoE 架构）为例，一个 Transformer Layer 的结构如下：

```
Transformer Layer
├── RMSNorm (input_layernorm)
├── Self-Attention
│   ├── qkv_proj  ← 【Linear 量化】(ColumnParallelLinear, 融合 q/k/v 三个投影)
│   ├── Attention 计算 (Paged Attention) ← 【Attention 量化】(KV Cache 量化发生在这里)
│   └── o_proj    ← 【Linear 量化】(RowParallelLinear)
├── RMSNorm (post_attention_layernorm)
└── MoE FFN
    ├── router (gate) — 专家路由，不量化
    ├── Shared Experts (如有)
    │   ├── gate_proj + up_proj ← 【Linear 量化】
    │   └── down_proj           ← 【Linear 量化】
    └── Routed Experts (256 个专家，每次激活 top-8)
        ├── gate_proj + up_proj (融合为 w13) ← 【MoE 量化】
        └── down_proj (w2)                    ← 【MoE 量化】
```

**具体对应关系**：

| 量化层类型 | Qwen3-30B 中的具体层 | 说明 |
|-----------|---------------------|------|
| **Linear** | `qkv_proj`、`o_proj`、Shared Expert 的 `gate_up_proj`/`down_proj`、`lm_head` | 普通的 `nn.Linear`（TP 切分的 ColumnParallel/RowParallel），每层 2-4 个 |
| **MoE** | `FusedMoE`（256 个 routed experts 的 `w13_weight` + `w2_weight`） | 所有专家的权重融合为一个 `[num_experts, ...]` 的 3D 张量，用 `npu_grouped_matmul` 批量计算 |
| **Attention** | Attention 计算过程中的 KV Cache | 不量化权重，而是量化 **KV Cache 存储**（C8 INT8）或 Attention 计算中的 K 张量（FAKQuant） |

**核心区别**：
- **Linear vs MoE**：本质都是矩阵乘法，但 MoE 层的权重是 3D 张量（多了 expert 维度），需要专家路由（top-k 选择）+ AllToAll 通信，所以单独抽象为 `AscendMoEScheme`
- **Attention 量化**：不量化模型权重，而是量化运行时动态生成的 KV Cache，减少长序列场景的显存占用。在 vllm-ascend 中通过替换 attention 后端实现（`AscendC8AttentionBackendImpl`）

---

## User

qkv_proj、p_proj、gate_proj、up_proj、down_proj是啥

---

## Assistant

这些都是 Transformer 中的线性投影层（`nn.Linear`），各自的作用：

**Attention 部分**：
| 层名 | 作用 | 形状 (Qwen3-30B) |
|------|------|-----------------|
| `q_proj` | 将 hidden_states 投影为 **Query** 向量 | `[hidden, num_heads × head_dim]` |
| `k_proj` | 投影为 **Key** 向量 | `[hidden, num_kv_heads × head_dim]` |
| `v_proj` | 投影为 **Value** 向量 | `[hidden, num_kv_heads × head_dim]` |
| `o_proj` | Attention 输出的 **Output** 投影 | `[num_heads × head_dim, hidden]` |
| `qkv_proj` | vLLM 将 q/k/v **融合**为一个层，减少 kernel launch | `[hidden, (num_heads + 2×num_kv_heads) × head_dim]` |

计算流程：`x → qkv_proj → Q,K,V → Attention(Q,K,V) → o_proj → output`

**FFN/MoE 部分**（SwiGLU 激活）：
| 层名 | 作用 | 公式 |
|------|------|------|
| `gate_proj` | **门控**投影 | `gate = SiLU(xW_gate)` |
| `up_proj` | **上投影** | `up = xW_up` |
| `down_proj` | **下投影**（降维回 hidden_size） | `output = (gate ⊙ up) × W_down` |

vLLM 通常将 `gate_proj` 和 `up_proj` 融合为 `gate_up_proj`（MoE 中叫 `w13`），`down_proj` 对应 `w2`。

**直观理解**：
```
Attention:  x → [qkv_proj] → Q,K,V → 注意力计算 → [o_proj] → x'
FFN/MoE:   x' → [gate_up_proj] → 激活 → [down_proj] → x''
```

`gate_proj` 和 `up_proj` 将 hidden_size 扩展到更大的 intermediate_size（通常 4 倍），经过逐元素相乘和激活后，`down_proj` 再压缩回 hidden_size。这就是 Transformer 中"先扩展再压缩"的 FFN 设计。

---

## User

先扩展再压缩的FFN设计的目的是啥

---

## Assistant

核心目的是**在高维空间中进行更丰富的非线性特征变换**，然后压缩回原维度以兼容残差连接。

**类比理解**：把 FFN 想象成一个"知识加工车间"：

1. **扩展**（hidden → 4×hidden）：将信息投射到更高维空间，让原本在低维空间中线性不可分的特征变得可分（类似 SVM 的核技巧思想）
2. **非线性激活**（SiLU/GeLU）：在高维空间中做非线性变换，学习复杂的特征组合
3. **压缩**（4×hidden → hidden）：将加工后的信息压回原维度，作为"提炼后的知识"通过残差连接叠加回去

**为什么不直接 hidden → hidden？** 因为单层线性变换 `y = Wx` 的表达能力有限。扩展到 4 倍维度后，等效于用 `W_down · SiLU(W_up · x)` 拟合一个更复杂的非线性函数，参数量从 $d^2$ 增加到 $8d^2$（gate + up + down），模型容量大幅提升。

**为什么不保持 4×hidden？** 残差连接要求输入输出维度一致（`x + FFN(x)`），所以必须压缩回来。同时压缩本身也是一种信息瓶颈（information bottleneck），迫使模型学习更有价值的特征表示。

---

## User

up_proj里的proj是什么

---

## Assistant

`proj` 是 **projection（投影）** 的缩写。

在深度学习中，"投影"本质就是**矩阵乘法**（线性变换）：`y = xW`。叫"投影"是因为从数学角度看，将输入向量乘以一个权重矩阵，等价于将向量从一个空间映射（投影）到另一个空间。

- `up_proj`：向上投影（低维 → 高维，hidden → 4×hidden）
- `down_proj`：向下投影（高维 → 低维，4×hidden → hidden）
- `q_proj`：投影到 Query 空间
- `gate_proj`：投影到门控空间

名字只是约定俗成，实际就是一个 `nn.Linear` 层。

---

## User

quantization.md文档里的附录C量化执行流程总览，结合一个具体的例子，比如输入一个prompt，量化流程是怎么执行的

---

## Assistant

以 `vllm serve /path/to/Qwen3-30B-W8A8 --quantization ascend` 为例，完整流程分两个阶段：

## 阶段一：模型加载（启动时，一次性）

```
用户执行 vllm serve 命令
    │
    ▼
① detect_quantization_method()
   扫描模型目录，发现 quant_model_description.json
   → 返回 "ascend"
    │
    ▼
② AscendModelSlimConfig.from_config()
   读取 quant_model_description.json，得到每层的量化类型：
   {
     "model.layers.0.self_attn.qkv_proj.weight": "W8A8_DYNAMIC",
     "model.layers.0.self_attn.o_proj.weight": "W8A8_DYNAMIC",
     "model.layers.0.mlp.experts.0.gate_proj.weight": "W8A8_DYNAMIC",
     "model.layers.0.mlp.experts.0.down_proj.weight": "W8A8_DYNAMIC",
     "lm_head.weight": "FLOAT"   ← lm_head 不量化
   }
    │
    ▼
③ get_quant_method() — 为每个 nn.Module 选择量化方法
   遍历模型所有层，根据层类型 + 量化描述匹配：

   qkv_proj (ColumnParallelLinear)
     → get_scheme_class("W8A8_DYNAMIC", "linear")
     → AscendW8A8DynamicLinearMethod
     → 包装为 AscendLinearMethod(scheme)

   FusedMoE (256个专家融合层)
     → get_scheme_class("W8A8_DYNAMIC", "moe")
     → AscendW8A8DynamicFusedMoEMethod
     → 包装为 AscendFusedMoEMethod(scheme)

   lm_head (LinearBase, 标记为 "FLOAT")
     → is_layer_skipped_ascend() == True
     → AscendUnquantizedLinearMethod（不量化）
    │
    ▼
④ create_weights() — 创建正确 shape/dtype 的参数占位

   以 qkv_proj 为例（hidden=2048, heads=16, kv_heads=4, head_dim=128）：
   scheme.get_weight() 返回：
     weight: torch.empty(2560, 2048, dtype=torch.int8)   ← INT8 而非 BF16
   scheme.get_perchannel_param() 返回：
     weight_scale:  torch.empty(2560, 1, dtype=torch.bfloat16)
     weight_offset: torch.empty(2560, 1, dtype=torch.bfloat16)

   以 FusedMoE 为例（256 experts, intermediate=512, hidden=2048）：
   scheme.get_weight() 返回：
     w13_weight: torch.empty(256, 1024, 2048, dtype=torch.int8)
     w2_weight:  torch.empty(256, 2048, 512,  dtype=torch.int8)
   scheme.get_dynamic_quant_param() 返回：
     w13_weight_scale: torch.empty(256, 1024, 1, dtype=torch.float32)
     w2_weight_scale:  torch.empty(256, 2048, 1, dtype=torch.float32)
    │
    ▼
⑤ load_weights() — 从 safetensors 加载 INT8 权重到上面创建的参数中
    │
    ▼
⑥ process_weights_after_loading() — 权重后处理（关键步骤！）

   qkv_proj:
     weight [2560, 2048] INT8
       → transpose → [2048, 2560]
       → maybe_trans_nz → NZ 分形格式
     weight_scale [2560, 1] → flatten → [2560]

   FusedMoE:
     w13_weight [256, 1024, 2048] INT8
       → transpose(1,2) → [256, 2048, 1024]
       → npu_format_cast(NZ) → NZ 格式
     w13_weight_scale → flatten → FP32
```

## 阶段二：推理（每次请求）

用户输入 prompt：`"你好，请介绍一下自己"`

```
Tokenizer 编码 → token IDs: [108386, 3837, 99489, ...]
    │
    ▼
Embedding 查表 → hidden_states: [num_tokens, 2048] BF16
    │
    ▼
╔══════════════════════════════════════════════════╗
║  Layer 0 Self-Attention — qkv_proj (Linear)     ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  x: [num_tokens, 2048] BF16                     ║
║       │                                          ║
║       ▼                                          ║
║  npu_dynamic_quant(x)                            ║
║       │                                          ║
║       ├──→ quantized_x: [num_tokens, 2048] INT8 ║
║       └──→ pertoken_scale: [num_tokens, 1] FP32 ║
║              │                                   ║
║              ▼                                   ║
║  npu_quant_matmul(                               ║
║      quantized_x,     # INT8 激活                ║
║      layer.weight,    # INT8 权重 (NZ格式)       ║
║      layer.weight_scale,  # FP32 per-channel     ║
║      pertoken_scale,  # FP32 per-token           ║
║      output_dtype=BF16   # 反量化回 BF16         ║
║  )                                               ║
║       │                                          ║
║       ▼                                          ║
║  QKV: [num_tokens, 2560] BF16                   ║
║       │                                          ║
║       ▼                                          ║
║  Attention 计算 (Paged Attention)                ║
║  → 写入 KV Cache（如果启用 C8 则 INT8 存储）     ║
║       │                                          ║
║       ▼                                          ║
║  o_proj: 同样的 dynamic_quant → quant_matmul     ║
║       │                                          ║
║       ▼                                          ║
║  attn_output: [num_tokens, 2048] BF16           ║
╚══════════════════════════════════════════════════╝
    │
    ▼  残差连接: x = x + attn_output
    │
╔══════════════════════════════════════════════════╗
║  Layer 0 MoE FFN — FusedMoE                     ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  x: [num_tokens, 2048] BF16                     ║
║       │                                          ║
║       ▼                                          ║
║  router(x) → router_logits: [num_tokens, 256]   ║
║       │                                          ║
║       ▼                                          ║
║  select_experts(top_k=8)                         ║
║       ├──→ topk_ids: [num_tokens, 8]  (选中哪8个专家)║
║       └──→ topk_weights: [num_tokens, 8] (专家权重)║
║              │                                   ║
║              ▼                                   ║
║  moe_comm_method.fused_experts()                 ║
║  ┌──────────────────────────────────────────┐    ║
║  │ AllToAll: 按专家分发 token 到各 NPU 卡   │    ║
║  │                                          │    ║
║  │ 对每个专家执行:                           │    ║
║  │   npu_dynamic_quant(x) → INT8            │    ║
║  │   npu_grouped_matmul(                    │    ║
║  │     INT8_x, INT8_w13, scale) → gate+up   │    ║
║  │   SiLU(gate) ⊙ up → intermediate         │    ║
║  │   npu_dynamic_quant(intermediate) → INT8 │    ║
║  │   npu_grouped_matmul(                    │    ║
║  │     INT8_x, INT8_w2, scale) → output     │    ║
║  │                                          │    ║
║  │ AllToAll: 将结果收回到原始 NPU 卡        │    ║
║  └──────────────────────────────────────────┘    ║
║       │                                          ║
║       ▼                                          ║
║  moe_output: [num_tokens, 2048] BF16            ║
╚══════════════════════════════════════════════════╝
    │
    ▼  残差连接: x = x + moe_output
    │
    ▼  重复 48 层...
    │
    ▼
lm_head (FLOAT, 不量化): [num_tokens, 2048] → [num_tokens, vocab_size]
    │
    ▼
Sampler → 生成下一个 token → 循环直到结束
```

**关键理解**：量化发生在**每一次矩阵乘法**中。每个 Linear/MoE 层的 `apply()` 方法内部都执行三步：量化激活 → 量化矩阵乘 → 反量化输出。模型权重在加载时就已经 INT8/INT4 格式了（阶段一），推理时只需对激活值做动态量化。

---

## User

把这个也更新到文档里吧，就放在附录C后面

---

## Assistant

**Tool: read**

---

## Assistant

**Tool: edit**

---

## Assistant

已更新，新增内容作为 **附录 D. 量化执行流程具体示例（以 Qwen3-30B W8A8 为例）**，包含：

- **D.1 模型加载阶段**：从 `detect_quantization_method()` 到 `process_weights_after_loading()` 的 6 步完整流程
- **D.2 推理阶段**：从 prompt 输入到 token 输出，逐层展示 Linear（qkv_proj/o_proj）和 MoE（FusedMoE）的量化计算细节

---

## User

lmhead是干啥的，采样用的吗？

---

## Assistant

`lm_head` 是**语言模型头**（Language Model Head），作用是将最后一层 Transformer 输出的 hidden_states 映射到**词表空间**，得到每个词的 logits（未归一化的概率分数）。

```
最后一层 Transformer 输出: [num_tokens, hidden_size]  (如 [1, 2048])
        │
        ▼
lm_head (nn.Linear):  [hidden_size, vocab_size]  (如 [2048, 151936])
        │
        ▼
logits: [num_tokens, vocab_size]  (如 [1, 151936])  ← 每个词一个分数
        │
        ▼
Sampler (采样器): 从 logits 中选一个 token
  ├── greedy: argmax(logits) → 选分数最高的词
  ├── top-k: 从分数最高的 k 个词中采样
  └── top-p: 从累积概率达到 p 的词中采样
        │
        ▼
下一个 token ID
```

所以 `lm_head` 和 Sampler 是两个独立的步骤：
- **lm_head**：线性投影，`hidden → logits`（纯矩阵乘法，不涉及随机性）
- **Sampler**：从 logits 中采样选 token（引入随机性，受 temperature/top_k/top_p 控制）

这也是为什么量化模型中 `lm_head` 通常标记为 `FLOAT` 不量化——它是最后一层，输出直接影响采样质量，量化误差会被放大。

---

## User

lmhead是计算logits的对吧

---

## Assistant

对，就是一个矩阵乘法：`logits = hidden_states @ lm_head.weight`，输出每个词的概率分数。

---

## User

# vLLM Ascend Feature Design

## Persona

Senior ML systems engineer specializing in Ascend NPU hardware plugins for vLLM. Deep knowledge of the vllm-ascend patch system, platform hooks, NPU operator APIs (torch_npu), and distributed inference on Ascend 910B/C.

## Core Principles

- Prefer vLLM's formal extension points (entry points, registration, Platform hooks) over monkey-patching
- Use patches only when no formal extension point exists — see [references/patch-system.md](references/patch-system.md) for when to patch vs inherit
- Follow NPU constraints: no `tensor.item()` in hot paths, prefer in-place ops, use `torch_npu` APIs — see [references/feature-patterns.md](references/feature-patterns.md) for full constraints
- Line length: 120 chars (not 88)
- All env vars use `VLLM_ASCEND_*` prefix, defined centrally in `vllm_ascend/envs.py`
- Do NOT add test cases unless explicitly requested

## Workflow

### Step 1 — Clarify (if needed)

If requirements are ambiguous, ask up to 3 focused questions:
- Which vllm-ascend component does this feature touch? (patch/attention/ops/worker/scheduler/spec_decode/quantization)
- Does upstream vLLM provide a formal extension point, or is a patch needed?
- Target hardware: 910B, 910C, or 310P?

Otherwise proceed with the simplest valid assumption.

### Step 2 — Explore Codebase

Before designing, explore the relevant code:

1. Read `vllm_ascend/patch/__init__.py` to check for existing related patches
2. Read the relevant feature directory (e.g., `vllm_ascend/quantization/`, `vllm_ascend/spec_decode/`, `vllm_ascend/attention/`)
3. Check upstream vLLM for extension points: `vllm/platforms/`, `vllm/v1/attention/backends/`, `vllm/v1/sample/`, entry points in setup.py
4. Read [references/feature-patterns.md](references/feature-patterns.md) for implementation templates
5. For quantization/pooling/spec decode, also read [references/focus-areas.md](references/focus-areas.md)

### Step 3 — Design

Produce a design document (in Chinese) with:

1. **问题分析** — 需要解决的具体问题
2. **约束与假设** — NPU 硬件限制 + 明确假设
3. **高层设计** — 组件图（Mermaid）展示主要组件和数据流
4. **关键数据结构/接口** — Python class/dataclass 签名（无实现）
5. **关键路径** — 逐步执行流程（Mermaid sequence/flowchart）
6. **集成方式** — 使用 patch、继承、还是注册机制，以及为什么
7. **NPU 适配要点** — torch_npu API 使用、性能考量、ACL Graph 兼容性
8. **权衡** — 仅在有非显而易见的选择时

### Step 4 — Implement

Write core implementation code:

- Minimal, directly aligned with the design
- Match vllm-ascend codebase style (snake_case, type hints, 120 char lines)
- Organize by feature type:
  - **Patch**: Create `patch/{platform|worker}/patch_xxx.py`, add import to `__init__.py`, document in `patch/__init__.py`
  - **Attention backend**: Create `attention/xxx_v1.py`, register via `@register_backend`, add selection in `platform.py`
  - **Custom op**: Create `ops/xxx.py`, register via `register_custom_ops.py` or patch
  - **Quantization**: Create `quantization/methods/xxx.py`, register via `@register_scheme`, update `quant_type.py`
  - **Spec decode**: Create `spec_decode/xxx_proposer.py`, register in `get_spec_decode_method()`
  - **Worker/Scheduler**: Extend `NPUWorker`/`NPUModelRunner` or create scheduler subclass
- No test cases unless requested
- No unnecessary abstractions

### Step 5 — Save Document

Save the complete design document as Markdown to `./outputs/` (create if needed). Filename: `design-<feature-name>.md`.

The document must include:
- All sections from Step 3 (in Chinese)
- Code blocks with syntax highlighting
- At least one Mermaid diagram
- File structure showing where new/modified files live

Report the saved path to the user.

## Reference Files

| File | When to Read |
|------|-------------|
| [references/patch-system.md](references/patch-system.md) | When the feature requires patching upstream vLLM code |
| [references/feature-patterns.md](references/feature-patterns.md) | For implementation templates (attention, ops, worker, scheduler, platform, env vars, NPU constraints) |
| [references/focus-areas.md](references/focus-areas.md) | For quantization, pooling, or speculative decoding features |

## Communication Style

- Design documents in Chinese
- Precise, not verbose
- Use Mermaid diagrams for architecture and flow
- Use tables for comparisons
- State assumptions explicitly rather than guessing silently


vllm-ascend的当前量化特性代码整体比较冗余，并且量化分支很多，逻辑复杂。为了后面更好的维护vllm-ascend量化代码，提升量化易用性，对vllm-ascend的量化代码进行重构：1、只重构量化部分的代码，其他模块的代码哪怕有问题也不要动；2、输出整体的代码重构设计，梳理重构后的代码逻辑；3、输出详细的重构代码，并给出重构前后的差异对比，以及为什么要这么修改。

---

## Assistant

**Tool: skill**

---

## Assistant

**Tool: todowrite**

---

## Assistant

Let me do a thorough deep-dive into the specific redundancies across all quantization files.



**Tool: task**

**Tool: task**

**Tool: task**

---

## Assistant

**Tool: read**

**Tool: read**

**Tool: read**

**Tool: read**

**Tool: read**

---

## Assistant

**Tool: read**

---

## Assistant

**Tool: todowrite**

---

## Assistant

Now I have all the data. Let me write the comprehensive refactoring design document.



**Tool: todowrite**

---

## Assistant

**Tool: bash**

---

## Assistant

**Tool: write**

---

## Assistant

**Tool: todowrite**

---

## Assistant

重构设计文档已保存到 `/home/y00884608/code/.agents/skills/feature-design/outputs/design-quantization-refactor.md`。

**核心发现与重构方案（5 个阶段）**：

| 阶段 | 问题 | 方案 | 消除冗余 |
|------|------|------|:-------:|
| **R1** | 6 个 MoE `apply()` 复制粘贴 ~40 行公共骨架 | `AscendMoEScheme` 基类 Template Method，子类只覆写 3 个钩子 | **~240 行** |
| **R2** | 权重后处理 transpose→NZ→flatten 在 10 处重复 | 提取 `weight_utils.py` 公共函数 | **~88 行** |
| **R3** | `method_adapters.py` 混入 16 行 TP rank 解析 | 提取 `tp_utils.resolve_tp_rank()` | 解耦 |
| **R4** | 3 个 Config 类重复 isinstance dispatch | 提取 `base_config.py` 基类 | **~83 行** |
| **R5** | `modelslim_config.py` 839 行含 207 行纯数据字典 | 外移到 `model_mappings.py` | 解耦 |

**总计可减少约 430 行冗余代码**，同时显著提升可维护性。建议按 R5→R3→R2→R4→R1 顺序从低风险到高风险逐步实施。

---

## User

把刚刚的文档移动到outputs目录下

---

## Assistant

**Tool: bash**

---

## Assistant

**Tool: bash**

---

## Assistant

已移动到 `/home/y00884608/code/outputs/design-quantization-refactor.md`

---