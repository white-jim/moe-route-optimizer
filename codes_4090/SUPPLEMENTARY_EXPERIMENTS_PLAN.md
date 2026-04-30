# 补充实验实施说明

本文档面向后续在对应机器上实际改代码的 Codex/GPT。目标不是直接给出代码，而是给出一套可执行的实现路线，回答以下问题：

1. 当前这份代码是否支持补充实验。
2. 如果不完全支持，应该以什么方式增量修改。
3. 每类补充实验需要改哪些文件、记录哪些中间量。
4. 对同一个“模型 + 数据集”组合，补充实验应该一次性联采，还是一个一个做。

重要路径约定：

- 在**当前这台机器**上，本文档位于 `codes_4090/` 目录下。
- 在**另一台实际执行修改的机器**上，应当把 **当前 `codes_4090` 目录里的内容**视为**项目根目录**。
- 因此，本文档后续所有路径都按“项目根目录相对路径”书写，不再带 `codes_4090/` 前缀。

例如：

- 当前机器上的 `codes_4090/moe_route_optimizer/main.py`
- 到另一台机器上，应理解为 `moe_route_optimizer/main.py`

本文档只讨论这套代码，不讨论当前仓库根目录下的另一套实现。

---

## 1. 当前代码现状

### 1.1 当前主链路

`moe_route_optimizer/main.py` 当前实际使用的是 `HF Accelerate Adapter`：

- `moe_route_optimizer/main.py`
- `moe_route_optimizer/interfaces/hf_accelerate_adapter.py`

主训练流程中：

- 在第一个 `MoE block` 上注册扰动 hook。
- 训练阶段反复在固定 batch 上做推理。
- hook 会记录原始 hidden states、被选中的 token、被选中的维度、扰动后 hidden states、log_prob。
- 当前 latency 主要稳定拿到：
  - 端到端推理时间
  - EP all-to-all 通信总时间

### 1.2 当前已经具备的数据

当前实现已经能拿到：

- 原始 `hidden_states`
- 扰动后的 `perturbed_hidden_states`
- `selected_indices`
- `perturb_dim_indices`
- `log_prob`
- 端到端推理时间
- EP A2A 通信时间

因此以下分析已经有基础：

- 策略选择了哪些 token
- 策略对哪些维度做了置零
- 扰动前后表示差异 `delta H`
- 端到端与 A2A 的关系

### 1.3 当前缺失的关键量

当前实现没有稳定记录：

- 各层 `token -> expert` 路由结果
- 各层 top-k 专家分配
- 各层专家负载统计结果的持久化
- `router` 时间
- `expert compute` 时间
- 每层 A2A 时间的对外返回
- 受控的扰动强度参数扫描

因此补充实验不是“零修改直接可做”，但多数都属于**增量可实现**。

---

## 2. 核心原则：展示对象如果使用 Train 结果，哪些实验可以做

如果展示对象明确限定为：

- “训练阶段样本”
- “机制分析 / case study”
- “策略行为分析”

则补充实验是合理的。

但要注意：

- 可以用 train 样本做**解释性展示**
- 不要把 train 样本上的现象直接写成**泛化结论**

推荐文案口径：

- “在训练样本上观察到……”
- “以代表性训练样本为例……”
- “训练过程中策略倾向于……”
- “作为机制验证，我们对训练阶段样本进行了……”

不推荐直接写：

- “方法整体证明了……”
- “在未见样本上普遍如此……”

---

## 3. 建议的总实现策略

不要为每个补充实验单独临时打补丁。更合理的方式是分两步。

### 第一步：先做统一的分析埋点层

先把以下中间量统一记录下来：

- 每层路由结果
- 每层专家负载
- 每层 router / expert / dispatch / combine 时间
- 当前 hook 选择的 token / 维度
- 扰动前后 hidden states
- prompt、token ids、生成文本

建议把这部分做成“分析模式”或“analysis collection mode”，不要直接污染原训练逻辑。

### 第二步：基于同一批分析产物做多个图/表

如果共享的是同一份分析产物，则同一个模型 + 数据集的一次 instrumented run 可以同时支持：

- 路由热力图
- 专家负载变化
- 策略 token 选择统计
- A2A 通信来源分析
- 实际 `||delta H||` 与路由变化的相关性分析

这样可以避免同一组合重复跑太多次。

---

## 4. 建议新增的“分析模式”

### 4.1 目标

建议新增一个独立的分析模式，而不是直接复用现有 train/eval 开关。

原因：

- 当前 `training=True` 时会收集状态，但扰动是采样式的。
- 当前 `training=False` 时是确定性选择，但默认不收集状态。

对于论文里的 case study，通常更希望：

- 使用确定性策略
- 同时收集状态
- 对同一输入做“扰动前 / 扰动后”的稳定对比

### 4.2 行为定义

建议引入一个明确的分析模式，语义如下：

- `enabled=False`：基线推理，不加扰动，但记录原始路由/时延
- `enabled=True, deterministic=True, collect_analysis=True`：加扰动，且记录完整分析信息

这样可以对同一输入做一对严格可比的 before/after 结果。

### 4.3 需要调整的文件

- `moe_route_optimizer/hooks/hook_manager.py`
- `moe_route_optimizer/interfaces/hf_accelerate_adapter.py`
- `moe_route_optimizer/main.py`

重点不是改 PPO，而是让“收集状态”和“是否训练模式”解耦。

---

## 5. 各补充实验的实现路线

## 5.1 Case Study：扰动前后专家路由行为分析

### 目标

对同一个输入样本，比较：

- 扰动前的各层 token-to-expert 分配
- 扰动后的各层 token-to-expert 分配

并绘制：

- token × layer 的专家分配热力图
- top-k 专家变化图
- 专家负载变化图

### 当前可行性判断

**可做，但需要新增路由记录。**

当前各模型 wrapper 内部实际上都能拿到路由结果：

- Qwen：`router_logits / topk_indices / topk_weights`
- LLaMA-MoE：`topK_indices / topK_scores`
- JetMoE：`expert_size / batch_index / router_logits`
- DeepSeek：`topk_idx / topk_weight`

问题不在“没有算”，而在“算了但没有保存/回传”。

### 需要修改的文件

#### A. `moe_route_optimizer/interfaces/hf_accelerate_adapter.py`

在各类 EP wrapper 中增加“路由 trace 记录”。

建议按模型 wrapper 分别记录：

- `layer_idx`
- `num_tokens`
- `topk_indices`
- `topk_weights` 或 `topk_scores`
- 如果方便，补充 `router_logits`
- 每个 expert 的 token count

建议记录点：

- Qwen wrapper
- LLaMA-MoE wrapper
- JetMoE wrapper
- DeepSeek wrapper

统一封装为 route trace 结构，再挂到 worker 本地 buffer。

#### B. `moe_route_optimizer/interfaces/framework_interface.py`

如果后续希望接口层清晰，建议补一个分析接口，例如：

- 获取最近一次推理的 route traces
- 获取最近一次推理的 per-layer stats

这一步不是强制，但有助于避免所有逻辑都耦合在 adapter 私有变量里。

#### C. `moe_route_optimizer/main.py`

新增一个“analysis collection entry”或“case study runner”。

流程建议：

1. 选定一个样本或一个小 batch
2. 跑一次 baseline（hook disabled，但 route trace enabled）
3. 跑一次 perturbed（hook enabled + deterministic + route trace enabled）
4. 导出统一的分析文件

### 输出建议

建议输出到类似目录：

- `analysis_outputs/<run_name>/routing_case_study/`

文件建议：

- `baseline_routes.pt` / `baseline_routes.json`
- `perturbed_routes.pt` / `perturbed_routes.json`
- `sample_meta.json`

### 可视化建议

先不要在训练主代码里直接画图。  
建议先导出结构化数据，再用单独脚本画图。

### 结论

这是**增量可做**实验，优先级高。

---

## 5.2 专家负载分布变化分析

### 目标

验证扰动后是否出现：

- 高负载专家使用下降
- 低利用率专家参与上升
- 分配更集中或更均衡

### 当前可行性判断

**比热力图更容易做。**

因为它只需要：

- 每层每个 expert 命中的 token 数

而这些统计在 wrapper 内部都能从 top-k 分配直接得到。

### 需要修改的文件

主要还是：

- `moe_route_optimizer/interfaces/hf_accelerate_adapter.py`

### 建议记录字段

每层记录：

- `expert_token_counts`
- `tokens_per_gpu`
- `send_counts`
- `recv_counts`

### 输出建议

这部分可以和 routing case study 共用一份 route trace 数据，不必单独跑新实验。

### 结论

这是**低到中等改动可做**实验，建议和路由热力图一起实现。

---

## 5.3 推理加速来源分析

### 目标

把推理时间拆成：

- router 计算时间
- expert 计算时间
- A2A 通信时间

并比较扰动前后变化。

### 当前可行性判断

当前只能稳定做：

- 端到端时间
- EP A2A 总时间

因此：

- “端到端 vs A2A” 现在就能做
- “router / expert / A2A 三段完整拆解” 需要新增埋点

### 需要修改的文件

#### A. `moe_route_optimizer/interfaces/hf_accelerate_adapter.py`

在各 wrapper 的 `forward` 中新增分段计时。

建议统一分成：

- `router_time`
- `dispatch_a2a_time`
- `local_expert_time`
- `combine_a2a_time`

注意：

- 当前 A2A 时间已经有 CUDA Event 计时
- 需要补的是 router 和 local expert
- 每层都应独立累计

#### B. `moe_route_optimizer/interfaces/framework_interface.py`

建议把接口补全，至少支持：

- 获取 per-layer latency breakdown
- 获取最近一次推理的 breakdown summary

#### C. `moe_route_optimizer/main.py`

增加分析导出逻辑，而不是只打日志。

### 输出建议

建议输出：

- `latency_breakdown_baseline.json`
- `latency_breakdown_perturbed.json`

字段包含：

- 总时间
- A2A 总时间
- 每层 router / expert / dispatch / combine

### 结论

这是**中等改动可做**实验。  
若论文只需要“加速来源主要包括通信和负载变化”，则先做：

- 端到端
- A2A
- 专家负载变化

即可支撑第一版结论。

---

## 5.4 扰动策略行为分析

### 目标

分析：

- 被选中 token 的位置分布
- 被选中 token 的类型分布
- 不同任务上策略偏好哪些 token

### 当前可行性判断

**当前最容易做。**

因为 `selected_indices`、`perturb_dim_indices`、`hidden_states` 已经在现有 hook 状态中存在。

### 需要修改的文件

#### A. `moe_route_optimizer/hooks/hook_manager.py`

当前已经收集了：

- `selected_indices`
- `perturb_dim_indices`
- `hidden_states`

可以保持不动，或者补充更多上下文信息，例如：

- 当前 prompt id
- batch 内样本 id

#### B. `moe_route_optimizer/interfaces/hf_accelerate_adapter.py`

建议在回传 hook state 的同时，回传：

- prompt 文本
- tokenizer 编码结果或 token ids
- 生成文本

这样后处理可以直接把 `selected_indices` 映射回 token 字符串。

### 语义类型分析建议

建议分层次做：

第一层，低成本规则：

- 数字
- 标点/符号
- 大写字母
- 特殊 token
- 长词 / 短词

第二层，中成本：

- 关键词字典
- 问题词 / 逻辑词 / 实体词规则

第三层，高成本：

- 词性标注
- 命名实体识别

如果只做第一版论文补充，优先推荐前两层。

### 结论

这是**低改动可做**实验，建议优先做。

---

## 5.5 扰动强度与路由变化关系分析

### 目标

研究：

- `||delta H||` 增大时，路由分布如何变化
- top-k 变化率或 KL 散度如何变化

### 当前可行性判断

分两种情况。

#### 情况 A：只做观察性分析

即：

- 使用训练/分析过程中自然产生的扰动
- 离线计算每次的实际 `||delta H||`
- 再和路由变化指标做关联分析

这种是**可做的**。

#### 情况 B：做受控强度扫描

即：

- 人为控制 perturbation scale
- 扫多组强度
- 画响应曲线

这种当前**还不够**。

原因是：

- `PerturbationConfig.perturbation_scale` 虽然存在
- 但实际扰动实现里没有真正使用它
- 当前扰动是“选中的维度直接置零”

### 需要修改的文件

#### A. `moe_route_optimizer/core/perturbation_generator.py`

如果要做受控强度扫描，需要把“扰动强度”真正接入实现。

推荐两种思路：

1. 在当前“置零”方案上定义一个软化版本  
   例如不再直接置零，而是按比例衰减。

2. 保留置零逻辑，但把“选中多少 token / 多少维度”作为强度代理  
   这样可以做离散强度扫描，而不是连续 scale。

#### B. `moe_route_optimizer/hooks/hook_manager.py`

当前已经保存了：

- 原始 `hidden_states`
- 扰动后 `perturbed_hidden_states`

后处理可直接算：

- `delta H`
- `||delta H||`

### 路由变化指标建议

优先级建议如下：

1. `Top-k` 专家变化率  
   最容易统一支持所有模型。

2. 专家负载分布差异  
   可直接用 load 向量比较。

3. KL 散度  
   只有在各模型都稳定拿到完整路由分布时再做。

### 结论

这是**观察性版本可做，受控扫描版本需要改扰动定义**的实验。

---

## 5.6 不同任务/Prompt 类型敏感性分析

### 目标

比较不同任务：

- 长文本
- 多选题
- 逻辑推理
- 数学推理

对扰动的敏感程度。

### 当前可行性判断

**可做。**

因为数据集 evaluator 已经有多任务支持，且 train 阶段本就会反复使用固定 batch。

但要注意：

- 这类分析更适合做“同一模型，不同数据集”的横向比较
- 不建议把不同模型、不同数据集同时混起来解释

### 推荐做法

对于一个固定模型，分别在多个数据集上跑分析模式，记录：

- `||delta H||`
- top-k 变化率
- A2A 变化
- accuracy 变化

再比较分布差异。

### 结论

这是**在前述数据记录完成后可直接开展**的实验，不需要再为它单独设计底层埋点。

---

## 6. 一个模型 + 一个数据集：补充实验是一个一个做，还是一次性做几个

结论很明确：

### 6.1 可以共用一次 instrumented run 的实验

如果你已经做了统一分析埋点，那么对**同一个模型 + 同一个数据集 + 同一版策略**，以下实验可以一次性联采：

- 路由热力图
- 专家负载变化
- 策略 token 选择分布
- A2A 通信来源分析
- 实际 `||delta H||` 与路由变化的相关性分析

原因：

- 它们都依赖同一批中间量
- 重复跑没有必要
- 一次性采集还能保证图表来自同一个策略快照

### 6.2 建议单独跑的实验

以下实验建议单独跑：

- 扰动强度扫描
- 不同超参数下的对比实验
- 不同 checkpoint 间的比较
- baseline / perturbed 的严格确定性 replay

原因：

- 这些实验会改变策略或扰动定义本身
- 和联采型机制分析不是同一种 run

### 6.3 推荐工作流

对于一个模型 + 数据集组合，推荐这样做：

#### 阶段 1：先跑正常 train，挑一个代表性 checkpoint

目标：

- 找到一个有代表性的策略状态
- 不一定要求绝对最优，但要结果稳定、可解释

#### 阶段 2：基于该 checkpoint 做一次“分析模式采集 run”

这一次 run 同时采：

- baseline route trace
- perturbed route trace
- hook states
- latency breakdown
- token 选择统计

#### 阶段 3：离线生成多个补充实验结果

从同一份分析产物导出：

- 路由热力图
- 专家负载图
- token 位置/类别统计
- A2A / end-to-end 统计

#### 阶段 4：如果要做强度分析，再单独做 sweep run

不要把强度扫描混到阶段 2 里。

---

## 7. 推荐的修改优先级

### 第一优先级

这些改完，就能支撑大部分补充实验：

1. 分析模式：deterministic 且允许收集状态
2. route trace 记录与回传
3. 专家负载统计导出
4. prompt/token id/hook state 统一导出

对应文件：

- `moe_route_optimizer/hooks/hook_manager.py`
- `moe_route_optimizer/interfaces/hf_accelerate_adapter.py`
- `moe_route_optimizer/main.py`

### 第二优先级

如果要把“加速来源”写得更完整，再补：

5. router / expert / dispatch / combine 分段计时
6. 每层 breakdown 返回接口

### 第三优先级

如果要做强度曲线，再补：

7. `perturbation_scale` 真正接入扰动实现
8. sweep 驱动脚本

---

## 8. 不建议的做法

### 8.1 不要直接在训练主循环里即时画图

建议：

- 训练 / 分析 run 只负责导出结构化数据
- 单独用脚本画图

原因：

- 训练主链路应尽量稳定
- 画图逻辑不该和分布式推理耦合

### 8.2 不要把所有模型的路由统一成一个“抽象字段”后再硬凑

建议：

- route trace 统一字段可以有
- 但模型特有字段应保留

例如：

- Qwen 保留 `router_logits`
- LLaMA-MoE 保留 `topK_scores`
- JetMoE 保留 `expert_size / batch_index`

否则后续分析会丢失细节。

### 8.3 不要一开始就追“所有层全部可视化”

建议先做：

- 第一个 MoE 层
- 或少数代表层

这样修改范围更可控，也更容易先得到论文图。

---

## 9. 建议的文档结论

如果后续实现顺利，本目录下这套代码最适合支撑的补充实验顺序是：

1. 策略行为分析
2. 第一层或少数层路由热力图
3. 专家负载变化
4. 端到端 vs A2A 分析
5. 不同任务敏感性比较
6. 扰动强度扫描

其中：

- 1~4 可以共享一次分析采集 run
- 5 需要多数据集横向比较
- 6 最适合单独做 sweep

---

## 10. 最终结论

对于当前这份代码：

- **不是不能做补充实验**
- **而是应先补“统一分析埋点层”**

在“train 样本机制展示”的前提下：

- 路由行为分析：可做
- 专家负载分析：可做
- 策略行为分析：可做
- A2A 来源分析：可做
- 完整时延三段拆解：增量可做
- 扰动强度受控扫描：需要再扩展扰动实现

对一个固定的模型 + 数据集组合：

- **机制类补充实验建议一次 instrumented run 联采多个**
- **参数扫描 / 强度扫描 / checkpoint 对比建议分开跑**

这就是最务实、最稳定的实施方式。
