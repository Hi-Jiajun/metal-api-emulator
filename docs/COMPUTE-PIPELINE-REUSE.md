# compute 半边按形状复用 pipeline 对象（E 侧第七刀）

本文件记录一次 provider 增量：**compute 半边（`VulkanComputeProvider` 的提交路径）每次提交
重建一次的 pipeline 对象组，不再每次重建，而是按「模块 + 布局 + 常量 + local size」这一形状
复用上一次同形状提交留下的四个设备对象**。它接在 `docs/SUBMIT-PHASE-PROFILE.md` 的相位表上，
是 G3-A（残差拆条）点名的第二名、G3-C §5.2 的第一个候选，成法照渲染半边第三刀的
`docs/RENDER-SETUP-REUSE.md`（`crate::render_setup_reuse`）。

## 0. 缺口：每提交重建一次的那一组对象

G3-A 轮（reims `04f5d2e` × provider `883b60a`，166 窗 / 42 496 次提交）与 G3-C 的三臂
（reims `35dcd14` × provider `8e2836e`，300 s × 3）读到的形状一致：

| 条 | G3-A µs/提交 | G3-C 三臂 µs/提交（开 / 关 / 开） | 对象数/提交 |
|---|---|---|---|
| `rb_pipeline` | **118.6** | **98.0 / 96.7 / 89.0** | `rb_pipeline_n` **1.020**（三臂恒定） |
| `submit_td_pipeline` | **6.5** | 5.1 / 5.2 / 4.7 | 同组 |
| 合计 | **≈125** | ≈103 / 102 / 94 | — |

代价分成两半：**创建侧**（`rb_pipeline`）包住整次
`ExecutionResources::create_pipeline_objects`，**销毁侧**（`submit_td_pipeline`）是同一个组在
提交结束时被 `drop`。`rb_pipeline_n` 恒定 1.020 说明**每一个提交都建了自己的那一个组**，而
`PipelineObjects::create` 一次要建四个对象：

| 调用 | 对象 |
|---|---|
| `vkCreateShaderModule` | 这条 kernel 的翻译结果 |
| `vkCreateDescriptorSetLayout` | 反射列出的那组 binding |
| `vkCreatePipelineLayout` | 上面这个 set layout + 反射的 push-constant range |
| `vkCreateComputePipelines`（每个 local size 一条） | 该模块在该 threadgroup 尺寸下的 pipeline |

guest 桌面每帧提交的是同一批 kernel、同一批 threadgroup 尺寸，而 provider 每一提交都按新对象
重建一次。这四个对象的**身份**只由「即将交给驱动的那几个结构」决定；渲染半边的
`render_setup_reuse` 早就在做同一件事（`reuse_hit_n` 0.998），compute 半边没有。

## 1. 键就是交给驱动的字段

`ComputePipelineKey` 逐字段读回**即将交给驱动**的那些结构，不是另立一份描述
（`crate::compute_pipeline_reuse` 的模块文档有同一张表）：

| 对象 | 键读回的字段 |
|---|---|
| `VkShaderModule` | `VkShaderModuleCreateInfo::code` 的 SPIR-V words |
| `VkDescriptorSetLayout` | 按 binding 号排好序的 `VkDescriptorSetLayoutBinding` 列表（binding / descriptor type / count / stage bits；immutable sampler 列在每个站点都为空，用 `debug_assert` 钉住） |
| `VkPipelineLayout` | 上面那一个 set layout + 反射 kernel 契约给出的 `VkPushConstantRange`（offset / size；stage bit 恒为 `COMPUTE`，同样钉住） |
| `VkPipeline`（每个 local size 一条） | 模块、入口名（`main`，每个站点都是同一常量）、`COMPUTE` stage bit、layout、特化数据——特化数据**就是** local size 的三个字 |

键里 `local_sizes` 去重后升序排列，所以同一个 shape 的两个 plan 只是 region 顺序不同时共用一条
表项。比较是这四个字段的逐字段相等（`words` / `bindings` / `push_constant` / `local_sizes`），
摘要（FNV-1a）只用来分桶：碰撞的代价是一次比较，不是一个错误的复用。

**为什么复用是安全的**：驱动为一对「定义相同」的创建造出来的对象按定义可互换——
`VkDescriptorSetLayout` 的兼容性定义在它的 bindings 上，`VkPipelineLayout` 的定义是它的 set
layout 列表与 push constant 范围。所以复用的 pipeline layout 可以与**本提交自己新建的
descriptor set** 配对：那些 set 是从这个键逐字段比较过的 layout 分配出来的。驱动的默认值
（结构的 `flags` 字、从未设置的 `pNext` 链、`VkPipelineCache::null()`）对每一次创建都是一样的，
因此不承载形状，也不进键。

## 2. 命中省掉什么、不省什么

命中时跳过的是 `vkCreateShaderModule`、`vkCreateDescriptorSetLayout`、
`vkCreatePipelineLayout` 与每个 local size 一次 `vkCreateComputePipelines`。**不跳过**的是：
本提交自己的 descriptor pool、descriptor set、command pool / buffer / fence、每一条 buffer、
record / submit / fence / readback。因此提交产出的字节按构造不变；`tests/
compute_pipeline_reuse_e2e.rs` 仍然逐臂比较它们。

### 2.1 什么时候还回去

组在 `take` 时离开表（`PipelineObjects` 持有它），只有**观察到自己的 fence** 的提交才还：
`ExecutionResources::drop` 走销毁路径时 `completed && !device_lost` 为真才把组交给表，否则
照旧销毁。因此表里**永远不会**放着一条命令缓冲还可能正在执行的对象：离开表的是本提交自己的，
还回来的是 fence 之后的。

### 2.2 失败方向

`ReusablePipelineGroup` 自己带 `Some(device)` 时用 `Drop` 释放；表里的表项是**卸下 device 的**
（`device: None`），由表自己的 device 在逐出 / 关开关 / 上下文拆卸时释放。于是：

* 提交在 fence 之前失败（编码失败、被拒绝、取消）→ 组仍带 device，`Drop` 释放它，表里没有；
* 观察到设备丢失 → 同上，`returning_buffers` 为假，不还；
* 开关关闭 → 表清空并销毁持有的每一条表项；
* 命中后又失败 → 组已经离开表，没有任何一半留在表里被下一个提交取到。

上限 64 条表项，超限按最久未用逐出（在移除它的那把锁里销毁）；开关关闭时同样销毁。同一形状
已有表项时，还回来的第二份直接销毁（两条提交在冷表上同时建，多余的这份不留）。

### 2.3 失效与生命周期

* **模块 / 布局销毁**：表持有它们，逐出、清空、上下文拆卸三条路径都会销毁；每条路径都在
  device 还活着时做。
* **注册面变化**：compute 注册被 `release_pipeline` 摘掉时整张表清空——与渲染半边
  `invalidate_render_setup_reuse` 同一条规则。这一条是**内存上界**而不是正确性要求：键是模块
  自己的 SPIR-V，重新注册的函数若翻译结果不同就是不相同的 words，本来也碰不到同一条表项。
* **设备丢失**：丢设备的那条提交不还；上下文随设备消失时表随之消失。
* **开关关闭**：`set_enabled(false)` 先 `clear()` 再落开关。
* **没有「没键」这一臂**：与渲染半边不同，compute 的创建总能陈述自己的 words、bindings、
  push-constant range 与 local size，所以每一次创建都可以入表（`crate::render_setup_reuse`
  的 `reuse_unkeyed_n` 在这里没有对应物）。

## 3. 开关与读数

`METAL_API_VULKAN_COMPUTE_PIPELINE_REUSE=0`（也接受 `off` / `no` / `false` 及其大写）关掉整个
机制：不取、不留，每次创建只有一次 relaxed load，这就是对照臂。相位行新增
`compute_pipeline_hit_n` / `compute_pipeline_miss_n` / `compute_pipeline_mismatch_n` /
`compute_pipeline_disabled_n` / `compute_pipeline_return_n` / `compute_pipeline_drop_n` 六个
计数，把「没人问」「问了但拒绝」「摘要碰撞被全字段比较拒绝」分开；进程两侧的累计读数在
`ComputePipelineReuseCounts`（`provider.compute_pipeline_reuse_counts()`），测试自己的两条臂用
`provider.set_compute_pipeline_reuse(bool)`。

这六个与 `rb_pipeline_n` 是同一个种群的两种读法：`hit + miss + mismatch + disabled` 等于这一
窗口里**每一次创建**，`return + drop` 等于每一次 hand-back。`rb_pipeline_n` 本身的口径也跟着
变了：它原来数的是「这一窗口建了几个 pipeline 组」，现在只数**真的建了的**——被表服务的那次
没有建对象，也就不再计数。渲染半边的同名计数是 `reuse_*_n`，两组分开是因为两条 rail 有自己的
开关。

## 4. A/B 读数

三臂同一个 exe 字节（sha256 `c5e7c72a7a4e…`，`40-ab-exe-identity.txt`），launcher 只差
`METAL_API_VULKAN_COMPUTE_PIPELINE_REUSE=0` 一行；坐标 reims `35dcd14` × provider `f69a9a8`，
300 s × 3 背靠背（臂间 settle 75 s），三臂都 attempt=1 / alive / 0 panic / `BOOT_EXIT=0`。
完整读数、形状表与红线在 `evidence/compute-pipeline-reuse-f69a9a8-2026-09-20/`。

| µs/提交（窗口和 / Σn） | g3d2（开） | g3d2-off（关） | g3d2-b（开） | 关 → 开（两开臂均值） |
|---|---|---|---|---|
| **`total`** | **2 103.3** | **2 181.7** | **2 069.0** | **−95.5（−4.4%）** |
| `resource_build` | 60.4 | 142.2 | 54.1 | −85.0 |
| ├ **`rb_pipeline`** | **5.9** | **92.9** | **4.1** | **−87.9（−94.6%）** |
| ├ `rb_buffer`（未被碰） | 49.2 | 44.7 | 44.7 | +2.2 |
| └ `rb_descriptor`（未被碰） | 4.9 | 4.0 | 4.8 | +0.8 |
| **`submit_td_pipeline`** | **0.5** | **5.4** | **0.5** | **−4.9（−90.8%）** |
| `render_readback`（对照条） | 97.1 | 81.6 | 92.5 | +13.2（+16.1%） |
| `render_landing`（对照条） | 59.1 | 59.8 | 61.4 | +0.4（+0.7%） |
| `render_teardown`（对照条） | 131.8 | 135.5 | 131.1 | −4.1（−3.0%） |
| `fence_wait` | 93.7 | 143.8 | 97.1 | −48.5（−33.7%） |
| `plan` / `pool` / `submit_validate`（纯 CPU，与本刀无交集） | 145.2 / 187.7 / 144.2 | 131.8 / 166.4 / 134.3 | 143.5 / 183.2 / 151.8 | +12.6 / +19.1 / +13.7 |
| 机制计数 `hit_n` / `miss_n` / `disabled_n` / `return_n` / `drop_n` | **1.000 / 0.020 / 0 / 1.000 / 0.020** | 0 / 0 / **1.020** / 0 / **1.020** | **1.000 / 0.020 / 0 / 1.000 / 0.020** | — |
| 人口 `rb_pipeline_n`（真的建了几个组） | **0.020** | **1.020** | **0.020** | — |
| 提交数 / 窗口数 | 59 904 / 234 | 59 904 / 234 | 62 208 / 243 | — |

**R 侧独立读数**（帧剖面，与相位仪无关）：`prov_submit_us_mean` 与窗口的 `draws` 是强相关
（r ≈ 0.92–0.97），按窗口归一后的每 draw 成本中位 **2 092 / 2 187 / 2 102 µs**（关 → 开臂均值
**−90 µs，−4.1%**）、均值 **3 148 / 3 076 / 2 685**（**−159，−5.2%**），与 E 侧 `total` 的 −4.4%
同向同量级。

**这一刀砍掉什么**

* **创建侧**：`rb_pipeline` 92.9 → 5.0（−94.6%）。开臂剩下的 5 µs 是「算键 + 查表 + 取回」本身，
  以及每约 50 次提交一次真的建组（`rb_pipeline_n` 0.020）——本轮 guest 一共只陈述了 10 个形状。
* **销毁侧**：`submit_td_pipeline` 5.4 → 0.5（−90.8%）：fence 之后由「销毁四个对象」变成
  「交给表」。

**没砍的、与边界**

* 三臂之间仍有宿主状态带漂移（`fence_wait` −48.5、`render_readback` +13.2、`plan` +12.6、
  `pool` +19.1…，加总约 +40 µs/提交），所以 `total` 的 −95.5 与机制自身两条的 −92.8 不是
  同一个数；G3-C 的同一现象记在 `docs/COMPUTE-BUFFER-POOL.md` §4。
* `render_readback` 移动了 +16.1%（两开臂 97.1 / 92.5 对关臂 81.6）。它在前一轮（`g3d` 三臂，
  修计数缺陷之前）以同一方向复现（96.0 / 93.0 对 86.5），而本刀不碰渲染半边的 readback
  路径、两臂 capture 又逐字节相同，所以这里只把它如实记为「机制以外的一处系统性偏移」。
* 覆盖面无回归：三臂都是 `out_of_class_shape_lines distinct=0`、`render_seam
  ok/ok_resident/out_of_class = 0/0/0`、`compute_provider_out_of_class=47`（与 g3a/g3c 同值），
  `pass_color_slots_1 == render_provider_canonical`（60 023 / 60 128 / 62 386），三个形状表
  **逐字节相同**（sha256 `c15ebae2…`），五个零读保持 0。
* 红线上，生产三项与轮首逐字节相同、`find -newer` 为空；像素口径不可用（QMP `screendump`
  三臂全黑，与 g3a/g3c 同，属该 display 的已知边界）。

## 5. 这一刀不碰什么

* 不改任何 wire / 契约 / 能力位：`PipelineObjects` 的四个对象之外没有新对象，提交产出的字节
  与 writeback 逐字节不变。
* 不池化 descriptor pool / descriptor set / command buffer / fence：它们不是形状决定的
  （绑定内容与队列状态是每次提交自己的）。
* 不动渲染半边的 `render_setup_reuse`：两条 rail 各自的开关与计数保持独立。
* `submit_validate` 的读数与本刀无交集（G3-C 三臂里三臂一致，见 §6）。

## 6. `submit_validate` 的重复派生：先量后判

G3-A / G3-C 都把 `submit_validate`（131.0 / 128.2 / 124.3 µs/提交，纯 CPU、三臂一致）列为
第三名，并指出它的成本不在 writeback 数量上（`writebacks_us` 只有 0.4–0.6 µs/提交），而在
`validate_writebacks_for_trace` **重新解出**一遍 `trace.serial_resources()` 与
`trace.serial_texture_resources()`——同一份资源池在 `plan` / `pool` 里已经解过一次。读代码
还多一层：`serial_resources()` 自己先调 `validate_serial_buffer_reuse()`，而后者第一句就是
`self.validate()`，也就是**把已经通过 admission 的 trace 再整体校验一遍**。

本轮先把这段重复量清（测量构建 + 探针，见 `evidence/compute-pipeline-reuse-probe-*`），
再决定落不落第二刀。测量结果与结论见 §6.1。

### 6.1 测量结果

**测量轮**：census tag `g3dprobe`（reims `35dcd14` × provider `6359bba`，150 s 驻留、
`ONLY=8` + `import=on`，attempt 2 alive / 0 panic / `BOOT_EXIT=0`）。探针（一处临时
`eprintln`，**已从本轮正式提交里移除**）在每个 `submit_validate` 区间里对**同一条 trace**
多测四次：`validate_for_trace` 的第一/二/三次调用，以及 `trace.validate()`、
`trace.serial_resources()`、`trace.serial_texture_resources()` 各自单独调用的耗时。
原始 38 652 行在 `evidence/gate3-census-g3dprobe-2026-09-20/g3dprobe-qemu-boot.log`，
提取脚本与该脚本的输出在同目录的 `g3dprobe-validate-split.{py,txt}`。

**这一轮的 trace 形状**（38 652 次提交）：`passes` ∈ {2, 4}（37 873 / 779）、
`compute_passes` ∈ {1, 2}、`render_passes` **恒 1**、`buffer views` ∈ {1, 2}、
`texture views` **恒 0**、`attachments` **恒 1**、`writebacks` ∈ {0, 1}
（20 013 / 18 639，即约一半提交没有可读回写的落地）、`writeback_bytes` 从 16 KiB 到 74 KiB 余。

**耗时（µs/次）**：

| 测点 | mean | median | p90 | p99 | max |
|---|---|---|---|---|---|
| `validate_for_trace`（第一次调用 = 相位条量到的那次） | **136.81** | **19.60** | 55.2 | 1 818.5 | 16 289.0 |
| 同一调用第二次 | 112.11 | 7.70 | 20.3 | 1 690.8 | 7 819.9 |
| 同一调用第三次 | 110.43 | 7.20 | 19.7 | 1 674.6 | 7 676.4 |
| `trace.validate()` | 0.55 | 0.40 | 0.6 | 3.7 | 15.1 |
| `serial_resources()`（内含一次 `trace.validate()`） | 87.48 | 4.40 | 13.2 | 1 389.0 | 8 706.4 |
| `serial_texture_resources()` | 0.05 | 0.00 | 0.1 | 0.2 | 5.8 |

**结论：这条相位条不是一个可 memo 的工作量。**

1. **均值是尾巴，不是工作。** 7.37 % 的调用 >100 µs（均值 1 570.8 µs），它们占全部时间的
   **84.6 %**；其余 92.6 % 的调用均值只有 22.8 µs、中位 19.6 µs。也就是说
   `submit_validate` 的 131.0 / 128.2 / 124.3 µs/提交（G3-C 三臂）量到的是**同一形态的
   长尾**，而不是每次提交都要付的 CPU。
2. **尾巴是突发性的，不是本区域的代码。** >1 ms 的调用占 6.95 %，**相邻两次之间的中位
   间隔是 1 次提交**，且 `corr(total_i, total_{i+1}) = 0.429`——停顿成串出现，是宿主调度/
   内存路径的签名，而不是一条每次都会走的代码路径。（同样地，`writebacks=0` 与
   `writebacks=1` 两种形状的中位只差 4.6 µs（22.20 / 17.60），而均值差 3.8 倍
   （211.8 / 56.2）——差在尾巴率 11.63 % / 2.79 %。）
3. **任务点名的"重复派生"在尾巴里也极小。** 即使在 >100 µs 的那 2 847 次调用里，
   `trace.validate()` 的均值也只有 **2.254 µs**（max 15.1 µs）：要删掉的那一次整体校验，
   在最长的那次调用（16 289 µs）里也只占 2.2 µs。
4. **它本身有多大**：在 ≤100 µs 的常态调用里，三项加起来 **5.31 µs = 该次调用的 23.3 %**
   （`trace_validate` 0.415 + `serial_resources` 4.854 + `texture_resources` 0.042），其中
   **可以省掉的只有"再解一次池"这一份**（即 `serial_resources` + `serial_texture_resources`
   合计 ≈4.9 µs，以及 `serial_resources` 自己多跑的那一次 0.4 µs 校验）。对照 G3-C 的整笔
   `total` 2 208.1 µs/提交，**这是 0.2 %**；对照它自己那条被长尾抬起来的 131 µs，
   一次 A/B 也只会看到噪声。

**处置：不落第二刀。** 把一个 4.9 µs（0.2 %）的重复派生收掉，代价是给核心加一个"调用者自己
交池、跳过重校验"的入口——那是一条**能绕过校验**的新接口，为一个落在噪声下的收益扩宽契约面
不划算。G3-A 把它排到第三名（4.4 %）是因为相位条读的是窗口和 / Σn 的**均值**，而这条的均值
由长尾决定；这一节把这件事量清并留档，等真正需要它的时候（例如批量提交把每次提交的常数项
乘掉之后）再按同一份数据重估。

**顺带一条给相位表的判读**：任何一条 region 的读数如果不能用"每次提交都要付"解释，就应该像
这一轮一样先看分布（中位 vs 均值、尾巴率、尾巴的时序聚集度）再决定要不要动刀。
