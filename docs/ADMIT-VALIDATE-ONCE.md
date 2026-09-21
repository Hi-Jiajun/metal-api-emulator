# 准入走查：一次走路只校验一遍 trace

E 侧第十一刀。开关 `METAL_API_CORE_VALIDATE_ONCE`（**默认关**），机制与证明写在
`crates/metal-api-core/src/admit_validate_once.rs` 的模块文档里；本文记录
**为什么是这一处**、**三臂的读数**与**不碰什么**。

## 0. 缺口：同一条走路的 55% 是同一件工作被做了三遍

第十刀的拆条（`docs/SUBMIT-ADMIT-PROFILE.md`）把一次准入走路切成 15 段。开臂的生产姿态
（`sa2b`）里最大的三件都不是门身：`trace_validate` **23.70** + `serial_reuse`
**23.16** + `resources` **20.92** = **67.8 µs/walk**，占那条 123 µs 走路的 **55%**。
而 `serial_reuse` 与 `resources` 两段**就是** `trace_validate`：它们各自的函数体只是几个
`Vec`/`BTreeMap` 扫描，µs 来自它们开头的同一行
`ComputeTrace::validate()`。

## 1. ① 为什么会跑三遍

三个区域各自调用一个**公开入口**，而三个公开入口的第一行都是同一条结构校验：

| 区域 | 调用 | 那一行 |
|---|---|---|
| `TraceValidate` | `ComputeTrace::validate()` | —— （走路自己的门） |
| `SerialReuse` | `ComputeTrace::validate_serial_buffer_reuse()` | `self.validate()?;` |
| `Resources` | `ResourceTableSnapshot::validate_trace(trace)` | `trace.validate()?;` |

后两行的存在是对的：它们是公开 API，调用方**可能没有准入过就直接调**，那时结构校验必须
自己跑。但在**一次准入走路内部**，这三遍是同一件事：

1. `validate()` 是 `&ComputeTrace` 的纯函数，走路全程只持有 `&ComputeTrace`，两次调用之间
   没有任何东西能改掉它读到的值；
2. 走路的第一段在 `Err` 上**直接结束**（`?`），所以能走到后两段时，答案已经是 `Ok`；
3. 后两段自己的规则（串行池、资源命名空间）与刀前逐条相同，顺序与按名拒绝都不动。

## 2. ② 这一刀：一次走路只校验一遍

`METAL_API_CORE_VALIDATE_ONCE`（默认关）：

* **关**（未设 / 空串 / `0` / `off` / `false` / `no` / 本刀不认识的任何词）：三段各走各的
  公开入口，`validate()` 跑三遍——**语句序列与刀前逐条相同**（两个公开入口的第一条语句
  仍是 `validate()?`，函数体只是被同文件的私有入口承载）；
* **开**（`1` / `on` / `true` / `yes`，大小写不敏感、两端去空白）：第一段保留
  `trace.validate()`，后两段调用**crate 私有**的两条入口
  （`ComputeTrace::validate_serial_buffer_reuse_after_validate`、
  `ResourceTableSnapshot::validate_trace_after_validate`），它们是同一条函数体**只少了
  那一行校验**。

### 2.1 为什么两条入口必须是 crate 私有

跳过校验只在"同一次走路刚刚对同一个 `&ComputeTrace` 答过 `Ok`"这一个状态下成立。
它是给走路用的**内部续段**，不是给外部用的开关：两条入口是 `pub(crate)`，调用点只有
`ProviderCapabilities::admit_walk` 里的两处，所以 reims rail、`metal-api-vulkan`、
native provider 都拿不到它们，"没校验就跳"这条路在本 crate 之外不存在。

### 2.2 为什么答案不会变

1. `validate()` 的输入是 `&ComputeTrace`，走路持有的是同一个不可变借用，值不会变；
2. 第一段在 `Err` 上结束走路，后两段只可能在 `Ok` 之后被走到——被摘掉的两次调用的答案
   一定是 `Ok`，摘掉它们不可能把"通过"变成"拒绝"，也不可能把拒绝换个名字；
3. 后两段自己的规则、顺序、按名拒绝逐条保留（`SerialReuse` / `Resources` 两条 bar 在开臂
   上塌到门身，但门还在走）。

rail 证据（`crates/metal-api-core/src/provider.rs`
`the_two_arms_of_the_validate_once_walk_answer_the_same`）：同一个进程里用
`admit_validate_once::set_arm` 把同一组 fixture 走两遍，逐条比较
`(fixture, 是否通过, 拒绝 slug)` 序列，并且逐个 fixture 钉住它该被谁按什么名字拒绝
（含"结构坏掉的 trace 两臂都按 `trace_contract_invalid` 拒"与"两段并发 trace 两臂都按
`trace_contract_invalid` 拒"）；另有两条读数把"被摘掉的确实是唯一差别"钉住——
结构坏的 trace 上公开入口答 `Err(EmptyTrace)`、私有入口答 `Ok`。

## 3. ③ 读数（A/B）

同一个 exe 字节流、生产姿态、背靠背 300 s ×3（`av1` 关 / `av1b` 开 / `av1c` 关），
三臂都 `alive`、`BOOT_EXIT=0`、`serial_panic_count=0`（`av1b` 是 attempt 1 静默死亡后
attempt 2 alive；attempt 1 只活了 37 s、那份窗口里没有任何 `ADMIT` 行，下表全部来自
attempt 2 的完整窗口。三臂 launcher 只差一行环境。）

| 读数（`route=validate`） | `av1`（关） | **`av1b`（开）** | `av1c`（关） | 对照均值 → 开臂 |
|---|---|---|---|---|
| `trace_validate` µs/walk | 27.018 | **24.139** | 23.909 | 25.464 → 24.139（−5.2%） |
| `serial_reuse` µs/walk | 27.640 | **2.075** | 23.704 | 25.672 → 2.075（**−91.9%**） |
| `resources` µs/walk | 24.784 | **0.686** | 21.469 | 23.127 → 0.686（**−97.0%**） |
| 三条合计 | 79.442 | **26.901** | 69.082 | 74.262 → 26.901（−63.8%） |
| `draw_passes`（第十刀那一次物化） | 45.014 | 37.430 | 37.341 | 41.178 → 37.430（−9.1%） |
| 四条渲染门合计 | 1.736 | 1.755 | 1.767 | 1.752 → 1.755（+0.2%） |
| **走路 `total`** | **140.809** | **79.153** | **121.619** | **131.214 → 79.153（−39.7%）** |
| `PHASE admit_us` µs/提交 | 134.602 | **73.887** | 115.444 | 125.023 → 73.887（−40.9%） |
| `walks` | 30 464 | 30 464 | 30 464 | 同样本量 |
| `draws_n` / `render_passes_n` | 1.985 / 1.655 | 1.984 / 1.654 | 1.985 / 1.655 | 同形 |
| `draw_list_materialize_n` | 0.146 | 0.146 | 0.146 | 同形 |

**收口**：被摘掉的两遍 = `serial_reuse` −23.597 + `resources` −22.440 = **−46.037 µs/walk**；
旁边三条 bar 的同口径净变化 = −5.069；合计 **−51.106** vs 实测总降落 **−52.061 µs/walk**
⇒ **98.2% 收口**。两条**完全相同**的对照臂自己差 **15.79%**（三条 bar 上 14.99%），
效应是它的 **2.7×**（三条 bar 上 **4.6×**）。三臂携带的工作同形（`walks` 都是 30 464、
`draw_list_materialize_n` 都是 0.146、`refused_n` 953/953/952）⇒ 少掉的是重复的问同一句话。

**R 侧互校**：`prov_admit_validate_us_mean` 10.27 / **5.80** / 8.87 ms/帧（对照均值 9.57 →
−39.4%）；E 侧账 46 µs/walk × 72.2 walks/帧 = 3.32 ms/帧，R 侧实测 3.77，两笔差 1.14×。
**端到端不宣称**：宿主 353.7 / **308.2** / 317.4 ms/帧、帧间隔 659.7 / 659.9 / 657.6
ms/帧——增量臂比两条对照都低，但本刀的宿主账只有 ≈7.3 ms/帧（2.2%），而两条相同对照臂
自己差 11.4%，差值比账大 3.7×。

## 4. 不碰什么

* **不动那三段的规则**：串行池的规则、资源命名空间的一致性规则、结构校验本身一字未改，
  本刀只把"同一次走路里同一件事"的次数从 3 降到 1；
* **不动公开 API**：`validate` / `validate_serial_buffer_reuse` / `validate_trace` 的签名、
  可见性与行为都不变（关臂就是它们本来的语句序列）；
* **不给外部留后门**：两条 `_after_validate` 入口是 `pub(crate)`，外部无法绕过校验；
* **不动 `submit` 那条二次走路**：`validate_trace`（调用方）与 owner 复核（`submit`）是
  两条**独立**走路的两遍（每个提交 ≈ 2 × 258 µs），要不要合并是另一件事，需要
  `ValidatedComputeTrace` 记住"被哪个快照准入"的新契约（见任务报告 §6.1 的设计段）。
