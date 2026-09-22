# 准入走查：一条多画列表的单画 pass 只建一遍

E 侧第十二刀。开关 `METAL_API_CORE_ADMIT_DRAWS_ONCE`（**已翻默认：未设即开**，四个控制词
`0`/`off`/`false`/`no` 回到刀前路径），机制与证明写在
`crates/metal-api-core/src/admit_draws_once.rs` 的模块文档里；本文记录**为什么是这一处**、
形状读数、两臂读数与**不碰什么**。

## 0. 缺口：`admit_capabilities` 的 85% 是同一批单画 pass 被建了两遍

CT1（`docs/SUBMIT-POOL-ONCE.md`）把 `admit_capabilities` 标成"同一张声明表的第三遍走查"
（5.85 ms/帧、94.9% 落尾）。E-C2 的形状轮（tag `aw1`，E `a46ffb4` × R `9ab840c6`，
生产姿态、`DWELL=300`、451 帧、69.44 提交/帧、宿主 460.7 ms/帧）把这条走查拆到 15 个
命名区域，读数与 CT1 的猜想**不同**：它**不是** plan/validate 那份池派生的另一遍，
而是**它自己内部**的一次重复构造。

| 区域（route=submit，31 317 次走查） | µs/走查 | 占均值 | p99 |
|---|---|---|---|
| **`draw_passes`**（走查自己物化 render 条目） | **59.975** | **54.4%** | 1 286.5 |
| **`trace_validate`**（trace 结构校验 + 列表自校验建的单画 pass） | **31.689** | **28.7%** | 1 337.2 |
| `serial_reuse` | 1.323 | 1.2% | 6.6 |
| 四个 render 门（`render_passes`/`render_textures`/`pixel_samplers`/`stage_buffers`） | 0.938 | 0.9% | 4.4 |
| `resources` | 0.319 | 0.3% | 1.7 |
| 其余七段 | 0.319 | 0.3% | ≤1.2 |
| **合计** | **110.265**（p50 3.3 / p90 14.7 / p99 2 746.1） | 100% | |

两件事把上面那张表钉死：

* **成本只出现在有多画列表的走查上**。走查里物化过列表的有 4 944 次（15.8%），
  它们的均值是 **673.4 µs**（`draw_passes` 378.9 + `trace_validate` 193.3）；没有列表的
  26 373 次只有 **4.7 µs**。
* **尾部把全部成本带走**。top 5% 的 1 565 次走查占这条 lane 的 **95.0%**，其中
  `draw_passes` 占尾部的 56.6%（该段总量的 98.8% 落在尾部）、`trace_validate` 占 28.7%。

折算到宿主帧：这条走查 **7.66 ms/帧**，其中 `draw_passes` 4.16、`trace_validate` 2.20。
（这是 base tip 那一轮的绝对值；下面两臂的 off 臂读 44.056 / 23.184 µs/走查，低的那部分
是那一轮宿主更轻——366.3 vs 460.7 ms/帧，本刀只报**臂内差**。）

## 1. 两次构造是哪两次

一条 `TracePass::RenderDraws` 的规则是"N 个 draw 就是 N 个单画 pass"
（`RenderDrawsDescriptor::materialize`）。一次准入走路里，这批单画 pass 被建**两遍**：

| # | 谁 | 建了什么 | 建完 |
|---|---|---|---|
| ① | `RenderDrawsDescriptor::validate`（在 `trace_validate` 区域内） | 每个尾画一份 `head.with_draw(draw)`，逐条过单画 pass 的规则 | **校验完就丢**，只留判词 |
| ② | 走查的物化（`draw_passes` 区域内，`ComputeTrace::render_draw_passes`） | 同一批 `head.with_draw(draw)`，外加头画的一份 `head.clone()` | 交给四个 render 门读 |

（第十刀 `docs/SUBMIT-ADMIT-PROFILE.md` 已把 ② 从"每个门各建一遍"收成"每次走路一遍"，
① 从那以后就是剩下的一份多余构造。）

形状轮的读数正是这两段的比值：列表走查里 ② = 378.9 µs、① = 193.3 µs，二者比值
`N : (N−1)`（该轮每个列表平均 2.82 个 draw：`draw_list_materialize_passes 0.446/walk ÷
draw_list_materialize_n 0.158/walk`），单画 pass 的建价 ≈ **110–135 µs/份**
（列表头带 ~0.5 MB 的纹理声明，`texture_owned_bytes_n 489 339/walk`）。

## 2. 这一刀：列表自校验建的那批单画 pass 交给走查

`METAL_API_CORE_ADMIT_DRAWS_ONCE`（翻默认后：未设即开）：

* **开**（未设 / 空串 / 本刀不认识的任何词）：走查改调
  `ComputeTrace::validate_collecting_draws()`，它是同一条校验体、**把列表自校验建出的
  单画 pass 留下来**：头画按借用交回（走查全程持有 `&ComputeTrace`，头画不必克隆），
  每个尾画的 pass 就是自校验本来就要建的那份。四个 render 门读到的就是这批值，
  走查自己不再物化，`draw_passes` 段因此读到 0——**构造搬了位置，没有消失**。
* **关**（`0` / `off` / `false` / `no`，大小写不敏感、两端去空白）：走查先
  `trace.validate()`（列表自校验照旧建一遍、校验完即丢），再自己物化一遍 render 条目。
  **语句序列与刀前逐条相同**，是两臂轮与合流后复跑用的对照臂。

翻默认的依据是两臂轮 `aw2a`（关）/`aw2b`（开）：同一 tip、同一 exe 身份、背靠背、各 300 s，
单画 pass 构造 **0.7137 → 0.2762/走查（−61.3 %）**、`draw_passes` 段
**44.056 → 0.000 µs/走查**、走查自身 **81.648 → 29.960 µs（−63.3 %）**、
`admit_capabilities_us` **95.550 → 49.055（−48.7 %）**，而 R 侧独立计时的
`prov_admit_validate_us_mean` **8.860 → 5.043 ms/帧（−43.1 %）**；两臂控制面同形
（恒等式闭合、五个零读 0、桶键集 diff 空、红线 0），141 份 capture 逐字节相同。

开关只在第十刀的那一份物化还在时才有意义（关掉第十刀，四个门各建各的，本刀没有可交接
的对象），所以臂的条件是 `shared_draws && admit_draws_once`。

### 2.1 为什么答案不会变

1. `validate_collecting_draws` 与 `validate` 共用同一条函数体（`validate_into`），
   sink 只决定"建出来的 pass 交给谁"：天花板检查、头画校验、逐尾画
   `with_draw` + `validate` 的**顺序、调用点、按名拒绝**逐条不变，`?` 的落点也不变；
2. 交回的值就是 `render_draw_passes()` 会建的那些值：单画条目借用、列表头借用、尾画
   `head.with_draw(draw)`，顺序与索引逐条相同（`the_draws_a_validation_hands_over_are_
   the_ones_the_list_materializes` 钉住值，`the_two_arms_of_the_draws_once_walk_answer_
   the_same` 钉住四个名字）；
3. 四个门只**读**手里的 `&RenderPassDescriptor`，trace 的不可变借用覆盖整个走查，
   没有任何东西能在交回之后改掉这些值。

### 2.2 机制读数

`metal_api_core::admit_profile` 的三个事件把两边数清楚（每次走查，route=submit）：

```text
draw_list_materialize_n        —— 走查自己物化列表的次数（关臂 >0，开臂 0）
draw_list_materialize_passes   —— 走查交给门的单画 pass 份数（关臂 ΣN，开臂 0）
draw_list_validate_build_n     —— 列表自校验建出的单画 pass 份数（两臂都 Σ(N−1)）
```

**一支列表的单画 pass 构造数** = `materialize_passes + validate_build_n`：
关臂 `N + (N−1) = 2N−1`，开臂 `(N−1)`。这是本刀的机制条。

## 3. 读数（两臂，`aw2a` off / `aw2b` on，同 tip `75426dd`、300 s、背靠背）

姿态：`aw2a` 31 821 提交 / 438 帧 / 宿主 366.3 ms 每帧；`aw2b` 29 882 / 434 / 376.1
（on 臂宿主更重 2.7%、提交更少 6.1%，方向都对 on 臂不利）。两臂都是 `verdict=alive`。

| 读数（每次走查，route=submit = `admit_capabilities`） | off | on | 差 |
|---|---|---|---|
| `draw_list_materialize_n` | 0.1539 | **0.0000** | −100% |
| `draw_list_materialize_passes` | 0.4338 | **0.0000** | −100% |
| `draw_list_validate_build_n`（新计数） | 0.2799 | 0.2762 | −1.3%（同一件工作） |
| **单画 pass 构造总数** | **0.7137** | **0.2762** | **−61.3%（2.58×）** |
| 每列表构造数 | **4.637 = 2N−1** | **1.685 ≈ N−1** | — |
| `draw_passes_us`（区域） | 44.056 | **0.000** | −100% |
| `trace_validate_us`（区域） | 23.184 | 21.934 | −5.4%（同量构造 + 姿态噪声） |
| 15 个命名区域合计 | 69.404 | 23.894 | −65.6% |
| 走查 total（含仪器 census） | 81.648 | 29.960 | **−63.3%** |
| 调用方那条走路（route=validate） | 90.652 | 40.009 | −55.9% |
| 相位条 `admit_capabilities_us` mean / p99 | 95.550 / 2 459.5 | 49.055 / 1 289.6 | −48.7% / −47.6% |
| （对照）整次提交 `total_us` | 2 055.2 | 2 194.2 | +6.8%（姿态） |
| R 侧 `prov_admit_validate_us_mean`（独立尺） | 8.860 ms/帧 | 5.043 ms/帧 | **−43.1%** |

尾部（top 5%）也跟着换形：off 的尾部里 `draw_passes` 占 56.0%（该段 99.0% 落尾），
on 的尾部里它读到 0，剩下的是 `trace_validate`（78.0%）——也就是"门仍然只吃物化好的
单画 pass"那一份，下一刀的对象。

关时等价性：冒烟两臂 **141 份 capture 逐字节相同**，R rail 两臂 **203/0**，
两臂控制面同形（覆盖率恒等式闭合、五零读 0、桶键集 diff 空、红线 0）。

## 4. 不碰什么

* **不碰判据**：天花板、`UnusedPipeline`、逐画的结构规则、四个门的按名拒绝一字未改；
* **不碰 plan/validate**：`serial_resources_ref`（CT1 的刀）与 `submit_pool_once` 不动，
  这一刀与它读的表无关（见 `e_admit_walk_reuse-report.md` §3 的判词）；
* **不碰值**：交回的是同一批值（上表末两行的等价性就是它的读数）；
* **不翻默认**：这一刀以关为默认落地；翻档由主代理在读数与门都复核之后决定。
