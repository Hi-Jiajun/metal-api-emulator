# draw list 的 per-draw 对象回池：成员不再"建一次、毁一次"

E 侧第十二刀（`draw_object_release`，开关 `METAL_API_VULKAN_DRAW_OBJECT_RELEASE`，
**默认关**）。机制与证明见模块文档 `crates/metal-api-vulkan/src/draw_object_release.rs`；
本文记录**为什么是这一处**、**两臂读数**与**不碰什么**。

## 0. 缺口：回池那一步没走

析构轮（tag `ct2`，E `ecfb012` × R `9ab840c6`，300 s 生产姿态、28 152 次提交 / 381 帧）
先把"析构族"拆到**对象级**，再问"这些对象为什么没回到池里"：

| 读数 | 值 | 说的是 |
|---|---|---|
| `td_memory_us` | 358.4 µs/提交（26.48 ms/帧） | `render_teardown_us` 29.11 ms/帧 的 **91%** |
| 每个 `vkFreeMemory` | **149 345 ns** | 对比 `vkDestroyBuffer` 760 ns、view 924 ns、image 1 319 ns |
| 上传池 `buf_miss_n` | 30 391 | 一次创建池没接住 |
| **`td_unreturned_n`** | **30 329** | 同一个对象**带着池键**被析构 |
| 这些 miss 时池里有多少条目 | 平均 **52.6**（`held_sum ÷ miss_n`） | 池是有货的 |
| 撞到空池 / 跟随淘汰 | **3 / 0** | 既不是冷池、也不是预算淘汰 |
| 逐提交切分 | `batch_n>0` 的行 0 个；`batch_n==0` 的行 **30 329** 个 | 与"批次"互斥 = draw list 自己的那一类提交 |

按 kind 是 `stage buffer` 13 955 / `vertex input` 8 050 / `index input` 8 052
（`attachment previous bytes` 0，因为那条路径本来就还）。读码落点：
`execute_offscreen_render_draws` 给每个成员建一套自己的 `OffscreenObjects`
（`build_draw_objects`），而 `finish_prepared_offscreen_pass` **直接 drop 它们**——
pass 自己那套有四个 hand-back（reuse / textures / imports / uploads），成员一套都没有。

## 1. 本刀的形状

开关开时，成员在 **fence 之后、drop 之前**，按 pass 自己那套**逐字同序**的四个调用交回：

| 家族 | 交回给谁 |
|---|---|
| pipeline / layout / 两个 module | 形状缓存 `crate::render_setup_reuse` |
| 采样后备 | `crate::render_texture_pool` |
| owner-window import | `crate::render_import_pool` |
| 上传对（buffer + memory） | `crate::render_buffer_pool` |

整段记进新 bar `render_release_draws_us`（渲染半边残差的第 14 个成员）。
开关关时这段不进，成员照旧直接 drop ⇒ **逐字节同形**。

## 2. 为什么驱动看不出区别

1. **在 fence 之后。** 这段就坐在 pass 自己那四个 hand-back 的同一个位置：
   `vkWaitForFences` 已经让提交退休，成员的对象不再被任何命令缓冲读。
2. **回读不受影响。** pass 的回读（`read_back_offscreen`）读的是 **pass 自己**的对象与
   决策；成员那套不向它贡献任何附件图像或回写目的地。
3. **字节仍是自己的。** 复用的是一对**分配**，不是内容：每条 draw 照样把自己的
   顶点/索引/舞台字节写进它拿到的那对宿主可见内存（`crate::render_buffer_pool` 的规矩）。
4. **别的都没动。** 成员照样建、照样录、照样提交；两臂的 141 份 capture 逐字节相同（§3）。

## 3. 开关与两臂读数

`METAL_API_VULKAN_DRAW_OBJECT_RELEASE=1`（也接受 `on`/`true`/`yes`）打开；**默认关**，
其余取值（含未设）就是刀前的路径，也就是两臂里的对照臂。

两臂（`ct2a` = off / `ct2b` = on，其余逐行同形、同一把锁背靠背、`DWELL=300`、
`CENSUS_SCOPE_DEADLINE=420`、同一 tip 的两套启动器只差这一行）：

| 读数 | off | on | 差 |
|---|---|---|---|
| **`td_unreturned_n`（全轮）** | **34 400** | **0** | **−100.0%** |
| `buf_miss_n`（全轮） | 34 480 | 145 | −99.6% |
| `buf_miss_stage_n` / `_vertex_n` / `_input_index_n` | 15 931 / 9 251 / 9 255 | 81 / 21 / 16 | −99.5% / −99.8% / −99.8% |
| `buf_return_stage_n`（窗表） | 142 350 | 151 943 | +6.7% |
| `td_memory_n` / `td_buffer_n`（全轮） | 76 170 / 63 579 | 34 815 / 27 975 | −54.3% / −56.0% |
| `pool_miss_n`（纹理后备池） | 5 381 | 117 | −97.8% |
| `render_teardown_us`（每提交） | 425.4 | 246.3 | −179.1 µs（**−42.1%**） |
| `render_setup_us`（每提交） | 565.3 | 424.4 | −140.9 µs（**−24.9%**） |
| `td_memory_us`（每提交） | 385.0 | 209.1 | −175.9 µs（−45.7%） |
| `teardown_buffers_us`（每提交） | 137.2 | 0.6 | −99.6% |
| `teardown_textures_us`（每提交） | 42.5 | 1.4 | −96.7% |
| `render_total_us`（每提交） | 2 031.3 | 1 664.2 | −367.1 µs（−18.1%） |
| `total_us`（每提交） | 3 170.8 | 2 794.2 | −376.6 µs（−11.9%） |
| 尾部（top 5%）`render_teardown_us` | 2 171.8 | 410.2 | −81.1% |
| 尾部（top 5%）`render_setup_us` | 3 919.8 | 2 693.9 | −31.3% |
| 尾部（top 5%）`total_us` | 20 017.5 | 16 658.1 | −16.8% |
| **`render_release_draws_us`（窗表均值）** | 0.000（从未进过） | **167.6 µs/窗** | 交回本身约 **0.66 µs/提交** |
| （对照）`fence_wait_us` | 125.8 | 129.6 | +3.0% |
| （对照）`plan_us` | 413.3 | 409.1 | −1.0% |

读法：**这一刀同时打在两侧**——成员不再建（`render_setup_us` −24.9%，少一次
`vkCreateBuffer`/`vkAllocateMemory`/`vkBindBufferMemory`），也不再毁
（`render_teardown_us` −42.1%，其中 `teardown_buffers_us` 几乎归零）；
交回本身的成本是 **0.66 µs/提交**（`render_release_draws_us` ÷ 窗内提交数），
对照它省下的 ~320 µs/提交，比约 **480×**。
两条对照 lane（`fence_wait`、`plan`）只动 1.0–3.0%，而两臂的宿主每帧几乎相同
（502.5 vs 498.4 ms，0.8%）⇒ 省下来的时间落在被碰的那两条上，不是被别的 lane 吸收。
折算到宿主帧：`total_us` 由 238.3 ms/帧 降到 212.8 ms/帧（**−25.5 ms/帧**）。

控制面（两臂逐项）：覆盖率恒等式**都闭合**（`pass_color_slots_1 − fell_back =
canonical + engine`）、五个零读**都 0**、`class_exit_buckets` 键集 **diff 空**、
红线 `draws_skipped_after_engine_refusal` **两臂都不出现（= 0）**。

## 4. 这一刀不碰什么

* **不碰键、不碰预算。** 池的键、上限、淘汰策略一字未改；这一刀只让"本来就该还的"
  回到池里（`after_evict=0`、`same_length=0` 说明加宽键或加预算解决不了这个问题）。
* **不碰 pass 自己那套的时序。** 四个 hand-back 的位置、顺序、bar 都不变，
  成员那套是同一顺序的**追加**。
* **不碰回写与像素。** 成员自己的 writable stage buffer 本来就不回写（B-2 的既有边界，
  见报告 §7.2）；两臂的 141 份 capture 逐字节相同，R rail 两臂 203/0。
* **不碰纹理池的形状归因。** `pool_miss` 里"同 carrier、同 format、**不同 extent**"
  的那一族（1 184/5 342）另开一刀：那要动键与载体，不是 hand-back。

## 5. 翻档

这一刀以**关**为默认落地。翻档与否由主代理按读数与门决定：`td_unreturned_n`
在开臂上**精确为 0**、覆盖率与红线两臂同形，是"机制计数类、可翻"的形状；
但翻档前必须复核主代理自己的两臂读数（本文件只记录 track 侧的实测）。
