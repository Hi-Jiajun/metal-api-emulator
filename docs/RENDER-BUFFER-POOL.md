# 逐 pass 上传 buffer 的复用（E 侧第四刀）

本文件记录一次 provider 增量：**一条 offscreen pass 上传进 GPU 的那几个 host-visible
buffer 不再每 pass 建一次、销一次，而是按「大小 + 用途」从池里取上一次同形状创建留下的
buffer 与 memory**。它接在 `docs/SUBMIT-PHASE-PROFILE.md` 的相位表上，是「拆 pass
teardown」那一轮（sp10）读数直接选出来的刀口。

## 0. 一轮读数：teardown 的十个组谁最大

第三刀（`docs/READBACK-MEMORY.md`）之后，sp8 相位表里最大的未命名条是
`render_teardown`（494.2 µs/submit，18.3%）——pass 结束时那一段显式 drop。sp10 轮
（reims `65db73f` × provider `41f7e7b`，300 s 驻留、187 个窗口 / 47 872 次提交）把它拆
成 `OffscreenObjects::drop` 自己走的十个组：

| 组 | µs/submit | 占 teardown | 对象 |
|---|---|---|---|
| **`teardown_buffers`** | **398.4** | **78.5%** | 上一次传、命令池之外的所有 buffer：stage buffer、vertex stream、index / indirect buffer |
| `teardown_readbacks` | 43.3 | 8.5% | 每个 stored attachment 的 readback 目的地（unmap + buffer + memory） |
| `teardown_attachments` | 27.8 | 5.5% | 颜色附件的 image / view / memory（含 resolve 目标） |
| `teardown_previous` | 21.3 | 4.2% | `Load` 的 previous-byte staging buffer |
| `teardown_sync` | 12.7 | 2.5% | 完成 fence 与 command pool |
| `teardown_descriptors` | 1.5 | 0.3% | descriptor pool / set layout / 空 layout |
| `teardown_passes` | 0.6 | 0.1% | render pass、seed render pass、两个 framebuffer |
| `teardown_textures` | 0.5 | 0.1% | 采样声明自己的 sampler / view / image / memory |
| `teardown_depth_stencil` / `teardown_pipeline` | 0.1 | 0.0% | 本轮为零（无深度、pipeline 已被 shape cache 收走） |
| 缝 | 1.4 | 0.3% | 十组之间的函数调用边界 |
| **`render_teardown`** | **507.7** | 100% | 整条（total 的 18.1%） |

同时新增的五个计数给出每个组的**人口**（`td_buffer_n` 5.147/submit、
`td_memory_n` 5.314/submit、`td_image_n` / `td_view_n` 0.140/submit、
`td_sampler_n` 0.626/submit）：teardown 里 78% 的时间花在每提交 **5.1 个 buffer + 5.3 个
memory** 的销毁上，折算**每个 buffer ≈ 80 µs**。这十条里没有 `vkDeviceWaitIdle` 一类同步
（fence 早在那之前等过了），所以「同步」这一组就是 fence / command pool 的销毁本身。

拆条还给出对照读数：teardown 之前那段「还回去」是 **免费的**——`render_release_reuse`
4.0、`render_release_pool` 0.4、`render_release_import` 0.1、`render_retire` 0.1
µs/submit，四条合计不到 teardown 的 1%。同一批对象的**创建**侧在 `render_setup` 里：
`setup_stage_buffers` 181.1 + `setup_inputs` 130.6 + `setup_readbacks` 26.9
µs/submit——也就是说这 5 个 buffer 每提交被建一次（≈340 µs）又被销一次（≈460 µs）。

## 1. 缺口

`OffscreenObjects::create_host_visible_buffer` 是这些 buffer 的共同出口（第四刀把
indirect 命令那条内联的重复实现也并了进来，所以现在只有一条创建路径）：

| 调用 | 作用 |
|---|---|
| `vkCreateBuffer(size, usage, EXCLUSIVE)` | 一条 host-visible 的 device buffer |
| `vkGetBufferMemoryRequirements` | 拿 memory 要求 |
| `vkAllocateMemory` | 真正的代价所在（第一刀/第二刀的读数已经证明驱动的分配是毫秒级） |
| `vkBindBufferMemory` | 绑上 |
| `vkMapMemory` / 拷贝声明自己的字节 / `vkUnmapMemory` | 把这条声明的 bytes 写进去 |

guest 桌面每帧绑定的是同一批形状（同一个 quad 的 vertex stream、同一个索引表、同一组
stage buffer），而 provider 每一 pass 都按新 buffer 重建一次。buffer 的**身份**就是交给
驱动的两个字段：`VkBufferCreateInfo` 的 `size` 与 `usage`——它们相同，驱动被交付的就是
同一条创建。

## 2. 规则与证明

### 2.1 键就是交给驱动的字段

`UploadKey` 逐字段取 `byte_length`（`VkBufferCreateInfo::size`，即调用方的 `byte_length`）
与 `usage`，比较是这两个字段的相等——没有摘要可碰撞，因为人口是十来个形状的线性扫描，比
它省下的四次驱动调用便宜得多。

### 2.2 命中省掉的是分配，不是写

命中时跳过的正是 `vkCreateBuffer`、`vkGetBufferMemoryRequirements`、`vkAllocateMemory`
与 `vkBindBufferMemory`；`vkMapMemory`／拷贝这条声明的 bytes／`vkUnmapMemory` 照旧执行
（命中路径用 `VK_WHOLE_SIZE` 映射整块分配，覆盖原来 `requirements.size` 覆盖的同一段）。
因此 pass 绑定的每一个字节都是这条声明自己的：这是**分配**的复用，不是内容的复用。写方向
的相干性没有变——池里的 pair 与新建的 pair 是同一个 `HOST_VISIBLE | HOST_COHERENT`
选型的 allocation。

### 2.3 什么时候还回去

只有走完 readback 的成功 pass 才还（在 fence 之后、和另外三个 hand-back 并排）；那时
`stage_buffer_readback_bytes` 已经把可写 stage buffer 的字节拷进自有 `Vec`，所以不会再有人
读这块 memory。还回去时把字段里的两个 handle 置空，pass 自己的 teardown 就无事可做。

### 2.4 失败方向

池里的 pair 在 `take` 时就离开了池：创建失败、map 失败、pass 在 fence 前失败、或机制在
中途被关掉，都走原来的销毁路径（fail-closed）。上限 64 个形状 / 192 MiB（按 key 自己的
`size` 记账，是分配大小的下界），超限按最久未用逐出；被逐出或在开关关闭时被清空的 pair
在移除它的那把锁里销毁。

### 2.5 这一刀不碰什么

readback 目的地（`create_readback` 那条 `readback_memory` 选型的路径）、颜色附件、深度／
模板面、command pool、render pass / framebuffer、descriptor 状态、采样纹理自己的
sampler / view / image 都保持原样——它们各自只占 teardown 的 0.1%–8.5%，而这张读数的
第一支配项是 buffer 那一组。owner 窗口的 import 仍走 `crate::render_import_pool`，
构造上不带 upload key。

## 3. 开关与读数

`METAL_API_VULKAN_RENDER_BUFFER_POOL=0`（也接受 `off` / `no` / `false` 及其大写）关掉整个
机制：不取、不留，每次创建只有一次 relaxed load，这就是对照臂。相位行新增
`buffer_hit_n` / `buffer_miss_n` / `buffer_disabled_n` / `buffer_return_n` /
`buffer_drop_n` 五个计数，把「没人问」与「问了但拒绝」分开；进程两侧的累计读数在
`RenderBufferPoolCounts`。

## 4. A/B 读数（同一个 exe 字节）

三臂同一个 exe 字节（sha256 `d3542d86…`，`evidence/gate3-census-sp11-2026-09-20/01-ab-exe-identity.txt`），
launcher 只差 `METAL_API_VULKAN_RENDER_BUFFER_POOL=0` 一行；坐标 reims `65db73f` ×
provider `13a59ee`，300 s × 3 背靠背（每臂之间 settle）：

| µs/submit | sp11（开） | sp12（关） | sp11b（开） | 关→开（均值） |
|---|---|---|---|---|
| **`total`** | **2 307.1** | **2 694.2** | **2 325.0** | **−378.1（−14.0%）** |
| `render_total` | 1 099.1 | 1 567.9 | 1 126.9 | −454.9 |
| ├ `render_setup` | 246.5 | 438.0 | 256.7 | −186.4 |
| │ ├ `setup_stage_buffers` | 27.8 | 183.2 | 22.3 | −158.1 |
| │ └ `setup_inputs` | 91.9 | 136.7 | 98.5 | −41.6 |
| ├ **`render_teardown`** | **117.4** | **513.1** | **132.9** | **−387.9** |
| │ ├ **`teardown_buffers`** | **26.5** | **399.8** | **28.5** | **−372.4** |
| │ └ `teardown_previous` | 0.0 | 21.9 | 0.0 | −21.8 |
| └ `render_release_uploads`（这一刀自己的还回） | 103.5 | 0.2 | 109.8 | **+106.4** |
| `render_readback`（未被碰的对照条） | 76.2 | 65.5 | 70.3 | +7.7 |
| `render_wait`（未被碰） | 201.6 | 217.5 | 212.6 | −10.4 |
| `resource_build` / `pool` / `plan` / `writebacks`（纯 CPU，与本刀无交集） | 267.5 / 163.4 / 120.4 / 122.8 | 257.5 / 126.7 / 106.3 / 103.8 | 298.5 / 129.9 / 113.2 / 110.5 | +25.5 / +19.9 / +10.6 / +12.9 |
| 机制计数 `buffer_hit_n` / `buffer_miss_n` / `buffer_disabled_n` / `buffer_return_n` | **3.386 / 1.002 / 0 / 4.388** | 0 / 0 / **4.389** / 0 | **3.399 / 0.988 / 0 / 4.387** | — |
| 人口 `td_buffer_n` / `td_memory_n` | **0.785 / 0.924** | 5.178 / 5.327 | 0.789 / 0.942 | — |

**R 侧独立读数（`frame_span`，与相位仪无关）**：`prov_submit_us_mean` 按每 draw 折算
**2 354 / 2 738 / 2 371 µs**，两侧量的是同一次调用差（E −378.1，R −384 / −367），
差 2% 以内。

**这一刀砍掉什么、还剩什么**

* **砍掉**：每提交 4.4 次创建里的 3.4 次（`buffer_hit_n`）与销毁（`teardown_buffers`
  −372.4、`teardown_previous` −21.8），创建侧对应 `setup_stage_buffers` −158.1 与
  `setup_inputs` −41.6。真实销毁的 buffer 从 5.18/submit 降到 0.79/submit。
* **没砍**：readback 目的地（`teardown_readbacks` 44.5，**未被碰**）、颜色附件
  （`teardown_attachments` 29.5，未被碰）、fence / command pool（13.0，未被碰）、
  stage buffer 的字节上传本身。
* **本刀自己的开销**：`render_release_uploads` 103.5 µs/submit（4.39 次还回 ⇒
  每次 ≈ 23 µs）。这条 bar 在对照臂是 0.2 µs，所以它全部是这一刀引入的工作；读数给出的
  两个候选解释是「池上限在逐出」或「还回本身的簿记」，需要下一轮用
  `entries` / `held_bytes` / `evictions` 三个池状态读数判定（本轮的相位行没有打印它们，
  所以这里只报数、不下结论）。
* **公平性**：每提交的 canonical draw 三臂都是 1.003 / 1.005 / 1.004，`render_offscreen_n`
  三臂都是 1.000；按提交归一的活一样多。覆盖面无回归（`06-coverage-ab.txt`：三臂
  `out_of_class_shape_lines distinct=0`、`render_seam 0/0/0`、
  `compute_provider_out_of_class=47`，`pass_color_slots_1 == render_provider_canonical`）。
