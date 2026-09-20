# compute 半边逐提交上传 buffer 的复用（E 侧第六刀）

本文件记录一次 provider 增量：**compute 半边（`VulkanComputeProvider` 的提交路径）每次提交建
一次、销一次的 host-visible buffer 不再每次重建，而是按「大小 + 用途」从池里取上一次同形状
创建留下的 buffer 与 memory**。它接在 `docs/SUBMIT-PHASE-PROFILE.md` 的相位表上，是 G3-A
（残差拆条）的读数直接选出来的刀口，成法照渲染半边第四刀的 `docs/RENDER-BUFFER-POOL.md`。

## 0. 一轮读数：g3a 表里最大的两条 compute 项

G3-A 轮（reims `04f5d2e` × provider `883b60a`，300 s 驻留、166 个窗口 / 42 496 次提交）把
一次提交拆开后，compute 半边最大的两条与「每提交建/毁一个对象」直接相关：

| 条 | µs/提交 | 占整笔 | 对象数/提交 | µs/对象 |
|---|---|---|---|---|
| `rb_buffer`（`resource_build` 的 buffer 族） | **195.4** | 7.2 % | 1.020 | **191.7** |
| `submit_td_buffers`（`submit_teardown` 的 buffer 族） | **155.9** | 5.7 % | — | — |
| 两条合计 | **≈351** | **12.9 %** | 1.020 建 / 1.020 毁 | — |

`rb_buffer_n` = 43 329 与 `rb_memory_n` = 43 329 同值，即**每创建一个 buffer 就配一次
`vkAllocateMemory`**；同轮 `submit_td_buffers` 的人口（85 825 条 = 每提交一次 per-buffer
region 加一次间接 replay 的 region）说明**每个提交恰好建一个、毁一个**。渲染半边在第四刀之前
读到的形状与此完全一致（`teardown_buffers` 5.1 个 buffer 加 5.3 个 memory / 提交，每个
≈80 µs），而那半边已经用 `crate::render_buffer_pool` 收走了同一形状的对象。

## 1. 缺口

compute 半边每次提交建的那一个 pair 出自 `ExecutionResources::create_owned_backing`——本提交
自己的 staged 字节、borrowed-and-copied 的字节、以及 guest-runs 收集出来的窗口，都是从这里
变成一条 device buffer 的：

| 调用 | 作用 |
|---|---|
| `vkCreateBuffer(size, STORAGE_BUFFER, EXCLUSIVE)` | 一条 host-visible 的 device buffer |
| `vkGetBufferMemoryRequirements` | 拿 memory 要求 |
| `vkAllocateMemory` | 真正的代价所在（g3a 读数：191.7 µs/个） |
| `vkBindBufferMemory` | 绑上 |
| `vkMapMemory` / 拷贝这条声明自己的字节 | 把这条提交的 bytes 写进去 |

guest 桌面每帧提交的是同一批形状（同一条 stage buffer 的同一段 view、同一个 shared 分配、同
一条间接命令），而 provider 每一提交都按新 buffer 重建一次。buffer 的**身份**就是交给驱动的
两个字段：`VkBufferCreateInfo` 的 `size` 与 `usage`——它们相同，驱动被交付的就是同一条创建。

## 2. 规则与证明

### 2.1 键就是交给驱动的字段

`ComputeBufferKey` 逐字段取 `byte_length`（`VkBufferCreateInfo::size`）与 `usage`，比较是这
两个字段的相等——没有摘要可碰撞，因为人口是十来个形状的线性扫描，比它省下的四次驱动调用便宜
得多。这条准则与渲染半边第四刀的 `UploadKey` 是同一条：**键是从即将交给驱动的那个结构里读回
来的**，不是另立一份描述。

### 2.2 命中省掉的是分配，不是写

命中时跳过的正是 `vkCreateBuffer`、`vkGetBufferMemoryRequirements`、`vkAllocateMemory` 与
`vkBindBufferMemory`；`vkMapMemory`、拷贝这条提交的 bytes、（间接命令的）`vkUnmapMemory` 照
旧执行（命中路径用 `VK_WHOLE_SIZE` 映射整块分配，覆盖原来 `requirements.size` 覆盖的同一段）。
因此提交绑定的每一个字节都是这条声明自己的：这是**分配**的复用，不是内容的复用。写方向的相干
性没有变——池里的 pair 与新建的 pair 是同一个 `HOST_VISIBLE | HOST_COHERENT` 选型的
allocation。

### 2.3 什么时候还回去

只有**观察到自己的 fence** 的提交才还：`ExecutionResources::drop` 走到销毁路径时，
`completed && !device_lost` 为真才把 pair 交给池，否则照旧销毁。此时 readback（`read_updates`）
已经跑完，可写 view 的字节已经拷进自有 `Vec`，所以不会再有人读这块 memory。还回去之前先
`vkUnmapMemory`：新建路径会把映射留到提交结束（`vkFreeMemory` 隐式解除），而 Vulkan 不允许对
同一块 memory 重复 map，所以**池里驻留的状态是「未映射」**，下一个取用者按新建路径的同一顺序
再 map 一次。

### 2.4 失败方向

池里的 pair 在 `take` 时就离开了池：创建失败、map 失败、提交在 fence 之前失败、设备丢失、或
机制在中途被关掉，都走原来的销毁路径（fail-closed）。上限 64 个形状 / 64 MiB（按 key 自己的
`size` 记账，是分配大小的下界；compute 半边的上传都是小对象——一段 stage view、一条 12 字节
的间接命令——这个上限比渲染那半边的 192 MiB 低一个量级），超限按最久未用逐出；被逐出或在开关
关闭时被清空的 pair 在移除它的那把锁里销毁。设备丢失那一臂尤其不还：损坏的设备上取回的 pair
没有意义，也不该再喂给下一条命令缓冲。

### 2.5 这一刀不碰什么

四个创建点保持自己的路径，每一个都因为「形状不决定身份」：

* **owner 窗口的 import**（`import_host_buffer`）：它的身份是 owner 的指针，不是形状；渲染
  半边已有自己的 `crate::render_import_pool`，把一条 import 的 pair 交给一个声明了别的窗口的
  创建是错的。
* **heap 的 placement**（`create_heap_buffers`）：它的 memory 是本提交唯一的那块 slab，绑定
  偏移来自 heap plan，而 slab 的 memory type 是**所有 placement 要求的交集**——身份是
  `(size, usage)` 再加上「在这样一块 memory 里的这个偏移」，池化它要池化这层关联，那是另一个
  机制。（g3a 读数里这条路径为 0：`rb_buffer_n` 1.020/提交 就说明 heap plan 在 guest 轮里没有
  落地。）
* **storage image 的 transfer buffer**（`create_storage_texture`）：它的生命周期是那张
  storage image 的，和 image / view / 校验同一个 region 一起建、一起毁，不属于本提交的 owned
  pair。
* **开关关闭时创建的每一个 buffer**。

readback 目的地、颜色附件、command pool、descriptor 状态、采样/存储声明自己的
image / view / memory 也都保持原样：g3a 读到的第一支配项是「每提交一个 owned pair」，不是它们。

## 3. 开关与读数

`METAL_API_VULKAN_COMPUTE_BUFFER_POOL=0`（也接受 `off` / `no` / `false` 及其大写）关掉整个机
制：不取、不留，每次创建只有一次 relaxed load，这就是对照臂。相位行新增
`compute_buffer_hit_n` / `compute_buffer_miss_n` / `compute_buffer_disabled_n` /
`compute_buffer_return_n` / `compute_buffer_drop_n` 五个计数，把「没人问」与「问了但拒绝」分
开；进程两侧的累计读数在 `ComputeBufferPoolCounts`（`provider.compute_buffer_pool_counts()`），
测试自己的两条臂用 `provider.set_compute_buffer_pool(bool)`。

这五个与 `rb_buffer_n` 是同一个种群的两种读法：`hit + miss + disabled` 等于这一窗口里**每一次
创建**（建了的、取到的、没问的），`return + drop` 等于每一次 hand-back。渲染半边的同名计数是
`buffer_*_n`，两组分开是因为两条 rail 有自己的开关——一轮要能单独读出 compute 半边的复用。

## 4. A/B 读数

三臂同一个 exe 字节（sha256 `c506cf9e276b…`，`40-ab-exe-identity.txt`），launcher 只差
`METAL_API_VULKAN_COMPUTE_BUFFER_POOL=0` 一行；坐标 reims `35dcd14` × provider `8e2836e`，
300 s × 3 背靠背（每臂之间 settle 75 s），三臂都 attempt=1 / alive / 0 panic / `BOOT_EXIT=0`。
完整读数、形状表与红线在 `evidence/compute-buffer-pool-8e2836e-2026-09-20/`。

| µs/提交（窗口和 / Σn） | g3c（开） | g3c-off（关） | g3c-b（开） | 关 → 开（两开臂均值） |
|---|---|---|---|---|
| **`total`** | **2 208.1** | **2 371.0** | **2 115.2** | **−209.3（−8.8%）** |
| `resource_build` | 147.2 | 273.5 | 139.1 | −130.4 |
| ├ **`rb_buffer`** | **44.5** | **172.6** | **45.7** | **−127.5（−73.9%）** |
| └ `rb_pipeline`（未被碰） | 98.0 | 96.7 | 89.0 | −3.1 |
| `submit_teardown` | 23.5 | 148.7 | 21.7 | −126.1（−84.8%） |
| └ **`submit_td_buffers`** | **0.3** | **126.2** | **0.3** | **−125.8（−99.8%）** |
| `render_readback`（未被碰的对照条） | 78.5 | 75.1 | 73.6 | +0.9（+1.2%） |
| `render_landing`（未被碰） | 59.3 | 56.3 | 55.9 | +1.3 |
| `render_teardown`（未被碰） | 135.4 | 128.8 | 126.9 | +2.3 |
| `plan` / `pool` / `submit_validate`（纯 CPU，与本刀无交集） | 125.7 / 154.7 / 131.0 | 124.9 / 152.1 / 128.2 | 121.5 / 147.9 / 124.3 | −1.3 / −0.8 / −0.6 |
| `render_setup`（未被碰，宿主状态带） | 289.2 | 253.0 | 349.4 | +66.3 |
| 机制计数 `compute_buffer_hit_n` / `miss_n` / `disabled_n` / `return_n` | **1.020 / 0 / 0 / 1.020** | 0 / 0 / **1.020** / 0 | **1.020 / 0 / 0 / 1.020** | — |
| 人口 `rb_buffer_n` / `rb_memory_n` | **0.000 / 0.000** | 1.020 / 1.020 | **0.000 / 0.000** | — |
| 提交数 / 窗口数 | 63 488 / 248 | 60 416 / 236 | 52 480 / 205 | — |

**R 侧独立读数**（`frame_span`，与相位仪无关）：`prov_submit_us_mean` 按每 draw 折算
**352 973 / 373 100 / 321 695** —— 两侧量的是同一次调用差（E `total` −209.3 µs/提交，
R −35 766 µs/draw = −9.6%），差 0.8 个百分点以内。

**这一刀砍掉什么**

* **创建侧**：每提交 1.020 次创建降到 **0**（`rb_buffer_n` 1.020 → 0.000，`rb_memory_n` 同），
  `rb_buffer` −127.5 µs（−73.9%）——剩下的是"取池 + map + 拷贝"本身。
* **销毁侧**：`submit_td_buffers` 从 126.2 掉到 **0.3**（−99.8%）——那一档是"取不到就建"
  的形状；取到的 1.020 个 pair 走的是同一个 region 里的 unmap + 还回（`compute_buffer_return_n`
  = 1.020）。
* 两条合起来 **−253.3 µs/提交 = 关臂整笔提交的 10.7%**；整笔 `total` 掉 −209.3（−8.8%），
  差额来自**未被碰的渲染半边自己的宿主状态带**（`render_setup` 两个开臂之间就差 60 µs，
  关臂比开臂均值低 66 µs；`render_total` 整体 +81.1 µs），而不是这一刀的成本。

**没砍的、与边界**

* pipeline 仍然每提交重建一次（`rb_pipeline_n` 三臂都是 1.020）——G3-A 点名的第二名，
  是下一刀的候选。
* 覆盖面无回归：三臂都是 `out_of_class_shape_lines distinct=0`、`render_seam
  ok/ok_resident/out_of_class = 0/0/0`、`compute_provider_out_of_class=47`（与 g3a 同值），
  `pass_color_slots_1 == render_provider_canonical`（63 518 / 60 307 / 52 577），三个 shape
  表**逐字节相同**。
* "丢画"这条在本 regime 只有计数器 / 形状 / 路由三个口径：QMP `screendump` 拍到的三臂桌面
  都是全黑（与 g3a、sp11 同，属该 display 读不出来的已知边界），所以本轮**没有像素口径**，
  这一点在证据 README 的 Known limitation 里写明。
