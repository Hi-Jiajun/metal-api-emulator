# owner 窗口的 host 指针 import 复用（E 侧第二刀）

本文件记录一次 provider 增量：**一条 no-copy 采样声明不再每次 pass 重新 import 一次
owner 的窗口，而是按「范围 + 用途」从池里取上一次同范围声明留下的那次 import**。它接在
`docs/SUBMIT-PHASE-PROFILE.md` 的相位表上，是「残差拆条」那一轮（sp4）读数直接选出来的
刀口。

## 0. 一轮读数：残差与纹理缝各是谁

第一刀（`docs/TEXTURE-BACKING-POOL.md`）之后，sp1 相位表里那条 20.4% 的「未命名 render
工作」就成了最大的黑盒。sp4 轮（reims `65db73f` × provider `407fb7c`，300 s 驻留、
115 个窗口 / 29 440 次提交）把两处黑盒都拆成了命名条：

| 条 | µs/submit | 占 total |
|---|---|---|
| `render_readback` | 1 891.4 | 28.5% |
| `render_setup` | 1 511.1 | 22.7% |
| ├ **`texture_import`** | **1 117.7** | **16.8%** |
| ├ `texture_upload` | 16.5 | 0.2% |
| └ `texture_descriptor` / `texture_sampler` / `texture_backing` / `texture_view` | 5.6（合计） | 0.1% |
| `render_landing`（kept-frame 落地） | 991.7 | 14.9% |
| `render_teardown`（pass 对象销毁） | 566.8 | 8.5% |
| `render_wait` | 274.1 | 4.1% |
| `render_prepare` / `render_resolve` / 其余残差 | 177.7 | 2.7% |
| **total** | **6 642.7** | 100% |

读数给出三件事，都是这次增量之前不能说的：

1. **纹理缝不在 view / sampler / descriptor 上**：四者合计 5.6 µs/submit。把它们池化是
   一条被证否的路，省不到总成本的千分之一。
2. **`setup_textures` 剩下的 98% 是一次 host 指针 import**：`texture_import_us`
   1 117.7 µs/submit，而这一轮每条提交平均 0.73 条采样声明（`pool_hit_n` 0.729），
   折算**每次 import ≈ 1.5 ms**——这是五次驱动调用买到的全部东西，而且下一条声明（同一个
   窗口）会再买一次。
3. 第一支配项是 `render_readback`（28.5%），但它是 guest 要求观测的**字节**；残差里真正
   白花的是这一条 import：范围相同、用途相同、页是 owner 自己的页，每次重建只是让驱动
   重新把这段 host 范围做成可用的 device memory。

## 1. 缺口

`execute_render_pass` 的 no-copy 臂（`research/docs/23` §75, R5c）解析出
`RenderInputSource::Borrowed { window, .. }` 之后，`create_render_textures` 会为它
`import_host_pointer_buffer`：

| 调用 | 作用 |
|---|---|
| `vkCreateBuffer(len = window.len)` | 描述 owner 的那段页 |
| `vkGetBufferMemoryRequirements` | 拿要求（后面用它与 lease 的 capacity 比对） |
| `vkGetMemoryHostPointerPropertiesEXT` | 问驱动这段指针能不能被 import |
| `vkAllocateMemory(host pointer import)` | 真正的代价所在 |
| `vkBindBufferMemory` | 绑上 |

同一段窗口在 guest 桌面上是被反复采样的（合成器每帧都把同一批 surface 交给 GPU），
而 provider 每一 pass 都把它当新窗口重建一次。窗口的**身份**是
`(pointer, len, usage)`：这三项相同，驱动被交付的就是同一段 host 范围。

## 2. 规则与证明

### 2.1 键就是交给驱动的字段

`ImportKey = (pointer, len, usage)`，逐字段相等、不做摘要：`len` 就是
`vk::BufferCreateInfo::size`，`pointer` 就是
`vk::ImportMemoryHostPointerInfoEXT::host_pointer`，`usage` 决定 buffer 的 usage 位。
池的查找是线性扫描（窗口总体是一小撮），因此不存在「摘要撞车」这条失败方向。

同一个地址被后来的窗口复用时，**它就是同一段 host 范围**：import 描述的是 owner 声明
的那段地址，新窗口写进去的字节在执行时被读到，和重新 import 一次读到的是同一批字节
（`docs/23` §74：窗口的字节由 owner 在 pass 执行时决定，不是 provider 的快照）。

### 2.2 命中省掉的是 import，不是读

命中时 pass 仍然照旧记录同一个 `vkCmdCopyBufferToImage`（或它对同一个 buffer 的
其它读取），只是不再向驱动要一次新的 import。因此「每条采样纹理读到的是本轮声明自己的
字节」这条性质逐字节不变；`crates/metal-api-vulkan/tests/render_texture_extent_nocopy_e2e.rs`
（no-copy 臂的逐字节 oracle，5 例）在**开关开、关两臂**都全绿。

### 2.3 什么时候还回去

与 `render_texture_pool` 同一条规则、同一个位置：`release_imported_windows()` 在
`submit_and_wait` 成功返回后调用（`release_reusable()` / `release_pooled_textures()`
旁边），此时 fence 已经 signalled，没有命令缓冲还在读这些页。还回去时
`SampledTextureObjects.copy_source` 与 `imported` 都被取空，pass 自己的 `Drop` 因此对
它们什么都不做；**失败路径不还**——`Drop` 照旧销毁，和这条路径在增量之前逐字相同。

### 2.4 上限与失效

`ENTRY_CAP = 64` 个窗口、`BYTE_CAP = 192 MiB` host 范围；超上限从队首（最久未用）逐出并
在同一个锁里销毁。单个超过 `BYTE_CAP` 的窗口根本不入池，直接销毁。开关关掉时
`clear()` 掉全部条目；设备丢失重建会连同旧 context 一起丢掉整张表。

lease 的 capacity 检查**在命中时也要重做**：池里的 entry 带着当初驱动为这段范围给出的
`requirements.size`，它与新声明那条 lease 的 capacity 比对，超了就当 fresh 路径那样按名
拒绝（fail-closed），不会因为「池里有」而放行一条 fresh 路径会拒绝的绑定。

## 3. 开关与读数

* `METAL_API_VULKAN_RENDER_IMPORT_POOL=0`（`off` / `no` / `false` 同义）为对照臂：
  不取、不留，每条声明一次 relaxed load；默认开。
* 相位行新增五个窗口计数：`import_hit_n` / `import_miss_n` / `import_disabled_n` /
  `import_return_n` / `import_drop_n`，与 `pool_*` 同义；累计读数经
  `VulkanExecutor::render_import_pool_counts()` / `VulkanComputeProvider::render_import_pool_counts()`
  取出，同进程双臂经 `set_render_import_pool(bool)` 切换。

## 4. A/B 读数（同一个 exe 字节）

三臂同一个 exe（SHA-256 见 `01-ab-exe-identity.txt`），launcher 只有一行不同（`02-launcher-diff.txt`）：
`sp5`（默认，开）→ `sp6`（`…_IMPORT_POOL=0`）→ `sp5b`（默认，开），背靠背、同一宿主窗口。

| µs/submit | sp5（开） | sp6（关） | sp5b（开） | 关→开 |
|---|---|---|---|---|
| `total` | **5 422.2** | **7 027.0** | **5 340.7** | −1 604.8 / −1 686.3 |
| `render_setup` | 435.7 | 2 071.3 | 426.2 | −1 635.6 / −1 645.1 |
| ├ `setup_textures` | **30.7** | **1 719.0** | **26.0** | −1 688.3 / −1 693.0 |
| └ `texture_import` | **9.1** | **1 696.6** | **4.1** | −1 687.5 / −1 692.5 |
| `render_readback`（对照条，未碰） | 1 858.6 | 1 858.5 | 1 860.9 | −0.1 / +2.3 |
| `render_landing`（对照条，未碰） | 982.0 | 982.7 | 982.0 | −0.7 / −0.7 |
| `render_teardown`（对照条，未碰） | 514.7 | 513.7 | 515.8 | +1.0 / +2.1 |
| `import_hit_n` / `import_miss_n` / `import_disabled_n` | 0.520 / 0.001 / 0 | 0 / 0 / 0.533 | 0.523 / 0.001 / 0 | — |

R 侧同一对臂的独立读数（`frame_span`/`frame_profile`）：`prov_submit_us_mean`
**5 470 → 7 066 → 5 380 µs/draw**，`host per draw` 7 006 → 8 466 → 6 799 µs；
两侧量的同一次调用差在 **0.5%** 以内。公平性（每窗口工作量、
`render_readback`/`render_landing`/`render_teardown` 三条未被碰的条三臂几乎逐位相同）见
报告 §3.2。

**这一刀砍掉什么、剩下什么**：砍掉每条 no-copy 采样声明的四次驱动调用（含那次
`vkAllocateMemory`），实测 `setup_textures` −98%、整笔 submit −23%；**没砍**的是 texel 的
那一趟（`texture_upload` 15 不变）、pass 自己的对象构造与销毁（`render_teardown`
514.7 不动）以及 `render_readback` / `render_landing`（guest 要求观测与落地的字节）。
下一段的入口按读数排序是 `render_readback`（1 858.6，28–34%）与 `render_teardown`
（514.7）。
