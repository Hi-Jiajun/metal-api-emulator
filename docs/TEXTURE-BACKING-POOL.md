# 采样纹理的 backing 复用（E 侧增量）

本文件记录一次 provider 增量：**一个 pass 的每条采样声明不再各自重建 image /
memory / view 三件套，而是按「形状」从池里取上一次同形状声明留下的那一套**。它接在
`docs/SUBMIT-PHASE-PROFILE.md` 的相位表指出的刀口上：

* sp1 轮（reims `65db73f` × provider `07ffd55`，300 s 以上驻留、34 048 次提交）的相位表
  把 `render_setup` 放到 `provider.submit` 的第一位（**42.8%**），而它内部最大的一条是
  `setup_textures`（**39.9%，3 614 µs/submit**），比第二位 `render_readback`（20.0%）还高；
* guest 桌面的形状是「每 draw 一到两条全屏以上采样 + 一堆小纹理」：那一轮每个
  `store_routes` 窗口平均 110 次 `sampled_source_ge1m`、364 次 `sampled_source_lt64k`，
  `pass_sample_1` 与 `pass_color_slots_1` 一样是每 draw 一次；
* 每条这样的声明都走一遍 `vkCreateImage` + `vkAllocateMemory` + `vkBindImageMemory` +
  `vkMapMemory`/copy/`vkUnmapMemory` + `vkCreateImageView`，再用完就 `vkDestroyImage` /
  `vkFreeMemory`。形状不变、字节每次重写，只有**构造**是白花的。

## 1. 缺口

旧路径对每条采样声明都做一次完整的 backing 构造，而 guest 桌面反复提交同一小撮形状：
reims 侧的 `reuse_hit_n` 已经说明这一点（sp1 轮 0.9999 命中/提交，即每次提交一个同形状
pass）。渲染管线对象那一刀（`docs/RENDER-SETUP-REUSE.md`）已经把**形状决定的** shader
module / pipeline layout / pipeline 留住；剩下的 image、memory、view 也是形状决定的，
同样可以留。

## 2. 规则与证明

### 2.1 键就是交给驱动的结构

`BackingKey` 由即将写进 `vk::ImageCreateInfo` / `vk::ImageViewCreateInfo` 的字段拼成：
`image_type`、`format`、`extent[3]`、`view_type`，以及 `device_copy` 一位——后者决定
`OPTIMAL`+`TRANSFER_DST`+`DEVICE_LOCAL`+`UNDEFINED`（设备拷贝臂）还是
`LINEAR`+host-visible+`PREINITIALIZED`（宿主写入臂）。池的查找是**逐字段相等**，不做
摘要：形状的总体是几十个，线性扫描比它省下的驱动调用便宜得多，而摘要冲突这条失败方向
根本不存在。

### 2.2 entry layout：池子里的 image 处在 `GENERAL`

三条臂的 `record` 都在描述符绑定前把 image 发布成 `GENERAL`（宿主写入臂
`PREINITIALIZED → GENERAL`，设备拷贝臂 `UNDEFINED → TRANSFER_DST_OPTIMAL → GENERAL`，
pass-entry snapshot 臂由 `vkCmdCopyImage` 直接落到 `GENERAL`）。因此
`SampledTextureObjects.entry_layout` 记录的是「`record` 运行时 image 所处的布局」：
新建的宿主写入臂是 `PREINITIALIZED`，新建的设备拷贝臂是 `UNDEFINED`，**从池里取到的
是 `GENERAL`**；`record` 的三处 entry barrier 用这个字段当 `oldLayout`，于是复用来的
image 不会被从一个它并不在的布局里迁出（`GENERAL → GENERAL` 时这条 barrier 退化为纯
内存可见性声明，内容保留）。

### 2.3 什么时候还回去

与 `render_setup_reuse` 同一处、同一理由：`release_pooled_textures()` 在
`submit_and_wait` 成功返回后调用（`objects.release_reusable()` 旁边），此时 fence 已经
signalled，没有命令缓冲还在读这些 image；被还回去的句柄在 `SampledTextureObjects` 里
留空，pass 自己的 `Drop` 因此对它们什么都不做。失败路径不还——`Drop` 照旧销毁它们，
和这条路径在这个增量之前的行为逐字相同。

### 2.4 上限与失效

`ENTRY_CAP = 32` 个形状、`BYTE_CAP = 128 MiB`；超上限时从队首（最久未用）逐出并在
同一个锁里销毁。单个超过 `BYTE_CAP` 的 backing 根本不入池，直接销毁。开关关掉时
`clear()` 掉全部条目；设备丢失重建会连同旧 context 一起丢掉整张表。

## 3. 开关与读数

| 值 | 效果 |
|---|---|
| unset（默认）、`1`、`on`、`yes`、`true`、其它 | 开 |
| `0`、`off`、`no`、`false` | 关：每条声明一次 relaxed load，不留任何设备对象 |

对照臂就是**同一个 exe** 加 `METAL_API_VULKAN_TEXTURE_BACKING_POOL=0`。
provider 另给了 `set_render_texture_pool(bool)` / `render_texture_pool_counts()` /
`render_texture_pool_enabled()` / `invalidate_render_texture_pool()`，所以一个进程里
两个臂可以对着同一台设备跑（`tests/render_texture_pool_e2e.rs` 就是这么比的）。

相位行新增两组读数：

* `texture_backing_us` / `texture_upload_us`——**嵌套在 `setup_textures` 里面**的两条：
  前者是 `vkCreateImage`+`vkAllocateMemory`+`vkBindImageMemory`（命中时几乎为零），后者
  是 texel 自己进 image 的那一趟（宿主写入臂的 map/copy/unmap，volume 臂的 staging 写）。
  它们不是与 `setup_*` 并列的条，**不得**加进互斥和里；
* `pool_hit_n` / `pool_miss_n` / `pool_disabled_n` / `pool_return_n` / `pool_drop_n`——
  每条采样声明的归属（取到 / 自建 / 开关关着），以及一次 pass 结束时 backing 的去向
  （留在池里 / 被销毁）。

## 4. 边界

这一刀只动**构造**，不动字节：texel 仍然由各臂自己写进 image（宿主写入臂的
map/copy/unmap、无拷贝臂的设备 `vkCmdCopyBufferToImage`、snapshot 臂的 `vkCmdCopyImage`），
descriptor、sampler、render pass、framebuffer、readback buffer、命令池与 fence 全部照旧
per pass。所以帧字节不变是构造性的，真机轮与 rail 用例只是复核这一点。

它也不改变准入面：池不拒绝任何形状，命中与否只决定构造成本的归属。读数只代表本机本配
（WSL + patched WHPX + RTX 5060 + 该轮的 Dozen 队列），不外推成「canonical provider
一定慢」，也不构成完整 Metal conformance 证据。
