# 按写入矩形回读（E 侧增量）

本文件记录一次 provider 增量：**render 半边的回读只搬本 pass 实际写入的矩形**，
其余帧字节由「image 这份 pass 开始时吃进去的 seed」在宿主侧重建。它接在
`docs/SUBMIT-PHASE-PROFILE.md`（fp9/fp9b 相位表）指出的刀口上：

* fp9b 里 `provider.submit` 的 86% 是 render 半边；再切一刀，
  **`render_readback` = 50.3%（3.86 ms/draw）**，按 guest 主形状家族
  1920×1080×4 B 折算 ≈ 8 MB/draw（≈ 2.15 GB/s）——这是单笔最大的宿主成本；
* 同一轮的 census 形状分布里，大量记录只写几千个 texel
  （`draw_scissor_partial`、`draw_scissor_area_le25/_le10/_lt1` 一类）。

## 1. 缺口

旧回读只有一种形状：每个 stored 附件的 `vkCmdCopyImageToBuffer` 把
`width × height × texel` 整幅拷进 staging buffer，宿主再从那块映射里
`to_vec()` 同样多的字节。对一次 80×64 的更新抬 8 MB 回来，设备拷贝和宿主读都白花。

## 2. 规则与证明

### 2.1 写入矩形

单 sample 的 pass 只可能在

```
rect = viewport ∩ scissor ∩ extent
```

之内写像素：

* pass 的 render area 是附件整幅（`render_area = (0,0,width,height)`，
  `research/docs/23` §3.3）；
* 光栅化只产生落在这个 viewport 矩形里的 fragment（NDC 裁剪体映射到它，
  `research/docs/23` §3.1，v100）；
* scissor 测试把矩形外的 fragment 全部丢掉（`research/docs/23` §3.3，v29）。

两条声明都由 core 契约钉在附件 extent 之内（`ViewportOutsideAttachment`、
`ScissorOutOfBounds`），所以矩形本身就是「可能被写的 texel」的一个**上界**。

### 2.2 seed

矩阵外的 texel 保持 image 被 seed 时的字节：

| load 臂 | 未写区域的字节 | 本 rail 是否有宿主侧副本 |
|---|---|---|
| `Clear(payload)` | payload 在整幅上重复（clear 覆盖 render area） | 有（payload 就在请求里） |
| `Load` | 附件自己那份声明的字节（`vkCmdCopyBufferToImage` 上传） | 有（`TraceBytes`/`StagedBytes`/`ProducedBytes`/`GatheredBytes`，或 owner 的 live window） |
| `Resident` | provider 自己 image 里的字节 | **没有** → 退回整幅 |
| `DontCare` | 未定义 | **没有** → 退回整幅 |

`Load` 源的长度由 `resolve_attachment_load` 钉成附件紧密 extent，所以「源字节 =
帧字节」是构造性的，不是估计。

### 2.3 逐字节等价

于是「只读矩形 + 用 seed 重建」与「整幅回读」发布的是同一份帧：

* 矩形内：两者都取设备上 image 的字节（同一次 `vkCmdCopyImageToBuffer`，
  只是拷贝区域不同，Vulkan 保证区域拷贝的 texel 值就等于整幅拷贝在同一位置的
  texel 值）；
* 矩形外：整幅回读取的是 image 从未被写过的 texel，而 §2.2 的 seed 就是
  image 被写之前的那份字节。

因此**两个落地点**都逐字节不变：writeback 通道递出的帧，以及 owner 窗口
（guest 页）收到的帧（`land_owner_windows`，`research/docs/23` §114/§115）。

## 3. 回读与发布路径

1. **设备拷贝区域**：证明成立时 `VkBufferImageCopy` 只带
   `imageOffset = (rect.x, rect.y)`、`imageExtent = (rect.width, rect.height)`，
   `bufferRowLength = 0`（紧密行）——矩形自己紧凑地落在映射起点。
   空矩形（scissor 与 viewport 不相交，此时 draw 一个 fragment 都不产生）
   **不记录任何拷贝**（`imageExtent` 不允许为 0）；
2. **staging buffer 大小**：按矩形的字节数分配（原来固定整幅），
   宿主因此只读矩形的行；
3. **重建**：`frame = seed`，再把矩形那份设备字节按行钉回 `frame` 的对应位置
   （`patch_rect`）。`Clear` 走 `clear_frame`（按 4 KB pattern 展开，不是逐 texel），
   `Load` 走 seed 的字节；
4. **retain 顺序**：`Load` 的 seed 可能是 owner 的 live window
   （`BufferSource::BorrowedNoCopy`，census 的 `channel=borrowed no_copy=1`），
   它的可读性由这次提交的 retain 兜着。所以回读（含 seed 拷贝）**先做**、
   `retain.retire()` **后做**；pass 自己的 landing（写回那个窗口）在回读返回之后才跑，
   读到的必然是 pass 之前的字节；
5. **present 臂不在本增量**：present 的帧由 present action 自己读目标 image
   （`docs/24` §3.3），整幅语义更硬，本轮保持整幅。

## 4. 开关与计数

| 开关 | 效果 |
|---|---|
| `METAL_API_VULKAN_FULL_READBACK` 未设/其它 | **默认**：按写入矩形回读 |
| `=1` / `on` / `true` / `yes` | 退回增量之前的整幅回读（对照臂，进程级读一次） |

计数分两处，单位和用途不同：

* `VulkanExecutor::readback_region_counts()`：进程级、常开、累计
  （`rect_attachments` / `rect_bytes` / `rect_extent_bytes` / `full_*` /
  `switch_*` / `shape_*` / `bounds_*` / `whole_*`）。e2e oracle 用它判定
  某个形状走了哪条臂，不需要打开相位表；
* `PHASE submit` 行新增的 `readback_*` 字段：与其它字段一样是**该窗口的和**，
  于是「每次提交回读了多少字节」可以像其它相位一样相加再除。

退回整幅的每一种形状都单独计数（`shape` = seed 不在宿主、
`bounds` = 声明矩形不可证、`switch` = 对照开关、`whole` = 矩形就是整幅），
不合并成一个「fallback」。

## 5. 边界

1. **只覆盖单 sample 颜色附件**：多 sample 的可观测字节是 resolve target 的，
   resolve 会写满它，所以 `Multisample` 一律退回整幅（census 的 desk 形状全是
   `samples=1`，本轮不受损失）。深度/模板/可写 stage buffer 的回读保持整幅。
2. **多矩形**：当前请求形状是「一个 pass 一个 viewport + 一个 scissor」，
   所以「多个矩形取并集」退化成单个矩形本身；契约将来若长出多组 viewport/scissor，
   上界要按并集重述（并集之后仍要与 extent 相交），本增量不预先实现。
3. **负 origin 不可表示**：`viewport` / `scissor` 的四个分量都是 `u32`，
   core 契约也把两者钉在 extent 内，所以「负 origin」在类型上就不存在；
   越界/空矩形仍然按 `bounds` 退回整幅并计数。
4. **未覆盖的形状不赚也不亏**：矩形等于整幅时退回整幅（计 `whole`），
   避免为「只写几个 texel」之外的形状多走一趟重建。
5. **帧字节不变**：本增量不改 wire、不改 `RenderPassDescriptor` 语义、
   不改发布通道的字节。它是纯宿主侧搬运形状的变化。

## 6. 证据与复跑

```sh
W=/home/hiliang/hackintosh/worktrees/metal-readback-rect
# 单元：矩形决策与重建的纯函数
cargo test -p metal-api-vulkan --locked --lib readback_rect
# rail 级字节 oracle：两个进程分别跑「矩形臂」与「整幅臂」，逐字节比帧与 guest 页
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  cargo test -p metal-api-vulkan --locked --test render_written_rect_readback_e2e
REPO=$W bash /home/hiliang/hackintosh/tools/gates-local.sh
REPO=$W bash /home/hiliang/hackintosh/tools/lavapipe-smoke.sh
```

真机轮与前后表见 `evidence/readback-rect-<sha>-2026-09-19/README.md`。
