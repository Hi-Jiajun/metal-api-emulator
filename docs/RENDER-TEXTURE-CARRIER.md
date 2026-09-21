# 渲染纹理载具：设备不收 LINEAR 的形状改走设备拷贝（E-RTF1）

E 侧一刀（`render_texture_carrier`，开关
`METAL_API_VULKAN_RENDER_TEXTURE_LINEAR_FALLBACK`，**默认关**）。机制与证明见模块文档
`crates/metal-api-vulkan/src/render_texture_carrier.rs`；本文记录**用户可见黑屏的根因**、
**真值表**、**刀的形状**与不碰什么。任务书
`.agents/tasks/root/e_render_texture_format_rtx.md`，报告
`.agents/tasks/root/e_render_texture_format_rtx-report.md`，证据
`evidence/rtx-texture-format-<tip>-2026-09-21/`。

## 0. 根因：一维 LUT 的 LINEAR 图像，不是二维窗口

任务书给的初判是"`create render textures` 的二维 `LINEAR` 臂被 RTX 拒"，凭据是 it3 用户段
里那条
`render_execution_failed: create render texture image: ... Requested format is not
supported on this device pipe=30` 后面紧跟 `1920x1080` 的 `draws_skipped`。

**真值表把这条初判改正了**：RTX 5060 上二维 `LINEAR` 全部可建（`bgra8_unorm`、
`bgra8_srgb`、`rgba8_unorm`、`rgba8_srgb`、`rgba16_float`、`r32_float`、`r32_uint`、
`r16_float`、`r8_unorm` × `{1920x1080, 1024x1024, 4x4}` × `{SAMPLED,
SAMPLED|TRANSFER_DST}` 都是 `create=OK`）。被拒的是**一维与三维的 `LINEAR`**
（与格式无关：1D 的 `R32_SFLOAT`/`R8_UNORM`、3D 的 `R32_SFLOAT`/`B8G8R8A8_UNORM`
四种组合全拒，`ifp=ERROR_FORMAT_NOT_SUPPORTED create=ERROR_FORMAT_NOT_SUPPORTED`）。

it3 那一段的同一 pass 里确实带着这样一条声明：

```
sampled_ref_backing task=1 ref=27 view=16384x1 fmt=0x64 mid=0 map=0x0 map_fmt=0x0 route=guest_runs
```

`fmt=0x64` = `VK_FORMAT_R32_SFLOAT`（100），`view=16384x1` 是 census b10 记的
**单行 LUT**（`texture1d_array`，`MTLTextureType1DArray`，`16384x1`）——它就是那条被拒的
`vkCreateImage`。它的 `route=guest_runs` 说明字节走的是**上传臂**（provider 自己 gatter 的
guest runs），于是 `create_render_textures` 按上传臂给了 `LINEAR` + host-visible；
NVIDIA 对 `TYPE_1D` 根本不给 `LINEAR`，`vkCreateImage` 直接拒。

后果是整条 pass 被跳过：`draws_skipped_after_engine_refusal`（it3 段 160 次）、
`present_black`/`present_black_retain`、之后 124 次 `present_unbacked`、0 次新内容发布——
窗口从 t≈27 s 起全黑到结束。同一形状在 it2 用户段已经出现过（6 条 `class=execute`、
396 次跳过），**不是今晚的回归**，而是一维 lane 落地时只在 Lavapipe 上验过：
Lavapipe 的 `1d-16384x1-LINEAR` 两种 usage 都是 `create=OK`。

## 1. 真值表（124 个形状，RTX 5060 driver 616.368 / Lavapipe LLVM 22.1.8）

探针 `vk-image-shape-probe`（ash 0.38、`Entry::load()`、不建资源以外的任何东西）对每个形状
问三件事并打印三份答案：`vkGetPhysicalDeviceFormatProperties` 的两个 tiling 特征字、
`vkGetPhysicalDeviceImageFormatProperties`、以及**真的** `vkCreateImage`（按 rail 自己的
`VkImageCreateInfo`：单 mip、单层、单采样、`EXCLUSIVE`、`LINEAR` 用
`PREINITIALIZED`）。被拒形状再按**修法载具**真建一次（`OPTIMAL` +
`SAMPLED|TRANSFER_DST` + `UNDEFINED` + `DEVICE_LOCAL` 分配 + 绑定）。

| 形状 | RTX 5060 | Lavapipe |
|---|---|---|
| `2d` × 9 格式 × 3 extent × 2 usage × `{LINEAR, OPTIMAL}` | **全 `create=OK`** | 全 `create=OK` |
| `1d-16384x1` × `{R32_SFLOAT, R8_UNORM}` × 2 usage × `LINEAR` | **4/4 拒**（`ERROR_FORMAT_NOT_SUPPORTED`） | 全 OK |
| `3d-64x64x8` × `{R32_SFLOAT, B8G8R8A8_UNORM}` × 2 usage × `LINEAR` | **4/4 拒** | 全 OK |
| 上面 8 条被拒形状的**修法载具**（`OPTIMAL`+`TRANSFER_DST`+`DEVICE_LOCAL`） | **`create=OK memory_type=1 bind=OK` 8/8** | — |

即：**唯一的被拒维度是"非二维 + LINEAR"**，而修法载具在这台设备上 8/8 建得出来也绑得上。
原始表：`<evidence>/02-shape-table-rtx.txt`、`<evidence>/01-shape-table-lavapipe.txt`。

## 2. 刀的形状

* `crates/metal-api-vulkan/src/render_texture_carrier.rs`（新模块）：
  * `enabled_from_env()` —— 开关，默认**关**；`1`/`on`/`ON`/`true`/`yes` 才开。
  * `ask_linear_admission(..)` —— 对形状问设备，返回四份读数（查询结果、extent 是否落在
    `maxExtent` 内、`LINEAR` 特征字是否有 `SAMPLED_IMAGE`、两个 tiling 的特征字），
    并能在同一形状的**拒绝**上挂成字段 / 一行 `k=v`。
  * `uploaded_shape_takes_device_copy(..)` —— **先读开关**（关时一次驱动调用都不加），
    开时"设备不收这个 LINEAR 形状"就返回 `true`。
* `render.rs::create_render_textures`：`device_copy` 的第四项就是上面这一问；被升上去的声明
  与卷走**同一条**载具（`OPTIMAL` + `TRANSFER_DST` + `DEVICE_LOCAL`），字节由
  `upload_render_texture_into` 的 staging 缓冲 + `record` 的一次
  `vkCmdCopyBufferToImage` 送进去。`upload_render_texture_into` 的 `volume: bool` 改名为
  `device_copy: bool`（同一面旗子：镜像的 tiling/usage/initial layout 就是由它决定的）。
* 纹理池键本来就把 `device_copy` 算进去（`render_texture_pool::BackingKey::new`），
  所以设备拷贝建出来的 backing 不会被发给上传臂的声明。

## 3. 具名诊断（同刀，零提交字节变化）

`create render texture image` 这条拒绝此前只有驱动自己的文本，看不出一整个 pass 里**哪一条**
声明被拒（it3 的 pass 同时带着 `1920x1080` `bgra8_unorm` 窗口与 `16384x1` `r32_float` LUT）。
现在同一形状与设备的答案两边都落地：

* 结构化字段：`vk_format`、`view_format`（trace 自己的 `TextureFormat` 名）、`image_type`、
  `tiling`、`usage`、`extent`、`device_copy`、`source`（五条来源臂之一）以及
  `device_linear_answer` / `device_linear_sampled` / `device_linear_features` /
  `device_optimal_features` / `device_extent_admitted`；
* 同一句话尾追在 detail 里（owner 侧边界只打 detail、丢字段，所以只落字段等于没落）：
  `... (shape: view_format=R32Float vk_format=100 type=1d extent=16384x1x1 tiling=linear
  usage=0x4 carrier=host_visible source=gathered_bytes; device: linear_answer=... )`。

detail 的前半段与拒绝的 step/slug 逐字未动，所以既有 grep 口径不变。

## 4. 开与关的位置

| 项 | 关（默认） | 开 |
|---|---|---|
| 驱动调用 | 一次也不加 | 每条上传声明一次 `vkGetPhysicalDeviceImageFormatProperties` |
| 载具 | 与刀前逐字节同形 | 设备不收 `LINEAR` 的声明改设备拷贝 |
| 二维上传臂 | `LINEAR` + host-visible | 不变（设备收） |
| 一维 LUT / 三维卷 | 一维：`vkCreateImage` 被拒 → 整条 pass 跳过；三维：早就设备拷贝 | 一维改设备拷贝；三维不变 |

## 5. 不碰什么

* **契约/wire 一行不动**：形状、能力帧、tag 都没有新段，R 侧不随动。
* **二维上传臂不动**：47 套件 141 份 capture 里绝大多数采样的仍是 `LINEAR` host-visible 路
  （设备收这个形状，刀不下手）。
* **卷的既有设备拷贝臂不动**（它本来就是为同一个设备限制落的）。
* 开关关时**连一次驱动问询都不发生**，`LINEAR` 特征字、`maxExtent` 都不读。
