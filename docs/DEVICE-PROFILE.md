# 设备 profile：把"驱动声明"变成可存档、可 diff、可复用的文件

开关 `METAL_API_VULKAN_DEVICE_PROFILE`（**默认关**）。机制写在
`crates/metal-api-vulkan/src/device_profile.rs`；工具（探针 + 两个 Python 脚本）
在本工作区的 `tools/device-profile/`（**不入库**）。本文记录**为什么是这一层**、
**两台设备的读数**与**不碰什么**。

## 0. 缺口

能力声明的真值来自三层：**驱动声明**（Vulkan 查询）+ 我们的实现窗口 + 实测 quirk。
第一层原本散在能力帧日志、RTF 那轮的 `vk-image-shape-probe`、census 汇总与 RTX
验收轮里——没有一份"这台 (GPU, 驱动) 到底是什么样"的可存档文件，所以换卡/换驱动
版本时只能重跑全套，也答不出"相比上一轮**变了什么**"。

RTF 那轮的 README 正文把驱动写成 `616.368`，而**同一轮自己**的 transcript 里
`nvidia-smi` 读的是 `616.92`；本轮探针与产品读到的驱动自述也是 `616.92`。
同一个设备、两个写法，真值只有一个——这正是"引文与文书会跑偏、而文件不会"的例子。
另外记住：NVIDIA 的 `VkPhysicalDeviceProperties.driverVersion` 按 Vulkan 通用布局解出来
是 `104.368.0`，**别拿它当驱动版本**；profile 里两样都记（`driverVersion` = 驱动自述，
`driverVersionVulkan` = 通用解码）。

## 1. 三件东西

1. **独立探针** `tools/device-profile/probe/`（ash 0.38，WSL 与 Windows 都跑，
   Windows 侧从 WSL 交叉编译即可，不需要 MSYS2 工具链）→ 一份 JSON；
2. **产品启动 dump**：本开关打开时，`VulkanContext::new()` 在设备建好之后把**同一套
   读数**以 `DEVICE_PROFILE` 行块写到进程 stderr（真机轮里就是该轮的 boot log），
   `tools/device-profile/from-log.py` 把它转回**同一份 JSON 形状**；
3. **diff**：`tools/device-profile/diff.py` 分四类输出（数值 / 格式能力 / feature /
   形状），并能把多份 profile 与轮次读数排成一张 markdown 矩阵。

canonical 集合（两边逐条一致，改一处必须改另一处）：
9 个格式 × {`LINEAR`,`OPTIMAL`} × {`SAMPLED`, `SAMPLED|TRANSFER_DST`,
`COLOR_ATTACHMENT|SAMPLED`（只 2d）, `STORAGE`} × 5 个形状
（`1920x1080x1`、`1024x1024x1`、`4x4x1`、`64x64x8`、`16384x1x1`）= **324** 条；
外加 identity / 60 条 limits / 60 条 features / 队列族 / 内存堆与类型。

## 2. 为什么真建对象

每条形状记录 `vkGetPhysicalDeviceFormatProperties` 的两个 tiling 特征字、
`vkGetPhysicalDeviceImageFormatProperties` 的答案，**以及真 `vkCreateImage` 的成败**。
RTF 那轮已经把理由写死：RTX 5060 上 `R32_SFLOAT` 的 `LINEAR` 特征字 `0x1dd07`
**带 `SAMPLED_IMAGE`**，而查询与真建都答 `ERROR_FORMAT_NOT_SUPPORTED`。
本轮 324 条形状的读数复现了同一件事（`evidence/device-profiles-2026-09-22/03-*.md`）：

| 设备 | 拒收条数 | 被拒的 (类型, tiling) 块 |
|---|---|---|
| NVIDIA GeForce RTX 5060（`616.92`） | 54 / 324 | `1d`+`LINEAR`、`3d`+`LINEAR` |
| llvmpipe / Lavapipe（`Mesa 26.2.2`） | 0 / 324 | — |

被拒的 54 条里，设备拷贝载具（`OPTIMAL` + device-local + `TRANSFER_DST`）的
create/allocate/bind **三步全 OK**——这正是 `render_texture_carrier` 走的那条路。

## 3. 开关的读数

* **关（默认）**：设备构造里一次 relaxed load，零驱动调用。关臂 141 份 capture 与
  base `a46ffb4` 的 141 份 **sha256 逐个相同**（`24-off-vs-base.diff` 空）。
* **开**：`DEVICE_PROFILE` 行块 459 行（1 begin + 1 identity + 60 feature + 60 limit +
  1 queue_family + 堆/类型 + 9 format + 324 shape + 1 end；Lavapipe 无 carrier 行，
  RTX 有 54 行）；同一套件在开启下的 capture 与关臂 **逐字节相同**
  （`d988fc8a…`）= 开关只加日志，不动产品行为。
* **探针 ↔ 产品交叉验证**：同一台 Lavapipe 上，探针 JSON 与"产品行块 → from-log.py"
  的 JSON 在 `host`/时间戳之外**逐字段相同**（`12-diff-probe-vs-product-lavapipe.md`）——
  两份 canonical 表与两套实现互为对照。

## 4. 不碰什么

* 不改任何既有开关的语义，不加新的默认路径；关时连一次驱动查询都不多发。
* **计数 / 计时轮不要开**：形状表真建 324 个 image（随即销毁），census 会把这些真实
  对象数进去。它是取证开关，不是生产开关。
* 行块只写 stderr：R 侧 `/tmp/reims-vgpu-fail.log` 那条通道不受影响，`from-log.py`
  两种日志都能读。
