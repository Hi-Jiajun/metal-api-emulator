# 资源池一次派生：一次提交不再为同一张表走两遍

E 侧第十一刀（`submit_pool_once`，开关 `METAL_API_VULKAN_SUBMIT_POOL_ONCE`，**默认关**）。
机制与证明见模块文档 `crates/metal-api-vulkan/src/submit_pool_once.rs`；本文记录
**为什么是这一处**、**两臂读数**与**不碰什么**。

## 0. 缺口：同一张声明表在一个提交里走了两遍

形状轮（tag `ct1`，E `537c1d1` × R `a31bf9f9`，300 s 生产姿态、29 994 次提交）
把内容条的长尾钉出来之后，尾部（top 5%，占全轮 `total` 的 39.3%）里最大的两条 lane 是：

| lane | 全轮均值 | 该条落尾比 | 是什么 |
|---|---|---|---|
| `plan_resources` | 15.08 ms/帧 | **96.0%** | `plan` 里的 `ComputeTrace::serial_resources_ref` |
| `submit_validate_derive` | 15.43 ms/帧 | **93.5%** | `submit_validate` 里的同一次调用 |

第七刀（`docs/SUBMIT-RESOURCE-BORROW.md`）已经把两次派生都改成**借出**（不再克隆字节），
但两次**走查**都还在：同一次提交对同一份不可变的 trace 派生两遍。两遍的成本逐提交相等
（尾部均值 4 151.2 µs vs 4 138.6 µs，差 −12.6 µs；`plan_resources ≥ 1 000 µs` 的分层里
4 631.7 vs 4 616.6 µs），而长尾的形状正是把它放大的那一类——大声明：
`views_bytes > 8 MiB` 的桶里 `plan` 侧 887.4 µs、`≤ 64 KiB` 只有 153.1 µs。

## 1. 本刀的形状

`plan` 派生出来的那张 `pool` 在终结校验时**还活着**：执行器、渲染半边、以及
`pool_geometry` 都还在读它，它要到 `submit_release` 才析构。所以这一刀不是缓存、不是记忆化，
而是**交接**：

* 开关开时，`submit_validate` 的走查直接读 `pool`（`validate_with_pools(trace, pool, …)`），
  这一块**不做第二次派生**；
* 开关关时，这一块照旧派生自己那张表（拥有臂 `serial_resources` / 借出臂
  `serial_resources_ref` 两条路径原样保留），释放段释放的对象与之前逐字相同；
* 纹理那一半（`serial_texture_resources`）不受影响，两条臂都照常派生。

## 2. 为什么驱动看不出区别

1. **同一张表。** `serial_resources_ref` 是 `&self` 上的纯走查：同样的长度、同样的首次使用
   顺序、同样的 access 并集、同样的声明字节。trace 在整次 `submit` 里不可变，`plan` 派生时
   的输入与校验派生时的输入是同一份。
2. **借用不可能悬垂。** 表在 `plan` 里声明，晚于它的执行器/绑定/渲染半边都只读它，
   而它的析构在校验走查**之后**——这正是刀前第二张表的位置（它在 `submit_release` 里释放）。
3. **释放记账跟着刀走。** `submit_validate_release` 现在释放"这一块自己派生过的那两张表"：
   关臂是同一个 `free`（逐字同形），开臂两张都是 `None`、只剩纹理表。
4. **别的都没动。** 走查读到的引用、writeback 映射、覆盖检查、纹理池与 admission 一字未改；
   两臂的 141 份 capture 逐字节相同（§3）。

## 3. 开关与两臂读数

`METAL_API_VULKAN_SUBMIT_POOL_ONCE=1`（也接受 `on`/`ON`/`true`/`yes`）打开；**默认关**，
其余取值（含未设）都是刀前的路径，也就是两臂里的对照臂。

两臂（`ct1a` = off / `ct1b` = on，其余逐行同形、同一把锁背靠背、`DWELL=300`、
各 420 帧）：

| 读数 | off | on | 差 |
|---|---|---|---|
| `pool_derivations_n`（每提交） | **2.000**（30 146/30 146 全是 2） | **1.000**（29 652/29 652 全是 1） | 2.00× |
| `views_n`（每提交） | 2.014 | 1.004 | 2.01× |
| `views_bytes`（每提交） | 2 113 989 | 1 039 343 | 2.03× |
| `submit_validate_derive_us`（每提交） | 247.1 | 0.6 | −99.8% |
| `submit_validate_us`（每提交） | 248.0 | 1.8 | −99.3% |
| `total_us`（每提交） | 2 799.5 | 2 581.0 | −218.5 µs（−7.8%） |
| 尾部（top 5%）的 `total_us` | 21 143.3 | 17 417.0 | −3 726.3 µs（−17.6%） |
| 尾部（top 5%）的 `submit_validate_derive_us` | 4 617.7 | 0.9 | −100.0% |
| （对照）`plan_resources_us` | 239.1 | 235.4 | −1.5% |
| （对照）`render_total_us` | 1 607.4 | 1 658.5 | +3.2% |

第一行是这一刀的**机制读数**：它逐提交相等地 2 → 1，与负载无关（池为空的提交也读 2 → 1）。
`plan_resources` / `render_total` 是没被碰的两条 lane，它们只动了 1.5–3.2%（同一台机器上
两臂的宿主每帧本身就差 7%：429.7 vs 401.7 ms），所以省下来的那一份会计在
`submit_validate_derive` 上，而不是被别的 lane 吸收。

## 4. 这一刀不碰什么

* **不碰准入。** `admit_capabilities` 是同一张声明表的**第三**遍走查，但准入发生在
  `submit` 之前、走的是 `ResourceTableSnapshot`，要把它并进来是一条独立契约。
* **不碰像素与设备命令。** 两次走查的产物是同一张只读表；两臂的 141 份 capture 逐字节相同，
  R rail 两臂 203/0。
* **不碰采集与完成语义。** 派生的次数变了，`CompletionToken`、writeback 与 landing 都不变。
* **不翻默认。** 这一刀以关为默认落地；翻档（若要走）由主代理在读数与门都复核之后决定。
