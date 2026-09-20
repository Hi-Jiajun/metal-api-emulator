# 资源池借用：一次提交不再为它读的那张表付两份拷贝

E 侧第七刀（`serial_resources_borrow`，开关
`METAL_API_VULKAN_SUBMIT_RESOURCE_BORROW`，**默认关**）。机制与证明见模块文档
`crates/metal-api-vulkan/src/serial_resources_borrow.rs`；本文记录**为什么是这一处**、
**读数**与**不碰什么**。

## 0. 缺口：同一批声明字节的另外两份拷贝

第六刀（`docs/SUBMIT-BINDING-BORROW.md`）把 seam 拆开之后，§5 的第一名是同一批字节的
**下两份拷贝**。每次提交带的 `ComputeTrace` 声明了约 1.18 MB 的缓冲视图字节
（`BufferSource::OwnedBytes`），而调用为它们付了三份：

| 拷贝 | 在哪里做 | 谁释放 | 状态 |
|---|---|---|---|
| `plan` 的串行资源池 | `ComputeTrace::serial_resources` | `submit_release_views`（尾部） | **本刀** |
| 每个池化绑定一份 | 视图字节的 `Vec::clone` | `submit_release_bindings` | 第六刀已删 |
| 终结校验自己再派生一遍 | `submit_validate` 里的同一次调用 | `submit_validate` 条内那道缝（sp16 读作 43.0 µs/提交） | **本刀** |

两个调用点要的东西是同一张表：一个只读的**派生**——把 trace 的视图声明按首次使用顺序
摊平，并把每个视图的 access 合并成这次提交的并集。它是 trace 的纯函数，所以只**读**它
的调用者（`plan` 排绑定和窗口、`submit_validate` 走 writeback）没有任何理由**拥有**它，
而拥有它就是那 1.18 MB 的克隆。

## 1. 本刀的形状

core 多了一个**借出**的入口 `ComputeTrace::serial_resources_ref`：它返回
`Vec<SerialResource<'_>>`，每个条目是 trace 自己的 `&BufferView` 加上这次提交合并出来的
access（`SerialResource::view` / `::access`）。拥有型入口 `serial_resources` 就是这条派生
映射到 `SerialResource::to_view`——两个入口因此**不可能**对同一个池位置给出不同的身份、
区间、字节或 access。

vulkan 侧：

* `plan` 用借出入口派生池；`pool` 从 `Vec<BufferView>` 变成
  `Vec<SerialResource<'_>>`，Plan 之后的每一段（堆计划、派发表、绑定、渲染半边、writeback
  映射）都按视图读数；
* `submit_validate` 的两张表同样借出，并在 `submit_validate_release` 里显式释放（关臂就是
  那道缝）；
* 完成槽不再搬整张池：延迟臂存的是 **`PoolGeometry`**（视图身份 + allocation + offset），
  也就是一次 readback 真正用来把 landing 解成 writeback 的三个字段——设备的字节在录命令
  之前就已经拷进去了；
* 关臂（默认）就是刀前的路径：`plan` 里一次 `Vec::clone`、`submit_validate` 里再一次，以及
  它们各自的 free。

## 2. 为什么驱动看不出区别

1. **交给驱动的还是同一段字节。** 绑定拿到的切片从前是池表里那份克隆，现在是 trace 自己
   那份声明，指针、长度、顺序都一样；驱动建的 buffer、映射、上传区间与录进去的命令都不读
   宿主 vector 的身份。
2. **借用不可能悬垂。** 借出表借的是 `trace` 自己的声明，而 `trace`（`admitted.trace()`）
   在整次 `submit` 里都由 `ValidatedComputeTrace` 持有；关臂的拥有型表声明在借出表**之前**，
   因此析构在它之后——Rust 两头都管住，第六刀在绑定那一层已经用过同一条论证。
3. **完成槽不再需要池。** 延迟臂的 readback 只用池的**身份与几何**把 landing 解成
   writeback；整张池（连同字节）从槽里消失，`wait` 也不再克隆那 1.18 MB。
4. **别的都没动。** 池的长度、顺序、合并后的 access 和声明字节一字未改；渲染半边的
   `ResolvedOffscreenPass` 仍然持有指向 trace 声明的引用，规划器的视图窗口、绑定表形状与
   writeback 映射都不受影响。

## 3. 开关与读数

| 变量 | 关（默认） | 开 |
|---|---|---|
| `METAL_API_VULKAN_SUBMIT_RESOURCE_BORROW` | 未设 / `0` / `off` / `no` / `false` / 其它 | `1` / `on` / `ON` / `true` / `yes` |

* `plan_resources_us`：`plan` 里那次派生本身（关臂含克隆）；
* `submit_validate_derive_us` / `submit_validate_check_us` / `submit_validate_release_us`：
  `submit_validate` 的三段——两次派生、走查、释放；
* `submit_release_pool_us`：尾部里池表自己的释放（`submit_release_views` 的第二个孩子）；
* `submit_resource_copies_n` / `_bytes`：派生**拷贝**的声明字节（关臂 = 两次派生之和）；
* `submit_resource_borrows_n` / `_bytes`：同一批字节被**借**走的量（开臂 = 两次派生之和）。

## 4. A/B 读数

同一 exe（`sha256 e95795fa…`，三个名字一份字节流）、背靠背、三臂关 / 开 / 关，姿势与
第六刀的 `sp16`–`sp19` 一致（reims `439449f`、300 s、`ONLY=8`、import=on、批开关默认、
give-back census、R 侧 frame profile、E 侧 phase profile、**第六刀自己的开关缺省**——
这一轮量的是第七刀自己）。三臂都 attempt 1 / `alive` / `BOOT_EXIT=0` / 0 panic、覆盖率
100.000 %、红线 `draws_skipped_after_engine_refusal` 0。证据：
`evidence/serial-resources-4317eb6-2026-09-21/`。

### 4.1 机制四条（µs/提交）

| 读数 | `sr1`（关） | `sr2`（开） | `sr3`（关） | 对照臂之差 |
|---|---|---|---|---|
| `total` | 3 142.7 | **2 653.9** | 3 262.3 | 119.6 |
| `plan` | 222.7 | **91.2** | 227.5 | 4.8 |
| ├ `plan_resources` | 142.1 | **1.9** | 140.5 | 1.6 |
| `submit_release` | 167.1 | **108.0** | 171.0 | 3.9 |
| ├ views | 48.3 | **0.19** | 50.1 | 1.8 |
| │ └ pool（内嵌） | 48.2 | **0.07** | 49.9 | 1.7 |
| └ bindings（第六刀，未动） | 90.6 | 80.8 | 90.1 | 0.5 |
| `submit_validate` | 210.1 | **8.5** | 223.2 | 13.1 |
| ├ derive | 169.4 | **7.6** | 177.3 | 7.9 |
| ├ check | 0.70 | 0.65 | 0.82 | 0.1 |
| └ release | 39.6 | **0.06** | 44.7 | 5.1 |
| 命名 seam `submit_seam_us` | 402.3 | **137.9** | 420.1 | 17.8 |

每 pass（`04-sr*-e-phase-table`）：`total` 1 923.3 / **1 646.4** / 1 991.8，
`plan_resources` 87.0 / **1.2** / 85.8，`submit_validate_derive` 103.7 / **4.7** / 108.2。

### 4.2 是不是这一刀

* **四条机制条合计掉 389.6（对 `sr1`）/ 402.7（对 `sr3`）µs/提交** = 对照臂 `total` 的
  **12.4 % / 12.3 %**；而两条**完全相同**的对照臂之间，这四条合计只差 13.1。逐条看：
  `plan_resources` 的效应是它自身臂间差的 ~87 倍、`derive` ~21 倍、`release` ~8 倍、
  pool 的尾部释放 ~29 倍。
* **字节计数器给出机制证据**：关臂每次提交的两次派生拷贝 **2 041 397 B**（`sr1`）/
  **2 415 877 B**（`sr3`）、借用 0；开臂拷贝 **0**、借用 **2 483 719 B**。第六刀的绑定
  拷贝（1.02–1.24 MB/提交）两臂都在、一字未动，正好是这批字节的第三份。
* **端到端的判断（与第六刀不同）**：`total` 掉 488.8 / 608.4（−15.6 % / −18.6 %），
  对照臂自己差 119.6（3.8 %）——效应是它的 4–5 倍；每 pass 上是 −276.9 / −345.4，
  对照差 68.5；R 侧 `prov_submit` 每帧 311.9 / 258.7 / 291.2 ms、宿主每 draw
  4 261 / 3 433 / 4 792 µs、帧间隔 0.559 / 0.468 / 0.538 s（效应 2–4 倍对照差），
  开臂在每一项上都是三臂最低。**结论：这一刀的端到端方向可以读**，但主张仍以机制条为准
  （端到端里还带着控制臂之间 4–7 % 的口径差）。

### 4.3 一处口径修正

这一轮的 exe 把池的那条 bar 当成 release 的第四个孩子，于是它打印的
`submit_release_named_us` 把 `submit_release_pool_us` 算了两遍（215.1 µs，而三个孩子
合计 166.9）。池的释放其实**嵌在** `submit_release_views` 里；tip 的 `88c5f8b` 把集合
改回三个兄弟、把嵌套关系交给测试与文档，**bar、计数器与机制一个字节没变**。上面的表都
按三个孩子自己求和。

## 5. 这一刀不碰什么

* 不动 wire：`CommandRequest`/`CommandCodec`、frame 字节、任何 R 侧类型；
* 不动设备对象：不预建、不池化 buffer/view，创建与销毁次数一字不变；
* 不动 `submit_validate` 的**走查**：它是这条 bar 存在的理由，任何刀都删不掉；
* 不动 `serial_texture_resources`（它不带声明字节，sp16 读作 0.05 µs/提交）；
* 不动第六刀的绑定机制：它自己的开关与计数器照旧，本刀只是让被借的那张表不再克隆；
* 不动 staged lease / gathered guest runs 两个来源：它们各自持有自己的 `Vec`，本刀借不了。
