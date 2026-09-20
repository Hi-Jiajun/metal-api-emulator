# 绑定借用：一次提交不再为它上传的字节再拷一份

E 侧第六刀（`submit_binding_borrow`，开关 `METAL_API_VULKAN_SUBMIT_BINDING_BORROW`，
**默认关**）。机制与证明见模块文档 `crates/metal-api-vulkan/src/submit_binding_borrow.rs`；
本文记录**为什么是这一处**、**读数**与**不碰什么**。

## 0. 缺口：seam 残差里最大的一条无名区域

第五刀把 E 表读成 `total − 不相交条之和 = 1 317.7 µs/提交（41.5 %）`，并注明这块"没有名字"。
第六刀的插桩轮 `sp16` 把这块切成四段：`submit_release`（调用自身的尾部释放）、
`submit_lock` / `submit_bookkeep` / `submit_merge`、`submit_validate` 的两半。切完之后
`total` 里**没有名字**的部分只剩 **2.7 µs/提交（0.08 %）** —— 也就是说，残差不是"找不到的
代码"，而是三件各自可命名的事：

| `sp16`（reims `439449f` × provider `3b558ee`，300 s、`ONLY=8`、import=on） | µs/提交 | 占 `total` |
|---|---|---|
| `total` | 3 215.4 | 100 % |
| ├ 不相交条 | 913.2 | 28.4 % |
| ├ `render_total`（含五个子条与 13 条残差子条） | 1 882.2 | 58.5 % |
| └ 命名 seam `submit_seam_us` | 417.4 | 13.0 % |
| &nbsp;&nbsp;&nbsp;├ `submit_release` | **164.8** | 5.13 % |
| &nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;├ bindings（视图字节） | **88.8** | 2.76 % |
| &nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;├ views（资源池 + 纹理视图） | 47.6 | 1.48 % |
| &nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;└ plan（派发表 + heap + render plan + artifacts） | 28.2 | 0.88 % |
| &nbsp;&nbsp;&nbsp;├ `submit_validate` | 229.1 | 7.12 % |
| &nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;├ derive（两次池派生） | 185.4 | 5.77 % |
| &nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;└ check（writeback 走查） | 0.6 | 0.02 % |
| &nbsp;&nbsp;&nbsp;├ `submit_teardown` / `lock` / `bookkeep` / `merge` | 19.1 / 0.6 / 3.0 / 0.8 | 0.7 % |
| **无名残差（`total` − 全部条）** | **2.7** | **0.08 %** |

`submit_release` 是**调用自己的尾部**：`total` 是 `submit` 里第一个绑定，因此最后析构，
而它之后声明的那些值——绑定表（每个视图一份自己的字节）、资源池、纹理视图、派发表、
heap 计划、render 计划、pipeline artifacts——都在最后声明的 `settle` 守卫**析构之后**才
析构。它们落在 `total` 之内、任何别的条之外，这就是第五刀看到的那块无名区。

它的三个子条按释放的族划分，最大的一条是 **bindings（88.8 µs/提交）**：每个池化绑定都
携带它要上传的视图字节的一份**自己的拷贝**。拷贝的代价付了两次——`pool` 里的
`Vec::clone` 一次、释放时的 `free` 一次。

## 1. 为什么只有"轨迹自带的那一份"能借

一个绑定的字节有三个来源，只有第一个是**这次提交为自己做的拷贝**：

| 来源 | 绑定持有的东西 | 能不能借 |
|---|---|---|
| 轨迹的快照（`BufferSource::OwnedBytes`） | 视图字节的 `Vec::clone` | **能**：serial resource pool 整次调用都持有同一份字节 |
| staged lease（`BufferSource::StagedLease`） | 从 staging registry 的锁里拷出来的新 `Vec` | 不能：registry 的字节在互斥锁后面，上传期间不持锁 |
| gathered guest runs（`BufferSource::GuestRuns`） | gather 刚构造出来的新 `Vec` | 不能：gather 就是它的生产者 |

机制只做第一行：绑定拿**已经持有的那张表的借用**，而不是第二份拷贝。

## 2. 为什么驱动看不出区别

1. **交给驱动的还是同一段字节。** `BindingBytes::Borrowed(bytes)` 交给 `create_buffers`
   的切片，与 `BindingBytes::Copied(bytes.clone())` 交给它的是同一段：上传时的指针、长度、
   顺序都相同。驱动创建的 buffer、映射、上传区间和录制的命令都不读宿主 vector 的身份 ——
   上传就是一次 `copy_nonoverlapping`。
2. **借用不可能悬垂。** 借的是 serial resource pool 自己的视图源；pool 是同一次 `submit`
   的局部值，声明在绑定之前、析构在绑定之后，绑定只按引用传递、从不被存起来。借用检查器
   两头都管住：延迟臂必须先把绑定丢掉，才能把 pool 移进完成槽——所以那处 `drop(buffers)`
   是写出来的。
3. **别的都没动。** 另外两个来源仍持有自己的 `Vec`，所以绑定表的形状、宽度、规划器的
   视图窗口和 writeback 映射一字未改；`validate_with_pools` 走的仍是同一条契约校验。

关臂（默认）就是刀前的路径：同样的 `Vec::clone`、同一份字节、同样的释放。

## 3. 开关与读数

| 变量 | 关（默认） | 开 |
|---|---|---|
| `METAL_API_VULKAN_SUBMIT_BINDING_BORROW` | 未设 / `0` / `off` / `no` / `false` / 其它 | `1` / `on` / `ON` / `true` / `yes` |

两臂的读数由同一行相位表给出：

* `submit_binding_copies_n` / `submit_binding_copies_bytes`：这次提交**自己拷贝**的轨迹视图
  字节（关臂非零、开臂应读到 0）；
* `submit_binding_borrows_n` / `submit_binding_borrows_bytes`：机制**借**走的同一批字节
  （关臂 0、开臂应等于关臂的拷贝量）；
* `submit_release_bindings_us`：绑定释放那条 bar —— 借来的字节不再需要 `free`；
* `pool_us`：拷贝所在的那条不相交 bar。

## 4. A/B 读数

同一 exe（`sha256 46e6a4c6…`，三个名字一份字节流）、背靠背、三臂关 / 开 / 关，
姿势与 `sp16` 一致（reims `439449f`、300 s、`ONLY=8`、import=on、批开关默认、
give-back census、R 侧 frame profile、E 侧 phase profile），三臂都 alive、attempt 1、
0 panic、`BOOT_EXIT=0`、覆盖率 100.000 %、红线 0。

### 4.1 机制自己的条（µs/提交）

| 读数 | `sp17`（关） | `sp18`（开） | `sp19`（关） |
|---|---|---|---|
| `total` | 3 213.9 | **2 891.0** | 3 015.9 |
| `pool` | 290.9 | **116.6** | 231.3 |
| `submit_release` | 170.4 | **111.1** | 149.7 |
| &nbsp;&nbsp;└ bindings | 91.1 | **38.8** | 80.0 |
| `submit_binding_copies_n` / `_bytes` | 0.896 / **1 184 054 B** | 0 / 0 | 0.900 / 1 129 444 B |
| `submit_binding_borrows_n` / `_bytes` | 0 / 0 | **0.897 / 1 173 322 B** | 0 / 0 |

也就是说：**每次提交有 1.13–1.18 MB 是这次提交为自己拷的**，开臂把它们全部变成借用
（字节数与关臂同量级，差 0.9 %，来自两臂自身的提交/形状分布差异）。

### 4.2 判读

* **机制条能解出这一刀。** `pool` 掉了 174.3（对 `sp17`）/ 114.7（对 `sp19`），
  而它自己在两条**完全相同**的对照臂之间的差只有 59.6；`submit_release_bindings` 掉了
  52.3 / 41.3，对照臂之间的差只有 11.1。两个条上的效应都是自身臂间差的 2–3 倍，
  方向一致，且插入的字节计数器给出机制证据：拷贝 1.18 MB → 0、借用 0 → 1.17 MB。
  按这两条合计，这一刀值 **156–227 µs/提交（提交的 5–7 %）**。
* **端到端读数解不出一个更小的区间。** `total` 在 `sp17→sp18` 掉 10.0 %、在 `sp19→sp18`
  掉 4.1 %，而两条对照臂自己就差 6.2 %；R 侧同样：`prov_submit` 每帧 294.2/280.1/312.5 ms、
  宿主每 draw 4 151/4 063/4 430 µs、帧间隔 0.541/0.533/0.573 s —— 增量臂在**每一项**上都是
  三条里最低，但两条对照臂之间的差与效应同量级。**如实报告：机制条可用，端到端只作方向性
  佐证**（与第五刀同一形态的诚实结论，区别是这次效应比对照臂间差更大而不是更小）。
* `submit_release_views` / `_plan` 两臂几乎不动（49.2/45.0/43.7 与 29.9/27.0/25.8）——
  这一刀只动"轨迹自带那份字节"，其余两个来源仍各自持有自己的 `Vec`，与设计一致。

## 5. 这一刀不碰什么

* 不动 wire：`CommandRequest`/`CommandCodec`、frame 字节、任何 R 侧类型；
* 不动设备对象：不预建、不池化 buffer/view，创建与销毁次数一字不变；
* 不动 staged/gathered 两个来源（它们仍各自持有一份新 `Vec`），也不动
  `submit_release_views` / `submit_release_plan` 两条子条；
* 不动 `submit_validate` 的两半：`docs/COMPUTE-PIPELINE-REUSE.md` §6 已用分布证据
  量出"可删的重复派生"只有 ≈4.9 µs/提交（0.2 %），并据此否决了那条扩宽契约的入口；
  本刀只**读**这个拆分，不落那条机制。

> **后续（第七刀）**：§5 的第一名已经落刀 —— `ComputeTrace::serial_resources_ref` 让 `plan`
> 与 `submit_validate` **借**这张池表而不是各拷一份，见
> `docs/SUBMIT-RESOURCE-BORROW.md` 与开关 `METAL_API_VULKAN_SUBMIT_RESOURCE_BORROW`。
> 本刀（第六刀）的绑定借用与它的开关/计数器一字未改；两把开关各自独立，默认都关。
