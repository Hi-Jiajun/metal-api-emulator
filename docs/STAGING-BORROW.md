# staging 借用：同一批 owner 字节的第三份拷贝不再发生

E 侧第八刀（`staging_borrow`，开关 `METAL_API_VULKAN_STAGING_BORROW`，**默认关**）。
机制与证明见模块文档 `crates/metal-api-vulkan/src/staging_borrow.rs`；本文记录
**为什么是这一处**、**为什么不能借引用而能借句柄**、读数与不碰什么。

## 0. 缺口：同一批 owner 字节的第三份拷贝

第六刀（`docs/SUBMIT-BINDING-BORROW.md`）与第七刀
（`docs/SUBMIT-RESOURCE-BORROW.md`）收掉的是同一条链上的前两份；两份报告 §5 的
下一刀排序里都写着第三份：owner 把 staged lease 导入 provider 的 staging registry
（`LeaseRegistry`）之后，**registry 自己在锁后面把窗口又拷了一份**给绑定
（`LeaseRegistry::view_bytes` / `::texture_bytes` 里的 `to_vec`）。

第八刀的测量轮 `sp23`（provider `d7572e8` = 本刀插桩提交 × reims `660d89be`，
300 s 驻留、`ONLY=8`、`import=on`、批开关默认、give-back census、R 侧 frame profile、
E 侧 phase profile、第七刀的 `REIMS_VGPU_STAGED_BYTES_OWNED=on`）把它量到底：

| 读法 | 值 |
|---|---|
| `staging_window_us` | **69.544 µs/提交**（52.406 µs/pass） |
| 占 provider 每提交 `total_us`（2895.807） | **2.4 %** |
| `staging_window_copies_n` | **6.639 次/提交**（5.003 次/pass） |
| `staging_window_copies_bytes` | **0.270 MB/提交** → **10.02 GB/轮** |
| 每次拷贝平均 | 40.7 KB（与 R 侧 `render_provider_unaligned_window_staged` 的 40.3 KB/次 一致） |

交叉印证：R-LB1 报告 §1 从 owner 侧估的同一批是 **0.29 MB/提交 = 11–13 GB/轮**
（sp20 口径），这一轮从 provider 侧量到 **0.270 MB/提交 = 10.0 GB/轮**——两侧读数
同一量级、同一次提交都落在 6–7 个窗口上。三次拷贝的链条到此完整：

1. 类门/窗口拷贝（R 侧自己造的 `Vec`）——第七刀已改为**移入** lease；
2. 轨迹快照字节（第六刀的 `submit_binding_borrow`）——已改为借用池表；
3. **staging registry 的窗口拷贝**——本刀。

## 1. 本刀的形状

core 侧（`crates/metal-api-core/src/provider.rs`）：

* `LeaseRegistry` 的映射从 `BTreeMap<LeaseId, StagedLease>` 变成
  `BTreeMap<LeaseId, Arc<StagedLease>>`。`import(staged: StagedLease)` 的签名、语义和
  `StagedLease` 类型本身**未动**（owner 侧源码一字不改即可编译），`Arc::new` 只把
  `Vec` 的头搬进 Arc 块，字节不动。
* 新增 `LeaseWindowBytes`：窗口 + 对**同一份导入**的句柄，`as_slice()` / `len()` /
  `to_vec()` / `lease_id()` / `reservation()` 齐全。
* 新增 `view_window` / `texture_window`：与本刀前的 `view_bytes` / `texture_bytes`
  走**同一个** `window_bytes`（同一顺序、同一名字的身份/快照/epoch/边界检查），只是
  不再 `to_vec`。旧的两个入口保留并实现为 `view_window(..)?.to_vec()`——**关臂逐字节
  就是刀前路径**。

vulkan 侧（`crates/metal-api-vulkan/src/staging_borrow.rs`）：

* `METAL_API_VULKAN_STAGING_BORROW` 开关（默认关）决定
  `BindingBytes::{Copied, Staged}` 走哪一臂；
* 两个 rail 的三处解析点（compute 的 `pool`、render 的输入解析与纹理解析）都改走
  `resolve_view` / `resolve_texture`，两臂共用同一段检查代码；
* `RenderInputSource::StagedBytes` 的载荷从 `Vec<u8>` 变成 `BindingBytes<'_>`。

## 2. 为什么驱动看不出区别，以及为什么不能借引用

1. **交给驱动的是同一段字节。** 上传读的是 `as_slice()`：关臂是 registry 的 `to_vec()`
   结果，开臂是 registry 自己那份导入里的同一区间。指针（在同一臂内）、长度、顺序、
   内容都相同——单测把三件事都断言了（见 §4 的 `core/vulkan` 列）。
2. **句柄不会悬垂，也不会提前释放。** 句柄持有对导入的引用计数，因此字节与读取它的
   绑定同寿命；这正是拷贝原本给的寿命（拷贝只是**更长**地持有它自己那份），所以
   owner 在提交完成路径上 `release_staged_lease` 之后，绑定仍读得到它上传时那份字节。
3. **借 `&[u8]` 在这里做不到，理由不是"活得更久"而是死锁。** registry 的字节在它自己的
   `Mutex` 后面，借引用就要把 guard 一起借出去，而：
   * 同一次提交解析**不止一个** staged 窗口（`sp23` 读 6.639 次/提交），第二次解析会去
     拿第一次还握着的锁——`std::sync::Mutex` 不可重入；
   * owner 的 `settle` 在提交完成路径上调 `release_staged_lease` → `LeaseRegistry::release`，
     拿的是同一把锁；
   * 别条线程的 `import` / `release` 会被一次提交的设备工作时间挡住，而不是被一次查表挡住。
   所以本刀借的是**字节本身**（句柄）而不是引用的借用：锁只覆盖检查，句柄在锁外逃逸。
4. **别的都没动。** 检查、拒绝名与顺序是 registry 自己的；绑定表形状、`len()`、规划器的
   视图窗口、writeback 映射与录制都不受影响；`RenderInputRetains` 对 staged 臂仍然
   不取 owner 保留（它读的不是 owner 的映射）。

## 3. 开关与读数

| 变量 | 关（默认） | 开 |
|---|---|---|
| `METAL_API_VULKAN_STAGING_BORROW` | 未设 / `0` / `off` / `no` / `false` / 其它 | `1` / `on` / `ON` / `true` / `yes` |

* `staging_window_us`：解析一个 staged 窗口所在的**嵌套**区段（compute 半边在 `pool` 里，
  render 半边在它自己的输入解析里），两臂都计入；
* `staging_window_copies_n` / `_bytes`：registry 为绑定**拷**了多少个窗口、多少字节（关臂）；
* `staging_window_shares_n` / `_bytes`：同一批窗口被**按句柄借出**了多少次、多少字节（开臂）。

## 4. A/B 读数

三臂 `sb1` / `sb2` / `sb3` = **同一个 exe 字节流**（`sha256 10d6b6ae…`，三个名字）、
背靠背、300 s×3、臂间 settle = "QEMU 消失后连续三次读数 + 10 s"、无并行重活；姿势与
`sp23` 相同。三臂都 attempt 1 / `verdict=alive` / 0 panic / `BOOT_EXIT=0`、覆盖率
**100.000 % ×3**、红线 0 ×3、五个零读 0 ×3。证据
`evidence/staging-borrow-a99b842-2026-09-21/`（本例 exe 冻结的是 `a99b842`；`0fe97b1`
是它之后的 clippy 站点修正，借用写法改动、无机制字节差异）。

| 读法（每提交） | `sb1` 对照 | `sb2` 增量 | `sb3` 对照 | 判读 |
|---|---|---|---|---|
| `staging_window_us` | 66.231 | **0.949** | 62.093 | −98.6 % / −98.5 %；两对照只差 6.7 % |
| 占 `total_us` | 2.294 % | **0.036 %** | 2.379 % | 同上 |
| `staging_window_copies_n` | 6.517 | **0** | 6.476 | 拷贝条清零 |
| `staging_window_copies_bytes` | 0.263 MB | **0** | 0.265 MB | 同上 |
| `staging_window_shares_n` | 0 | **6.557** | 0 | 同一批窗口按句柄借出 |
| `staging_window_shares_bytes` | 0 | **0.263 MB** | 0 | 同一批字节 |
| 整轮（拷贝 / 借出） | 9.289 GB / 0 | **0 / 10.573 GB** | 11.114 GB / 0 | 绝对值随访客工作量漂，按每提交读 |

包住这次解析的 `render_prepare_us` 是 **210.187 → 135.118 → 197.398**（−75.1 / −62.3），
与机制条自己的落差（−65.3 / −61.1）同幅同向；compute 半边的 `pool_us` 三臂不动
（137.854 / 140.107 / 122.279），说明本姿势的流量几乎全在渲染半边。

**端到端这一轮解不出来**：`total_us` 是 2886.674 / 2624.591 / 2610.375 µs/提交，而两条
对照臂自己就差 9.5 %（访客提交数 35 328 / 40 192 / 41 984，差 12–17 %），增量臂落在对照
带内 ⇒ 只有机制条（65×，对照带宽 6.7 %）作判据。

等价性三件套：core 单测
`a_staged_window_handle_reads_the_same_bytes_as_the_copy`（两臂同字节、同拒绝名、句柄在
`release` 之后仍可读）；vulkan 单测
`the_two_arms_hand_over_the_same_bytes_from_different_allocations`（两臂同长度同内容，
句柄臂两次指针相同 / 拷贝臂两次指针不同）；两臂 141 份 capture sha256 集合相同 +
48 个集成二进制两臂全绿。

## 5. 不碰什么

* 不动 wire、不动 R 仓源码：owner 侧只读，且它的源码与本刀前**逐字节相同**地编译通过
  （本轮 freeze 的 `03-staticlib-build.log` 就是这条证据）。
* 不动 `BorrowedNoCopy`（no-copy 通道）、不动 `GuestRuns` 的 gather 拷贝——后者是**生产者
  自己造**的字节，没有更早的持有者，§0 的链上不属于同一位置。
* 不改完成路径的保留/释放语义：`plan.settle` 仍按原来的时机释放导入；本刀只是让绑定
  多持有一个引用计数，导入被释放后字节仍活到绑定析构为止。
