# 准入走查：一次走路只物化一次

E 侧第十刀（含它的拆条仪器）。开关 `METAL_API_CORE_ADMIT_SHARED_DRAWS`（**默认关**），
拆条开关 `METAL_API_CORE_ADMIT_PROFILE`（**默认关**）。机制与证明写在两个模块文档里
（`crates/metal-api-core/src/admit_shared_draws.rs`、
`crates/metal-api-core/src/admit_profile.rs`）；本文记录**为什么是这一处**、
**两姿态的读数**与**不碰什么**。

## 0. 缺口：`prov_admit_validate` 是生产姿态最大的单项，且没有内部

第六轮剖面（`r_submit_profile6` §5.1）把生产姿态（`import=off`，用户实际跑的那种）的账
钉在 `prov_submit` = **187.9 ms/帧 = 宿主帧的 47.7%** 上，其中
`prov_admit_validate` **18.9 ms/帧**，是同一读数在 census 姿态下的 **5.6×**。
这条 bar 在 reims 侧只包住一次调用（`provider.capabilities().validate_trace`），
在 E 侧只包住同一段代码的第二次运行（`VulkanComputeProvider::submit` 的
`admit_for_submission`）——**两边都只有一个总数，说不出是哪一道门**。

## 1. ① 拆条：一次走路被切成 15 段（`sa1` / `sa1c`，同一个 exe 字节流）

插桩提交 `fcf8910`：`metal-api-core::admit_profile` 按**路由**（`validate` / `submit` /
`direct`）与**区域**记账，关时每个点只付一次 relaxed load。一个 300 s 生产姿态轮
（`sa1`）与一个 300 s census 姿态轮（`sa1c`）用**同一个 exe**
（`94c5ec79…`，两姿态只差 launcher 的几行）：

| 区域（µs/walk） | `sa1` 生产姿态 | `sa1c` census 姿态 | 倍数 |
|---|---|---|---|
| `render_passes` | **48.650** | 6.456 | 7.5× |
| `render_textures` | **48.147** | 5.586 | 8.6× |
| `pixel_samplers` | **47.734** | 5.502 | 8.7× |
| `stage_buffers` | **46.640** | 5.532 | 8.4× |
| `trace_validate` | 24.210 | 3.266 | 7.4× |
| `serial_reuse` | 23.491 | 3.154 | 7.4× |
| `resources` | 21.422 | 1.266 | 16.9× |
| 其余八段合计 | 1.257 | 1.155 | 1.1× |
| **`total`** | **262.733** | **32.858** | **8.0×** |

四条渲染门（`render_passes` / `render_textures` / `pixel_samplers` / `stage_buffers`）
合计 **191.2 µs = 走路的 72.8%**，而它们每一个的**门身**都只是几个 `Vec` 扫描
（`admit_render_pixel_samplers` 是一个 1.4 趟 × 0.9 个 sampler 的双重循环）。
四条彼此只差 4%，形状完全不像"四条不同的检查"——像**同一件工作被做了四遍**。

### 1.1 为什么生产姿态是 census 姿态的 5.6×：同样的声明数，不同的字节数

两姿态的**形状普查**几乎一样（每走路的均值）：

| 普查 | `sa1` 生产 | `sa1c` census | 倍数 |
|---|---|---|---|
| `render_passes_n` / `draws_n` | 1.65 / 1.99 | 1.40 / 1.56 | 1.18 / 1.27 |
| `stage_buffers_n` | 4.56 | 3.77 | 1.21 |
| `textures_n` / `samplers_n` | 1.28 / 1.14 | 1.11 / 0.94 | 1.16 / 1.22 |
| `vertex_buffers_n` | 1.96 | 1.54 | 1.27 |
| `refused_n` | 0.031 | 0.022 | — |

**形状只差 1.2–1.3×，时间差 8×**。差异被拆成两个因子，第三个候选被否证：

* **① 多 draw 列表的条数**：生产姿态 **0.584 次物化/走路**，census 姿态（`sa2d`）
  **0.303**——**1.93×**；
* **② 每次物化的单价**：两姿态都用本刀的"一次物化"直读（`draw_passes_us` ÷ 物化次数，
  同一个 exe）：生产 `sa2b` **252 µs/次**，census `sa3d` **58 µs/次**——**4.3×**，
  而"每次物化搬几个 pass"只差 1.15×（2.80 vs 2.44）。
* **③ "字节多寡"被否证**：字节普查（`texture_owned_bytes_n` 等，`6bd95b1`）读出
  census 姿态的纹理真字节**更多**（757 KB/walk vs 生产 461 KB/walk），
  而两姿态的 buffer 侧都只是租约（生产 1 005 B、census 0.035 B）。

顺带两条否证：拒绝/重试不是原因（`refused_n` 两姿态都是 2–3%，且 `submit` 路由 **0**）；
`mint` 也不是（生产姿态的 `allocations_n` 11.2 vs census 9.1，只差 1.2×）。
剩下那 4.3× 只能落在**被克隆字节的内存状态**上（生产姿态的声明是每提交现铸的分配，
census 姿态的声明大半是对来宾窗口的引用——0.84/1.10 的纹理是 `BorrowedNoCopy`）；
要把它钉死需要在克隆点插仪，本轮没做，留作下一件仪器活。

## 2. ② 这一刀：一次走路只物化一次

四条渲染门都走 `ComputeTrace::render_draw_passes()`。这个迭代器对单 draw 条目是**借用**，
但对多 draw 列表（B-2 的 `TracePass::RenderDraws`）必须**物化**：
`RenderDrawsDescriptor::materialize` 克隆 pass 状态，并按 draw 克隆它自己的声明
（顶点流、采样纹理、运行时 sampler、stage buffer）——**每一条门各做一遍**。

这一刀（`METAL_API_CORE_ADMIT_SHARED_DRAWS`，默认关）在第一条门之前物化一次，
四条门走同一份值：

| | 关（刀前） | 开（这一刀） |
|---|---|---|
| 物化次数 / 走路 | 每遇一条列表就物化一次 × 4 条门 | 1 次 |
| 四条门的 bar | 各 ~48 µs | 只剩门身（几 µs） |
| 新 bar `draw_passes_us` | 0（不物化） | 那一次物化 |

### 2.1 为什么出的字节一定相同

1. `materialize` 是列表的纯函数（`head.clone()` + 每个 tail draw 的 `with_draw`），
   同一个列表两次调用得到相等的值；
2. 四条门只**读**交到手上的 pass（`&RenderPassDescriptor`），走路本身不修改物化值，
   也不修改它所来自的 trace（`&ComputeTrace` outlives 全部借用）；
3. 门的顺序、条目顺序、每条门按名拒绝的 slug 全部不变：同样的值以同样的顺序到达同样的检查。

rail 证据：`the_two_arms_of_the_render_entry_walk_answer_the_same`
在**同一个进程**里用 `admit_shared_draws::set_arm` 把同一个列表 fixture 跑两遍，
断言两臂走到的值逐条相等、两臂的准入都通过、两条按名拒绝（`render_multi_draw_unsupported` /
`render_draw_count_limit`）在两臂同名。

## 3. ③ 读数（A/B）

同一个 exe 字节流 `d29f6b87…`、生产姿态、背靠背 300 s ×3（`sa2` 关 / `sa2b` 开 /
`sa2c` 关），三臂都 `attempt=1`、`alive`、`BOOT_EXIT=0`：

| 读数 | `sa2`（关） | **`sa2b`（开）** | `sa2c`（关） |
|---|---|---|---|
| `total` µs/walk（`validate`） | 311.36 | **123.05** | 266.37 |
| 四条门合计 µs/walk | 223.17 | **1.92** | 193.19 |
| `draw_passes_us` | 0.000 | **39.00** | 0.000 |
| `draw_list_materialize_n` /walk | 0.584 | **0.155** | 0.584 |
| `draw_list_materialize_passes` /walk | 1.653 | **0.433** | 1.653 |
| `PHASE admit_us` µs/提交 | 334.0 | **115.7** | 287.9 |
| 宿主 ms/帧 | 406.9 | 390.2 | 358.1 |
| 覆盖率（批口径） | 98.56% | 98.66% | 98.56% |
| 五个零读 | 0 ×5 | 0 ×5 | 0 ×5 |

**机制决定性**：物化 4 → 1（两条计数都落在 4×，而两条完全相同的对照臂逐字相同）、
四条门塌掉 99.1%、那一次物化被 `draw_passes_us` 具名、走路总时间 −60.5%（对照臂之间
只差 16.9%）。**端到端解不出来**：三条臂的宿主每帧 406.9 / 390.2 / 358.1 ms，
增量臂落在两条对照之间，而对照彼此差 48.8 ms（13.6%）——按项目口径不宣称端到端数字。

两姿态的物化次数：生产 0.584 次/走路 vs census（`sa2d`）0.303 次/走路，
而单位成本差 5.0×（382 vs 76 µs/次），两者相乘 = 9.7× ≈ 两姿态四条门实测比（223.2/23.1）。

同一条物化的**直接**读数（cut 开的那一臂，`draw_passes_us` ÷ 物化次数，同一个 exe）：

| | 生产 `sa2b` | census `sa3d` |
|---|---|---|
| `draw_passes_us` /walk | 39.00 | 4.89 |
| 物化次数 /walk | 0.155 | 0.085 |
| **每次物化** | **252 µs** | **58 µs** |
| 每次物化的 pass 数 | 2.80 | 2.44 |
| 该臂四条门合计 | 1.92 µs | 1.73 µs |

⇒ 4.3× 是**同一条代码**在两姿态下的真实差价（不是平均值的口径差）。

## 4. 不碰什么

* **不动物化本身**：多 draw 列表仍然按"每个 draw 一个单 draw pass"被物化——那是 B-2
  的设计（每条规则只写一遍），本刀只把**次数**从四条门各一次降到一次。
* **不动契约**：`RenderDrawSource` / `DrawPasses` 是 crate 私有的走路内部类型，
  公共 API、能力位、线格式一律未动。
* **不动其它 `render_draw_passes()` 调用点**：`serial_resources` 的三处、以及
  `phase_profile` 的既有条，都不在准入走路里，本刀与它们无关
  （仪器的物化计数只在一次走路窗口打开时记账，所以那些调用点不会被误记）。
