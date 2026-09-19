# 交接：E 仓移交给新会话（2026-09-18 15:35，根会话停止点）

用户要求停止根会话的 subagent、并把 **metal-api-emulator 相关开发移交新会话**。
本文件由根会话写下，供新会话接力。

## 1. 停止点（就是现在）

| 项 | 状态 |
|---|---|
| 根会话的 subagent | **全部已中断**（`census_v23`、`r31_class_set0`） |
| Windows QEMU 进程 | **0 个**（census v23 在发射前被中断，没有留下 VM） |
| census v23 证据 | `evidence/gate3-census-v23-2026-09-18/` 只有 15 个文件（freeze/build/relink/smoke），**没有 boot、没有 README** |
| R31（reims class 层接受 set 0 共存） | **未开工、无改动**（reims worktree 干净） |

## 2. 三仓当前坐标（都已 push）

| 对象 | 位置 / 状态 |
|---|---|
| `metal-api-emulator` | main **`049bf74`**（E-TX8 `a6bb83a` 已合并），CI run 成功 |
| `reims-vgpu` fork 分支 `guest-wiring-step5b` | **`5597b83`**（E-TX8 的 reims 随动 `c85948b` 已合并；rail 97/97） |
| `research` | **`444e9fb`**（docs/23 §115 + docs/09 §115） |
| Windows 生产 exe | QEMU **11.1.1** `752c3502…`（备份 `reims-vgpu-backup-exe-2026-09-18\`） |

## 3. 建议的接力顺序（主线还没完）

1. **R31**（reims 侧，如果也归新会话）：让 class 层把"纹理 + stage buffer 共用 set 0"的
   形状交给 provider——census v22 里 `render_texture_layout_unsupported` 有 **1,234 行**
   全部是 reims `292dcde` 的按名出口，E-TX7（`115f01f`）已经把 provider 侧打开。
   任务书（未执行）在根工作区：`/home/hiliang/hackintosh/.agents/tasks/root/r31_class_set0.md`。
2. 跑一轮 census 验收（快 harness：`/home/hiliang/hackintosh/tools/census/`，驻留 210 s +
   cargo target 缓存，一轮 ~7 分钟）。三条件：`render_texture_layout_unsupported` ≈0、
   四个零读书（`chain_resident_land_fail` / `load_target_content_not_ready` /
   `draws_skipped_after_engine_refusal` / `vk_engine_target_read`）保持 0、
   raw `ok pipe` 不掉且抓图正常（登录窗按钮/密码框、桌面）。
3. 剩余深尾（v22 数字）：`shape` ~482、`load_seed` ~274、`texture_extent` ~248、
   `resident_source` ~198、`texture_sampler_wire` ~194、`texture_bind` ~189、
   `texture_source_order` ~173、`guest_backing` ~171。
4. 最终验收：Gate 2/3 全量复跑 + 五轨一致性 + 新面在 **RTX 5060** 与 **Apple CI** 的读数。

## 4. 工具（本会话刚给本仓配好，务必不要推上 GitHub）

- **CodeGraph**：本仓已建索引（138 文件 / 42 MB）。优先
  `codegraph explore "<符号或问题>"`（也可用 MCP `codegraph_explore`，带 `projectPath`），
  大合并后 `codegraph sync .`。fork 仓 `repos/fork-reims-vgpu` 也建了（480 文件 / 87 MB）。
- **OpenSpec**：本仓已 `openspec init --tools codex --language zh-CN`
  （`openspec/` + `.agents/skills/openspec-*`）。按折中口径：只有**契约级 / 跨模块大改动**
  走 proposal（`openspec change new <id>` → 用户审 → 实现 → `openspec archive`），
  日常加宽/修复仍走任务书 + census。
- 三者都写在 `.git/info/exclude`：`.codegraph/`、`openspec/`、`.agents/`、`worktrees/`。
  **永不进 GitHub**（fork 仓同理只排除 `.codegraph/`）。

## 5. 红线（一直有效）

- `research/docs/11-Metal-provider-contract-v0.md` 是用户 dirty 草稿，**不提交、不覆盖**。
- `repos/fork-reims-vgpu` 主工作树有用户 dirty（`vendor/qemu` gitlink、`vm/boot-windows.sh`、
  未跟踪脚本），不要动。
- 生产树 `C:/hackintosh/reims-vgpu` 零写入（真机轮前后三件套 sha 一致 + `find -newer` 空）。
- 提交用 DCO：`git -c user.name='Jiajun Liang' -c user.email='3138947285@qq.com' commit -s`。

## 6. 关键读物

- 根工作区 `FINAL-REVIEW-2026-09-18.md`：完整门演化表（v1→v22）、每轮读数、未完成清单、§6 待用户确认事项。
- `evidence/gate3-census-{v16..v22}-2026-09-18/`：最近几轮的结论与原始日志。
- `tools/census/README.md`：快 harness 的用法（含 `--print` 干跑、target 缓存、210 s 驻留）。
