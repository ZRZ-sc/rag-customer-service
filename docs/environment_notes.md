# 环境事故记录：C 盘写满导致 WSL 根文件系统只读

> 记录时间：2026-09-29 ~ 09-30 ｜ 影响范围：本机开发环境（Docker 镜像构建、PostgreSQL 写入）
> 结论：应用与 Dockerfile 均无缺陷；根因是 **Windows C 盘剩余空间归零**，WSL2 的虚拟磁盘位于 C 盘，
> 磁盘写满后 ext4 根文件系统被内核置为只读（`emergency_ro`），导致所有依赖容器写入的操作失败。

## 1. 现象

| 表现 | 证据 |
|---|---|
| Docker 镜像构建在最后一层失败 | `error committing ...: write /var/lib/docker/buildkit/containerd-overlayfs/metadata_v2.db: read-only file system` |
| `pip` 装完包后大量告警 | `WARNING: Failed to remove contents in a temporary directory '/tmp/pip-...'` |
| 问答日志/工单/反馈写库失败 | `dialogue log write failed: could not open file "base/16384/2601": Read-only file system` |
| 限流降级 | `rate limit check failed (fail-open): MISCONF Errors writing to the AOF file: Read-only file system` |
| 管理后台日志接口 500 | `asyncpg.exceptions.InsufficientPrivilegeError: could not open file "base/16384/2601": Read-only file system` |
| WSL 内任何写入失败 | `touch /root/x` → `Read-only file system`；`mount` 显示 `(rw,...,emergency_ro)` |

值得注意：**读操作仍然正常**（L1 缓存命中、Milvus 检索、RAG 生成都成功），因此表面看"服务是好的"，
只有写入路径（PG 落库、Redis AOF、镜像层提交）在静默失败——这正是本项目各项降级/兜底逻辑生效的地方：
对话仍能返回答案，同时日志里留下明确告警，而不是整站不可用。

## 2. 根因分析

```
C 盘（98.9GB）剩余 0GB
   └─ Ubuntu WSL 虚拟磁盘 ext4.vhdx（7.71GB）位于
      C:\Users\<user>\AppData\Local\wsl\{GUID}\
        └─ 磁盘写满 → ext4 触发错误保护 → 根文件系统 remount read-only（emergency_ro）
             ├─ /var/lib/docker 不可写 → buildkit 无法提交镜像层
             ├─ PostgreSQL 数据文件不可写 → INSERT/UPDATE 报 EROFS
             └─ Redis AOF 重写失败 → 限流/缓存写入报 MISCONF
```

排查要点（可直接复用的命令）：

```powershell
Get-Volume | Where-Object DriveLetter | Select-Object DriveLetter, SizeRemaining   # 看 C 盘是否写满
wsl -d Ubuntu -u root -e bash -c "mount | grep ' / '"                               # 看是否出现 emergency_ro
wsl -d Ubuntu -u root -e bash -c "touch /root/x && echo ok"                         # 写测试
Get-ChildItem "HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss" | ForEach-Object { Get-ItemProperty $_.PSPath }  # 发行版磁盘位置
```

## 3. 处置过程

1. **腾空间前的干扰项**：`wsl --shutdown` 后发行版立刻又被拉起，`wsl --manage --move` 报
   `WSL_E_DISTRO_NOT_STOPPED`。原因是两个常驻进程：
   - Docker Desktop（`Docker Desktop.exe` / `com.docker.backend`）会维持 WSL 集成发行版运行；
   - `scripts\infra_up.ps1 -Keepalive` 的保活窗口持有 `wsl -d Ubuntu ... "while true; do sleep 20; done"` 会话。
2. 退出 Docker Desktop 与保活窗口 → `wsl --shutdown` → 确认 `Ubuntu = Stopped`。
3. **迁移发行版磁盘到 D 盘**（原生指令，秒级完成，不复制数据、不丢数据）：

   ```powershell
   wsl --manage Ubuntu --move D:\WSL\Ubuntu
   ```

4. 重启后校验：`mount` 不再有 `emergency_ro`、写测试通过、`docker ps` 显示 7 个容器
   （`restart: unless-stopped`）自动恢复 healthy。

### 效果

| 指标 | 迁移前 | 迁移后 |
|---|---|---|
| C 盘可用 | ≈ 0 GB | **10.68 GB** |
| D 盘可用 | 86.5 GB | 78.78 GB（`D:\WSL\Ubuntu\ext4.vhdx` 7.76GB） |
| 根文件系统 | `rw,...,emergency_ro`（只读） | `rw,relatime,...,errors=remount-ro`（可写） |
| 写库验证 | `dialogue log write failed ... Read-only file system` | 新日志行 `id=157724` 正常落库 |
| 镜像构建 | 最后一层提交失败 | 见 `docs/production_plan.md` 同批次构建结论 |

## 4. 复现与预防建议

- **保持 C 盘余量 ≥ 10GB**：WSL2 的 `ext4.vhdx`、Docker 层、构建缓存都会写系统盘。
- **发行版磁盘放数据盘**：新装/迁移时用 `wsl --manage <distro> --move <path>`，避免系统盘被撑满。
- **构建前先探活**：`wsl -d Ubuntu -u root -e bash -c "touch /var/lib/docker/.t && rm /var/lib/docker/.t"`，
  只读状态下直接改用 `wsl --shutdown` 重启（或先腾空间），不要在只读状态里反复重试构建。
- **国内网络构建**：torch 走 `https://mirror.sjtu.edu.cn/pytorch-wheels/cpu`、其余依赖走清华 PyPI
  （实测 3.8MB/s vs 官方源 0.85MB/s，官方源还会长时间无连接卡死）。
- **监控写入健康**：`/ready` 已检查 PG/Redis/Milvus/LLM，建议再叠加"最近 5 分钟 dialogue_logs 是否有新增"
  的告警（只读故障下读接口全绿、写入全红，单看探针会漏报）。
