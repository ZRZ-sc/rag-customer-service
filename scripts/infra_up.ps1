# 基础设施一键启停脚本（本机环境：Docker 引擎运行于 WSL Ubuntu，docker.io 官方源被墙需走镜像）
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\infra_up.ps1          # 启动（含保活）
#   powershell -ExecutionPolicy Bypass -File scripts\infra_up.ps1 -Down    # 停止
#   powershell -ExecutionPolicy Bypass -File scripts\infra_up.ps1 -Status  # 查看状态
#   powershell -ExecutionPolicy Bypass -File scripts\infra_up.ps1 -Keepalive  # 启动并前台保活（窗口需保持打开）
# 注意：本文件必须保存为 UTF-8 with BOM，否则 Windows PowerShell 5.1 会按 GBK 解码中文导致语法错误。
param(
    [switch]$Down,
    [switch]$Status,
    [switch]$Keepalive
)

$compose = "docker-compose.infra.yml"
$ubuntu = "Ubuntu"

function Invoke-Wsl {
    param([string]$Cmd)
    wsl -d $ubuntu -u root -e bash -lc $Cmd 2>&1
}

# ---------------------------------------------------------------------------
# 保活：WSL VM 在没有活跃客户端连接时会被宿主回收（即使配置了 vmIdleTimeout=0），
# 表现为容器随之停止、应用侧报 PostgreSQL/Milvus 连接拒绝。
#
# 实测结论：用 Start-Process 启动的"分离进程"会被宿主进程树回收，起不到保活作用；
# 只有前台常驻的会话才能真正把 VM 挂住。因此提供 -Keepalive：前台守护，请保持窗口打开。
# ---------------------------------------------------------------------------
function Start-KeepaliveForeground {
    Write-Host "[infra] keepalive mode: 正在前台保持 WSL 会话（Ctrl+C 退出）..."
    while ($true) {
        wsl -d $ubuntu -u root -e bash -lc "while true; do sleep 20; done"
        Write-Host "[infra] WSL 会话中断，5 秒后自动重连并恢复容器..."
        Start-Sleep -Seconds 5
        Invoke-Wsl "systemctl start docker; cd /mnt/d/AIagent/RAG-Agent && docker compose -f $compose up -d 2>&1 | tail -3"
    }
}

if ($Down) {
    Invoke-Wsl "cd /mnt/d/AIagent/RAG-Agent && docker compose -f $compose down 2>&1 | tail -3"
    Get-CimInstance Win32_Process -Filter "Name='wsl.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -match "keepalive-hold" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Write-Host "[infra] stopped (containers + keepalive)"
    exit 0
}

if ($Status) {
    Write-Host "== WSL =="
    Invoke-Wsl "uptime"
    Write-Host "== containers =="
    Invoke-Wsl "docker ps -a --format '{{.Names}} {{.Status}}'"
    Write-Host "== windows-side ports =="
    foreach ($p in 5432, 6379, 19530) {
        $state = "DOWN"
        try {
            $c = New-Object System.Net.Sockets.TcpClient
            $iar = $c.BeginConnect("127.0.0.1", $p, $null, $null)
            if ($iar.AsyncWaitHandle.WaitOne(2500) -and $c.Connected) { $state = "CONNECTED" }
            $c.Close()
        } catch { }
        Write-Host ("  127.0.0.1:{0} -> {1}" -f $p, $state)
    }
    exit 0
}

Write-Host "[infra] ensuring docker daemon in WSL..."
Invoke-Wsl "systemctl enable docker >/dev/null 2>&1; systemctl start docker; sleep 3"
Write-Host "[infra] starting containers (registry mirror: dockerproxy.net)..."
Invoke-Wsl "cd /mnt/d/AIagent/RAG-Agent && docker compose -f $compose up -d 2>&1 | tail -6"
Invoke-Wsl "docker update --restart unless-stopped rag-etcd rag-minio rag-milvus rag-postgres rag-redis >/dev/null 2>&1"
Write-Host "[infra] waiting for healthy..."
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 5
    $s = Invoke-Wsl "docker inspect -f '{{.State.Health.Status}}' rag-milvus 2>/dev/null"
    if ($s -match "healthy") { Write-Host "[infra] milvus healthy after $($i*5+5)s"; break }
}
Invoke-Wsl "docker ps --format '{{.Names}} {{.Status}}'"
if ($Keepalive) {
    Start-KeepaliveForeground
} else {
    Write-Host "[infra] done. 提示：WSL VM 在无活跃会话时会被宿主回收，建议使用 -Keepalive 参数常驻保活。"
}
