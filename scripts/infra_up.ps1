# 基础设施一键启停脚本（本机环境：Docker 引擎运行于 WSL Ubuntu，docker.io 官方源被墙需走镜像）
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\infra_up.ps1          # 启动
#   powershell -ExecutionPolicy Bypass -File scripts\infra_up.ps1 -Down    # 停止
param(
    [switch]$Down
)

$compose = "docker-compose.infra.yml"
$project = "D:\AIagent\RAG-Agent"
$ubuntu = "Ubuntu"

function Invoke-Wsl {
    param([string]$Cmd)
    wsl -d $ubuntu -u root -e bash -lc $Cmd 2>&1
}

if ($Down) {
    Invoke-Wsl "cd /mnt/d/AIagent/RAG-Agent && docker compose -f $compose down 2>&1 | tail -3"
    Write-Host "[infra] stopped"
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
Write-Host "[infra] done. 注意：请保持一个 WSL 会话常驻（如 keepalive 循环），否则 VM 可能在空闲后被宿主回收。"
