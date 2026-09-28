# 监控栈一键启停脚本（Prometheus + Grafana）
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\monitoring_up.ps1            # 启动
#   powershell -ExecutionPolicy Bypass -File scripts\monitoring_up.ps1 -Status    # 状态与抓取目标连通性
#   powershell -ExecutionPolicy Bypass -File scripts\monitoring_up.ps1 -Down      # 停止
#
# 关键点：Docker 引擎在 WSL 内，FastAPI 在 Windows 宿主机上运行。
#   容器访问 Windows 需要走 WSL 默认网关 IP（会随 WSL 重启变化），
#   因此本脚本每次启动都会重新探测该 IP 并写入
#   monitoring/prometheus/targets/windows-host.json（Prometheus 用 file_sd 热加载）。
#
# 注意：本文件必须保存为 UTF-8 with BOM，否则 Windows PowerShell 5.1 按 GBK 解码中文会报语法错误。
param(
    [switch]$Down,
    [switch]$Status,
    [int]$AppPort = 8000,
    [string]$Ubuntu = "Ubuntu"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$composeFile = "docker-compose.monitoring.yml"
$targetFile = Join-Path $projectRoot "monitoring\prometheus\targets\windows-host.json"

function Invoke-Wsl {
    param([string]$Cmd)
    wsl -d $Ubuntu -u root -e bash -lc $Cmd 2>&1
}

function Get-WindowsHostIp {
    # WSL 默认网关即 Windows 宿主机在该网段的地址
    $out = Invoke-Wsl 'ip route show default | awk ''{print $3}'''
    $ip = ($out | Where-Object { $_ -match '^\d+\.\d+\.\d+\.\d+$' } | Select-Object -First 1)
    if (-not $ip) { throw "无法探测 WSL 默认网关（Windows 宿主机 IP）" }
    return $ip.Trim()
}

function Write-ScrapeTarget {
    param([string]$Ip, [int]$Port)
    $json = @"
[
  {
    "targets": ["${Ip}:${Port}"],
    "labels": {
      "service": "rag-customer-service",
      "host": "windows-host",
      "env": "local"
    }
  }
]
"@
    # 必须写入无 BOM 的 UTF-8，否则 Prometheus 解析 JSON 会失败
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($targetFile, $json, $utf8NoBom)
    Write-Host "[monitoring] 抓取目标已写入: $targetFile -> ${Ip}:${Port}"
}

function Test-ScrapeTarget {
    param([string]$Ip, [int]$Port)
    $code = (Invoke-Wsl "curl -s -m 5 -o /dev/null -w '%{http_code}' http://${Ip}:${Port}/metrics" | Select-Object -First 1)
    if ($code -match "200") {
        Write-Host "[monitoring] /metrics 可达（HTTP $code）"
        return $true
    }
    Write-Warning "[monitoring] 无法从 WSL 访问 http://${Ip}:${Port}/metrics（返回 '$code'）。请确认 FastAPI 已启动且监听 0.0.0.0:${Port}。"
    return $false
}

if ($Down) {
    Invoke-Wsl "cd /mnt/d/AIagent/RAG-Agent && docker compose -f $composeFile down" | Out-Null
    Write-Host "[monitoring] 已停止（Prometheus + Grafana）"
    exit 0
}

if ($Status) {
    Write-Host "== 容器状态 =="
    Invoke-Wsl "cd /mnt/d/AIagent/RAG-Agent && docker compose -f $composeFile ps"
    Write-Host "`n== 抓取目标文件 =="
    if (Test-Path $targetFile) { Get-Content $targetFile -Raw } else { Write-Host "（未生成）" }
    Write-Host "`n== Prometheus 目标健康 =="
    Invoke-Wsl "curl -s -m 5 http://localhost:9090/api/v1/targets | head -c 600"
    exit 0
}

$ip = Get-WindowsHostIp
Write-Host "[monitoring] Windows 宿主机（WSL 网关）: $ip"
Test-ScrapeTarget -Ip $ip -Port $AppPort | Out-Null
Write-ScrapeTarget -Ip $ip -Port $AppPort

Write-Host "[monitoring] 启动 Prometheus + Grafana ..."
Invoke-Wsl "cd /mnt/d/AIagent/RAG-Agent && docker compose -f $composeFile up -d 2>&1 | tail -8"

Write-Host "[monitoring] 等待容器就绪 ..."
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 5
    $ps = Invoke-Wsl "docker inspect -f '{{.State.Health.Status}}' rag-prometheus 2>/dev/null"
    $gs = Invoke-Wsl "docker inspect -f '{{.State.Health.Status}}' rag-grafana 2>/dev/null"
    if (($ps -match "healthy") -and ($gs -match "healthy")) {
        Write-Host "[monitoring] 就绪（prometheus=$($ps -join '') grafana=$($gs -join '')）"
        break
    }
}

Invoke-Wsl "cd /mnt/d/AIagent/RAG-Agent && docker compose -f $composeFile ps"

Write-Host ""
Write-Host "================ 监控栈已启动 ================"
Write-Host "Prometheus : http://localhost:9090         （抓取状态: /targets）"
Write-Host "Grafana    : http://localhost:3000         账号 admin / admin123"
Write-Host "仪表盘     : Grafana 左侧 Dashboards → 文件夹「RAG 智能客服」→ RAG 智能客服 - 总览"
Write-Host "抓取目标   : http://${ip}:${AppPort}/metrics （FastAPI 在 Windows 宿主机上）"
Write-Host "=============================================="
