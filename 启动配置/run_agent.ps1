# run_agent.ps1 - 半导体设备知识问答系统启动器 (单窗口版)
# 前后端都在本窗口运行(共享控制台输出), 不再分别弹窗。
# 三重保障, 关闭窗口 / Ctrl+C / 按任意键 任一方式退出都会终止全部进程:
#   1) Windows Job Object (KILL_ON_JOB_CLOSE): OS 级, 本进程退出即杀整个进程树(含 python/node)
#   2) SetConsoleCtrlHandler: 捕获 CTRL_CLOSE/C 事件 -> TerminateJobObject 杀 job 全树 + taskkill /T 兜底
#   3) finally: 退出时主动 TerminateJobObject + taskkill /T
$ErrorActionPreference = 'Stop'
# 本地服务互调(8001/8002/3000)必须绕过系统代理(Clash 等会把 127.0.0.1 请求代理成 502);
# 只排除本地回环,外网 LLM API 仍可正常走 HTTP_PROXY。
# 本地回环 + 国内端点必须直连:火山方舟 LLM/视觉(ark.cn-beijing.volces.com)与 HF 国内镜像
# 若被 Clash 等代理绕到国外节点会 ReadTimeout(实测经代理 5/5 超时,直连 5/5 ~0.5s)。
# 国外服务(Tavily/Serper 联网搜索)不在此列,仍走 HTTP_PROXY。
$env:NO_PROXY = '127.0.0.1,localhost,::1,.volces.com,volces.com,.hf-mirror.com,hf-mirror.com'
$env:no_proxy = '127.0.0.1,localhost,::1,.volces.com,volces.com,.hf-mirror.com,hf-mirror.com'
# Qdrant 走服务器模式(Docker :6333),不再用会每次重建 HNSW 的本地文件模式(path=)。
# 留空则回退本地文件模式;query.py / ingest_core.py / pdf.ingest.py 均读此变量。
$env:QDRANT_URL = 'http://127.0.0.1:6333'
try { chcp 65001 | Out-Null } catch {}
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

$root = Split-Path $PSScriptRoot -Parent
$py   = Join-Path $root '.venv_mineru\Scripts\python.exe'

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   半导体设备知识问答系统 启动器 (单窗口)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

if (-not (Test-Path -LiteralPath $py)) { Write-Host "[错误] 未找到 Python: $py" -ForegroundColor Red; exit 1 }
foreach ($p in @('server\main.py','web\package.json')) {
    if (-not (Test-Path -LiteralPath (Join-Path $root $p))) { Write-Host "[错误] 未找到: $p" -ForegroundColor Red; exit 1 }
}

$src = 'using System;
using System.Runtime.InteropServices;
public static class Semi {
  [StructLayout(LayoutKind.Sequential)]
  public struct BASIC { public long A; public long B; public uint LimitFlags; public UIntPtr C; public UIntPtr D; public uint E; public UIntPtr F; public uint G; public uint H; }
  [StructLayout(LayoutKind.Sequential)]
  public struct IO { public ulong a; public ulong b; public ulong c; public ulong d; public ulong e; public ulong f; }
  [StructLayout(LayoutKind.Sequential)]
  public struct EXT { public BASIC Basic; public IO Io; public UIntPtr PML; public UIntPtr JML; public UIntPtr PPM; public UIntPtr PJM; }
  [DllImport("kernel32.dll", CharSet=CharSet.Unicode)] public static extern IntPtr CreateJobObject(IntPtr a, string n);
  [DllImport("kernel32.dll")] public static extern bool SetInformationJobObject(IntPtr h, int t, ref EXT i, uint s);
  [DllImport("kernel32.dll")] public static extern bool AssignProcessToJobObject(IntPtr j, IntPtr p);
  [DllImport("kernel32.dll", SetLastError=true)] public static extern bool CloseHandle(IntPtr h);
  [DllImport("kernel32.dll")] public static extern bool TerminateJobObject(IntPtr j, uint c);
  public delegate bool CtrlHandler(uint t);
  [DllImport("kernel32.dll")] public static extern bool SetConsoleCtrlHandler(CtrlHandler h, bool add);
  public static IntPtr JobHandle = IntPtr.Zero;
  public static int[] Pids = new int[0];
  public static bool OnCtrl(uint t) {
    if (JobHandle != IntPtr.Zero) { try { TerminateJobObject(JobHandle, 1); } catch {} }
    foreach (var pid in Pids) {
      try { var p = System.Diagnostics.Process.Start("taskkill.exe", "/PID " + pid + " /T /F");
            if (p != null) p.WaitForExit(1500); } catch {}
    }
    return false;
  }
}'
if (-not ('Semi' -as [type])) { Add-Type -TypeDefinition $src }

$job = [Semi]::CreateJobObject([IntPtr]::Zero, $null)
if ($job -eq [IntPtr]::Zero) { Write-Host "[错误] CreateJobObject 失败" -ForegroundColor Red; exit 1 }
$ext = New-Object Semi+EXT
$ext.Basic.LimitFlags = 0x2000
[void][Semi]::SetInformationJobObject($job, 9, [ref]$ext, [uint32][System.Runtime.InteropServices.Marshal]::SizeOf($ext))

function Add-ToJob($proc) {
    $ok = [Semi]::AssignProcessToJobObject($job, $proc.Handle)
    if (-not $ok) { Write-Host "[警告] $($proc.ProcessName) 未能加入 Job(退出时由 taskkill 兜底)" -ForegroundColor Yellow }
}

function Start-Inline($file, $argList, $workdir) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $file
    $psi.Arguments = $argList
    $psi.WorkingDirectory = $workdir
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $false
    $p = New-Object System.Diagnostics.Process
    $p.StartInfo = $psi
    [void]$p.Start()
    return $p
}

Write-Host ""
Write-Host "向量库: Qdrant 服务器模式 (Docker $env:QDRANT_URL)" -ForegroundColor DarkGray

Write-Host ""
Write-Host "[1/3] 启动检索服务(MCP /mcp + HTTP)  http://127.0.0.1:8002 ..." -ForegroundColor Cyan
$rs = Start-Inline $py 'mcp_servers\retrieval\service.py' $root
Add-ToJob $rs

Write-Host "[2/3] 启动后端  FastAPI  http://127.0.0.1:8001 ..." -ForegroundColor Cyan
$be = Start-Inline $py 'server\main.py' $root
Add-ToJob $be

Write-Host "[3/3] 启动前端  Next.js  http://localhost:3000 ..." -ForegroundColor Cyan
Start-Sleep -Milliseconds 300
$fe = Start-Inline 'cmd.exe' '/c npm run dev' (Join-Path $root 'web')
Add-ToJob $fe

[Semi]::JobHandle = $job
[Semi]::Pids = @($be.Id, $fe.Id, $rs.Id)
$ctrl = [Semi+CtrlHandler] { param($t) [Semi]::OnCtrl($t) }
[void][Semi]::SetConsoleCtrlHandler($ctrl, $true)
$global:__semiCtrl = $ctrl

function Stop-All {
    try { [void][Semi]::TerminateJobObject($job) } catch {}
    foreach ($id in @($be.Id, $fe.Id, $rs.Id)) { try { taskkill /PID $id /T /F 2>$null | Out-Null } catch {} }
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  服务已启动 (前后端日志输出到本窗口):" -ForegroundColor Green
Write-Host "    前端  http://localhost:3000" -ForegroundColor White
Write-Host "    后端  http://127.0.0.1:8001" -ForegroundColor White
Write-Host "    检索  http://127.0.0.1:8002" -ForegroundColor White
Write-Host "------------------------------------------" -ForegroundColor Green
Write-Host "  关闭本窗口 / Ctrl+C / 按任意键  =>  结束全部" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""

try {
    while ($true) {
        $be.Refresh(); $fe.Refresh()
        if ($be.HasExited -or $fe.HasExited -or $rs.HasExited) { break }
        try { if ([Console]::KeyAvailable) { [void][Console]::ReadKey($true); break } } catch {}
        Start-Sleep -Milliseconds 200
    }
    if ($rs.HasExited) { Write-Host "`n[检索服务已退出]" -ForegroundColor Magenta }
    if ($be.HasExited) { Write-Host "`n[后端已退出]" -ForegroundColor Magenta }
    if ($fe.HasExited) { Write-Host "`n[前端已退出]" -ForegroundColor Magenta }
} catch {} finally {
    [void][Semi]::SetConsoleCtrlHandler($ctrl, $false)
    Write-Host "`n正在停止所有服务..." -ForegroundColor Yellow
    Stop-All
    try { [void][Semi]::CloseHandle($job) } catch {}
    Write-Host "已退出。" -ForegroundColor Green
}