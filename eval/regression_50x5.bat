@echo off
REM 50x5 并发回归测试(跑测+LLM裁判+基线对比,任一指标退化>0.05 则退出码 1)
REM 全量约 40 分钟;冒烟: regression_50x5.bat --users 5
REM 只比对现有结果: regression_50x5.bat --skip-run
set NO_PROXY=192.168.88.138,127.0.0.1,localhost,.volces.com
set HTTP_PROXY=
set HTTPS_PROXY=
cd /d %~dp0..
python -m eval.regression_50x5 %*
