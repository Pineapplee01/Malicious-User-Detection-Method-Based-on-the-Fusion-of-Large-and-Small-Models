@echo off
echo ===== 运行SeGA-LLM简化版脚本 =====
echo.

REM 设置Python解释器路径，如果使用python命令不可用可替换为完整路径
set PYTHON=python
REM set PYTHON=D:\Python312\python.exe

REM 设置工作目录为当前脚本所在目录
cd /d "%~dp0"
echo 当前工作目录: %CD%

REM 设置环境变量
set PYTHONPATH=%CD%;%PYTHONPATH%
echo PYTHONPATH: %PYTHONPATH%

echo 开始运行模型...
%PYTHON% simple_run.py
if %errorlevel% neq 0 (
    echo 模型运行失败，请查看上面的错误信息
    pause
    exit /b 1
)

echo.
echo 运行完成！
echo 结果已保存到evaluation_results目录
pause 