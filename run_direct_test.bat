@echo off
echo ===== 运行SeGA-LLM极简测试脚本 =====
echo.

REM 设置Python解释器路径，如果使用python命令不可用可替换为完整路径
set PYTHON=python
REM set PYTHON=D:\Python312\python.exe

REM 设置工作目录为当前脚本所在目录
cd /d "%~dp0"
echo 当前工作目录: %CD%

REM 检查测试脚本是否存在
if not exist direct_test.py (
    echo 错误: 测试脚本direct_test.py不存在
    pause
    exit /b 1
)

REM 检查数据和模型
if not exist checkpoints\20250410-171222_finetune_best_model.pt (
    echo 警告: 模型文件不存在，请检查路径
    pause
)

echo 开始运行极简测试脚本...
%PYTHON% direct_test.py
if %errorlevel% neq 0 (
    echo 测试脚本运行失败，请查看上面的错误信息
    pause
    exit /b 1
)

echo.
echo 运行完成！
echo 结果已保存到evaluation_results目录
pause 