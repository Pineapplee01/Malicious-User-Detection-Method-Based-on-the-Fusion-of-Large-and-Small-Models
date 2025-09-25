@echo off
echo ===== SeGA-LLM 模型运行脚本 =====
echo.

REM 设置Python解释器路径，如果使用python命令不可用可替换为完整路径
set PYTHON=python
REM set PYTHON=D:\Python312\python.exe

echo 步骤1: 准备LLM特征文件
echo 正在为所有可能的LLM模型创建特征文件...
%PYTHON% prepare_llm_features.py --all_models --dimensions 100
if %errorlevel% neq 0 (
    echo 创建LLM特征文件失败，请检查错误信息
    pause
    exit /b 1
)
echo.

echo 步骤2: 检查必要的预训练文件
if not exist ..\processed_data\pretrain_labels_index.pt (
    echo 警告: 预训练标签文件不存在
    echo 正在创建空的预训练标签文件...
    %PYTHON% -c "import torch; import os; torch.save(torch.zeros(100001, dtype=torch.long), os.path.abspath('../processed_data/pretrain_labels_index.pt')); torch.save(torch.zeros(100001, dtype=torch.long), os.path.abspath('../processed_data/pretrain_labels_index_l.pt'));"
    if %errorlevel% neq 0 (
        echo 创建预训练标签文件失败
        pause
        exit /b 1
    )
    echo 成功创建预训练标签文件
)
echo.

echo 步骤3: 运行主模型
%PYTHON% main_s.py
if %errorlevel% neq 0 (
    echo 模型运行失败，请检查错误信息
    pause
    exit /b 1
)
echo.

echo 运行完成！
echo 结果保存在 evaluation_results 目录中
pause 