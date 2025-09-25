@echo off
echo ===== SeGA-LLM 模型简易运行脚本 =====
echo.

REM 设置变量
set DATA_PATH=..\processed_data
set MODEL_PATH=.\checkpoints\20250410-171222_finetune_best_model.pt
set OUTPUT_DIR=..\evaluation_results
set PYTHON=python

echo 数据路径: %DATA_PATH%
echo 模型路径: %MODEL_PATH%
echo 输出路径: %OUTPUT_DIR%
echo.

echo 步骤1: 准备LLM特征文件
echo 正在为mistral模型创建特征文件...
%PYTHON% -c "import torch, os; data_path=os.path.abspath('%DATA_PATH%'); user_cat = torch.load(os.path.join(data_path, 'user_cat_properties_tensor.pt')); list_cat = torch.load(os.path.join(data_path, 'list_cat_properties_tensor.pt')); torch.save(torch.zeros((user_cat.shape[0], 100)), os.path.join(data_path, 'user_llm_features_mistral.pt')); torch.save(torch.zeros((list_cat.shape[0], 100)), os.path.join(data_path, 'list_llm_features_mistral.pt')); print('创建完成')"

if %errorlevel% neq 0 (
    echo 创建文件失败
    pause
    exit /b 1
)
echo.

echo 步骤2: 运行主模型
%PYTHON% -c "import sys; sys.path.append('..'); from SeGA_LLM.main_s import main; main()"

if %errorlevel% neq 0 (
    echo 模型运行失败
    pause
    exit /b 1
)

echo.
echo 运行完成！
echo 结果保存在 %OUTPUT_DIR% 目录中
pause 