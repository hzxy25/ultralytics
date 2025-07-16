@echo off
REM ===== 配置部分 =====
SET CONDA_BASE=D:\studywork\anaconda3
SET CONDA_ENV=pytorchStudy
SET MODEL_PATH=D:\studywork\python\PycharmProjects\ultralytics\runs\classify\train12\weights\last.pt

REM ===== 激活 Conda 环境 =====
call "%CONDA_BASE%\Scripts\activate.bat" "%CONDA_BASE%"
if errorlevel 1 (
    echo 错误: 无法激活 Conda 基础环境
    exit /b 1
)

call activate %CONDA_ENV%
if errorlevel 1 (
    echo 错误: 无法激活 Conda 环境 "%CONDA_ENV%"
    exit /b 1
)

echo 已激活 Conda 环境: %CONDA_ENV%

REM ===== 验证 PyTorch 环境 =====
echo 验证 PyTorch 环境...
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}');"

REM ===== 执行继续训练 =====
echo 正在恢复训练模型: %MODEL_PATH%
yolo train resume ^
   model="%MODEL_PATH%" ^
   device=0  REM 添加此参数使用 GPU

echo 训练完成!
pause