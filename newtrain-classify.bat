@echo off
REM ===== 配置部分 =====
SET CONDA_BASE=D:\studywork\anaconda3
SET CONDA_ENV=pytorchStudy
SET YOLO_CMD=yolo classify train ^
   data="E:\tool\Data\WebFG-496-yolo" ^
   model=yolo11n-cls.pt ^
   epochs=100 ^
   imgsz=640 ^
   workers=0

REM ===== 执行部分 =====
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
echo 正在执行 YOLO 训练命令...
%YOLO_CMD%

echo 训练完成!
pause