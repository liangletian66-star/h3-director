@echo off
chcp 65001 >nul
title h3-director 导演台
cd /d "%~dp0"

if not exist config.json (
    echo [!] 还没有 config.json
    echo     复制 config.example.json 为 config.json，改 comfy_dir 那一行
    echo.
    copy config.example.json config.json >nul
    echo     已生成 config.json，请先改 comfy_dir
    notepad config.json
    pause
    exit /b
)

echo ==========================================
echo   h3-director 导演台
echo   http://127.0.0.1:8190
echo   关闭本窗口 = 停止服务
echo ==========================================
echo.
echo 前提：ComfyUI 已在 8188 端口运行
echo.

start "" http://127.0.0.1:8190
python ui\h3d_server.py --port 8190
echo.
echo 服务已退出。按任意键关闭。
pause >nul
