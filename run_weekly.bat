@echo off
REM ============================================================
REM hongli_weekly weekly auto-run script
REM Called by Windows Task Scheduler every Friday 18:00
REM Pipeline handles: login -> fetch -> logout -> compute -> report
REM ============================================================
cd /d "C:\Users\Admin\Documents\agent_lab\hongli_weekly"

REM Clear stale lock file left by an interrupted run
if exist "data\.lock" (
    echo [%date% %time%] stale lock found, removing >> run_schedule.log
    del /f /q "data\.lock"
)

echo [%date% %time%] === hongli_weekly run start === >> run_schedule.log
".venv\Scripts\python.exe" -m src.hongli.pipeline >> run_schedule.log 2>&1
echo [%date% %time%] === run end, exit=%errorlevel% === >> run_schedule.log

REM Sync latest report to the published site dir (online link picks it up)
if exist "report\hongli_weekly_*.html" (
    for /f %%i in ('dir /b /o-d "report\hongli_weekly_*.html" ^| findstr /r "hongli_weekly_[0-9]*\.html"') do (
        copy /y "report\%%i" "report\site\index.html" >nul
        echo [%date% %time%] synced %%i -> report\site\index.html >> run_schedule.log
        goto :synced
    )
)
:synced
