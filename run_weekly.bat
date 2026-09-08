@echo off
REM ============================================================
REM hongli_weekly weekly auto-run script
REM Called by Windows Task Scheduler every Friday 18:00
REM Pipeline handles: login -> fetch -> logout -> compute -> report
REM                   + feishu notify (inside pipeline)
REM Then: publish latest report to GitHub Pages (gh-pages branch)
REM NOTE: never checkout/clean in the main working tree - we push
REM       gh-pages from a throwaway temp dir instead.
REM ============================================================
setlocal enabledelayedexpansion
cd /d "C:\Users\Admin\Documents\agent_lab\hongli_weekly"
set "GIT_SSH_COMMAND=ssh -F C:/Users/Admin/.ssh/config_github"

REM Clear stale lock file left by an interrupted run
if exist "data\.lock" (
    echo [%date% %time%] stale lock found, removing >> run_schedule.log
    del /f /q "data\.lock"
)

echo [%date% %time%] === hongli_weekly run start === >> run_schedule.log
".venv\Scripts\python.exe" -m src.hongli.pipeline >> run_schedule.log 2>&1
echo [%date% %time%] === pipeline end, exit=%errorlevel% === >> run_schedule.log

REM ---- find newest report and stage it as docs/index.html ----
set "LATEST="
if exist "report\hongli_weekly_*.html" (
    for /f "delims=" %%i in ('dir /b /o-d "report\hongli_weekly_*.html"') do (
        if not defined LATEST (
            set "LATEST=%%i"
            if not exist docs mkdir docs
            copy /y "report\%%i" "docs\index.html" >nul
            echo [%date% %time%] staged %%i -^> docs\index.html >> run_schedule.log
        )
    )
)

REM ---- publish: push docs/index.html to gh-pages via temp dir ----
if exist "docs\index.html" (
    if exist .ghpages_tmp rmdir /s /q .ghpages_tmp
    mkdir .ghpages_tmp
    copy /y "docs\index.html" ".ghpages_tmp\index.html" >nul
    pushd .ghpages_tmp
    git init -q -b gh-pages >> ..\run_schedule.log 2>&1
    git add index.html >> ..\run_schedule.log 2>&1
    git -c user.name="coffiney" -c user.email="coffiney@users.noreply.github.com" commit -qm "weekly report auto-update" >> ..\run_schedule.log 2>&1
    git remote add origin git@github.com:coffiney/hongli_weekly.git >> ..\run_schedule.log 2>&1
    git push origin gh-pages --force >> ..\run_schedule.log 2>&1
    echo [%date% %time%] gh-pages push exit=!errorlevel! >> ..\run_schedule.log
    popd
    rmdir /s /q .ghpages_tmp
)
echo [%date% %time%] === publish done === >> run_schedule.log
endlocal
