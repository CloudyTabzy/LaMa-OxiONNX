@echo off
REM Launch GIMP 3 with a visible console window.
REM After GIMP exits, shows the OxiONNX plug-in log.

set "GIMP_EXE=%LOCALAPPDATA%\Programs\GIMP 3\bin\gimp-3.2.exe"
if not exist "%GIMP_EXE%" set "GIMP_EXE=%ProgramFiles%\GIMP 3\bin\gimp-3.2.exe"
if not exist "%GIMP_EXE%" set "GIMP_EXE=%ProgramFiles(x86)%\GIMP 3\bin\gimp-3.2.exe"

if not exist "%GIMP_EXE%" (
    echo ERROR: Could not find gimp-3.2.exe under:
    echo   %%LOCALAPPDATA%%\Programs\GIMP 3
    echo   %%ProgramFiles%%\GIMP 3
    pause
    exit /b 1
)

echo Starting GIMP 3.2 with console messages...
echo.
echo The plug-in writes status to:
echo   %%APPDATA%%\GIMP\3.2\plug-ins\lama-oxionnx\lama.log
echo.
echo Close this window or quit GIMP to finish.

"%GIMP_EXE%" --new-instance --console-messages --verbose

echo.
echo ===== Plug-in log =====
if exist "%APPDATA%\GIMP\3.2\plug-ins\lama-oxionnx\lama.log" (
    type "%APPDATA%\GIMP\3.2\plug-ins\lama-oxionnx\lama.log"
) else (
    echo (no log entries yet - run the plug-in first)
)
echo =======================
echo.
pause
