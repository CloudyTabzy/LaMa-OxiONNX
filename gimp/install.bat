@echo off
REM install.bat - Install the LaMa OxiONNX plug-in for GIMP 3.2.
REM
REM Usage:
REM   install.bat
REM
REM What this installer does (no Python worker, no ML wheels):
REM   1. Creates the user-level GIMP interpreter alias for the plug-in
REM      shebang (never touches GIMP's installation files).
REM   2. Copies the GIMP-side bridge (lama-oxionnx.py) into the plug-in dir.
REM   3. Installs the LaMa model: copies it from this repo, from the
REM      existing ONNX Runtime plug-in install, or downloads it.
REM   4. Builds the pure-Rust OxiONNX worker with cargo and copies it
REM      next to the plug-in (plus the prebuilt session cache when one is
REM      available, so the first run does not have to build it).
REM
REM The plug-in installs side by side with the ONNX Runtime plug-in
REM (plug-ins\lama-inpaint): own procedure, own menu entry, own folder.

setlocal EnableExtensions

set "SRC=%~dp0"
set "DEST=%APPDATA%\GIMP\3.2\plug-ins\lama-oxionnx"
set "INTERPRETER_DIR=%APPDATA%\GIMP\3.2\interpreters"
set "REPO_ROOT=%SRC%.."
set "GIMP_ROOT="
set "GIMP_PYTHON="
set "MODEL_STATUS=already present"
set "WORKER_STATUS=already present"

if not defined APPDATA (
    echo ERROR: APPDATA is not defined.
    exit /b 1
)

if not exist "%SRC%lama-oxionnx.py" (
    echo ERROR: Required file not found: %SRC%lama-oxionnx.py
    exit /b 1
)

call :find_gimp
if not defined GIMP_ROOT goto :no_gimp
set "GIMP_PYTHON=%GIMP_ROOT%\bin\python.exe"
if not exist "%GIMP_PYTHON%" goto :no_gimp

echo GIMP install: %GIMP_ROOT%
echo GIMP Python:  %GIMP_PYTHON%
echo.

REM --- 1. User-level interpreter alias for the plug-in shebang ---
set "LAMA_INTERPRETER_DIR=%INTERPRETER_DIR%"
set "LAMA_GIMP_PYTHON=%GIMP_PYTHON%"
set "LAMA_GIMP_PYTHONW=%GIMP_ROOT%\bin\pythonw.exe"
if not exist "%LAMA_GIMP_PYTHONW%" set "LAMA_GIMP_PYTHONW=%GIMP_PYTHON%"
set "LAMA_GIMP_PYTHON_GUI=%LAMA_GIMP_PYTHONW%"
"%GIMP_PYTHON%" -c "import os; from pathlib import Path; directory=Path(os.environ['LAMA_INTERPRETER_DIR']); directory.mkdir(parents=True, exist_ok=True); console='lama-oxionnx-gimp-python=' + Path(os.environ['LAMA_GIMP_PYTHON']).resolve().as_posix() + chr(10); gui='lama-oxionnx-gimp-python=' + Path(os.environ['LAMA_GIMP_PYTHON_GUI']).resolve().as_posix() + chr(10); (directory / 'lama-oxionnx-gimp-python.interp').write_text(console, encoding='utf-8'); (directory / 'lama-oxionnx-gimp-python_win.interp').write_text(gui, encoding='utf-8')"
if not errorlevel 1 goto :interpreter_written

REM Fallback: write the same mappings with batch echo if the bundled
REM Python refuses to run standalone. GIMP accepts forward slashes.
echo NOTE: bundled Python could not write the mappings; using batch fallback.
if not exist "%INTERPRETER_DIR%" mkdir "%INTERPRETER_DIR%"
> "%INTERPRETER_DIR%\lama-oxionnx-gimp-python.interp" echo lama-oxionnx-gimp-python=%LAMA_GIMP_PYTHON:\=/%
> "%INTERPRETER_DIR%\lama-oxionnx-gimp-python_win.interp" echo lama-oxionnx-gimp-python=%LAMA_GIMP_PYTHON_GUI:\=/%

:interpreter_written
if not exist "%INTERPRETER_DIR%\lama-oxionnx-gimp-python.interp" goto :interpreter_failed
if not exist "%INTERPRETER_DIR%\lama-oxionnx-gimp-python_win.interp" goto :interpreter_failed

REM --- 2. Plug-in directory and bridge ---
echo Creating plug-in directory: %DEST%
if not exist "%DEST%" mkdir "%DEST%"
if errorlevel 1 (
    echo ERROR: Could not create the plug-in directory.
    exit /b 1
)

copy /Y "%SRC%lama-oxionnx.py" "%DEST%\lama-oxionnx.py" >nul || goto :copy_failed

REM --- 3. LaMa model (dynamic-H/W export) ---
set "MODEL_DEST=%DEST%\lama_fp32.onnx"
set "MODEL_URL=https://github.com/CloudyTabzy/Gimp-lama-inpainting/releases/download/v1.1.0/lama_fp32.onnx"
set "CACHE_DEST=%DEST%\lama_fp32.onnx.r1.oxicache"

if exist "%MODEL_DEST%" (
    echo Model: already present
    goto :model_ready
)
if exist "%SRC%lama_fp32.onnx" goto :model_from_repo
if exist "%APPDATA%\GIMP\3.2\plug-ins\lama-inpaint\lama_fp32.onnx" goto :model_from_ortp
goto :model_download

:model_from_repo
set "MODEL_SRC_DIR=%SRC%"
goto :model_copy

:model_from_ortp
set "MODEL_SRC_DIR=%APPDATA%\GIMP\3.2\plug-ins\lama-inpaint\"
goto :model_copy

:model_copy
echo Copying LaMa model from:
echo   %MODEL_SRC_DIR%lama_fp32.onnx
copy /Y "%MODEL_SRC_DIR%lama_fp32.onnx" "%MODEL_DEST%" >nul || goto :model_copy_failed
set "MODEL_STATUS=copied"
if not exist "%MODEL_SRC_DIR%lama_fp32.onnx.r1.oxicache" goto :model_ready
if exist "%CACHE_DEST%" goto :model_ready
echo Copying the prebuilt OxiONNX session cache, about 373 MB...
copy /Y "%MODEL_SRC_DIR%lama_fp32.onnx.r1.oxicache" "%CACHE_DEST%" >nul
if errorlevel 1 (
    echo NOTE: cache copy failed. The worker will rebuild it on first run.
) else (
    echo Cache copied. The first inference will be fast.
)
goto :model_ready

:model_download
echo.
echo LaMa model not found locally. Downloading from:
echo   %MODEL_URL%
echo This is a ~200 MB file and may take a moment.
echo.
powershell -NoProfile -Command "try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%MODEL_URL%' -OutFile '%MODEL_DEST%' -UseBasicParsing } catch { exit 1 }"
if errorlevel 1 (
    echo.
    echo WARNING: Could not download the LaMa model automatically.
    echo   Please download it manually from:
    echo     %MODEL_URL%
    echo   and place it at:
    echo     %MODEL_DEST%
    echo The plug-in will not work without it.
    echo.
    set "MODEL_STATUS=download failed"
) else (
    echo Download complete: %MODEL_DEST%
    set "MODEL_STATUS=downloaded"
)
goto :model_ready

:model_copy_failed
echo WARNING: Could not copy the model to %MODEL_DEST%.
set "MODEL_STATUS=copy failed"
goto :model_ready

:model_ready

REM --- 4. Pure-Rust OxiONNX worker ---
set "WORKER_DEST=%DEST%\lama-worker-oxionnx.exe"
set "WORKER_SRC=%REPO_ROOT%\target\release\lama-worker-oxionnx.exe"

if exist "%WORKER_DEST%" (
    echo Worker: already present
    goto :worker_done
)

call :find_cargo
if errorlevel 1 goto :worker_no_cargo

echo.
echo Building the OxiONNX worker. The first build takes a few minutes...
cargo build --release --manifest-path "%REPO_ROOT%\Cargo.toml"
if errorlevel 1 goto :worker_build_failed

if not exist "%WORKER_SRC%" goto :worker_binary_missing

copy /Y "%WORKER_SRC%" "%WORKER_DEST%" >nul
if errorlevel 1 goto :worker_copy_failed

echo Worker installed: %WORKER_DEST%
set "WORKER_STATUS=installed"
goto :worker_done

:worker_no_cargo
echo.
echo WARNING: cargo was not found on PATH; the worker was not installed.
echo   Install Rust 1.94 or newer from https://rustup.rs, then build with:
echo     cargo build --release --manifest-path "%REPO_ROOT%\Cargo.toml"
echo   and copy:
echo     %WORKER_SRC%
echo   to:
echo     %WORKER_DEST%
set "WORKER_STATUS=not installed (no cargo on PATH)"
goto :worker_done

:worker_build_failed
echo.
echo WARNING: cargo build failed; the worker was not installed.
echo   Fix the build error above, then re-run this installer.
set "WORKER_STATUS=not installed (build failed)"
goto :worker_done

:worker_binary_missing
echo WARNING: cargo reported success but the worker binary is missing:
echo   %WORKER_SRC%
set "WORKER_STATUS=not installed (binary missing)"
goto :worker_done

:worker_copy_failed
echo WARNING: Could not copy the worker to %WORKER_DEST%.
set "WORKER_STATUS=not installed (copy failed)"

:worker_done

REM --- Summary ---
echo.
echo Installed LaMa OxiONNX to:
echo   %DEST%
echo User-level GIMP interpreter mappings:
echo   %INTERPRETER_DIR%
echo Worker status: %WORKER_STATUS%
echo Model status:  %MODEL_STATUS%
echo.
if not "%MODEL_STATUS%"=="already present" if not "%MODEL_STATUS%"=="copied" if not "%MODEL_STATUS%"=="downloaded" goto :warn_incomplete
if not "%WORKER_STATUS%"=="already present" if not "%WORKER_STATUS%"=="installed" goto :warn_incomplete
echo Restart GIMP, then use Filters ^> Enhance ^> LaMa Inpaint (OxiONNX)...
echo This is a separate entry from the ONNX Runtime plug-in; both can be
echo installed and compared side by side.
echo.
echo The first inference builds the OxiONNX session cache next to the model
echo when one was not copied, taking about two minutes and ~373 MB. Later
echo runs load the cache in under a second.
endlocal
exit /b 0

:warn_incomplete
echo WARNING: The plug-in is installed but not fully operational yet.
echo   Fix the worker and/or model issues reported above, then re-run this
echo   installer or restart GIMP to retry.
endlocal
exit /b 1

:no_gimp
echo ERROR: Could not find GIMP 3 bundled Python.
echo Looked for:
echo   %LOCALAPPDATA%\Programs\GIMP 3\bin\python.exe
echo   %ProgramFiles%\GIMP 3\bin\python.exe
echo   %ProgramFiles(x86)%\GIMP 3\bin\python.exe
exit /b 1

:interpreter_failed
echo ERROR: Failed to write the GIMP interpreter mappings to:
echo   %INTERPRETER_DIR%
exit /b 1

:copy_failed
echo ERROR: Failed to copy a plug-in file to:
echo   %DEST%
echo Existing unrelated files in that directory were left in place.
exit /b 1

:find_gimp
if defined LOCALAPPDATA if exist "%LOCALAPPDATA%\Programs\GIMP 3\bin\python.exe" set "GIMP_ROOT=%LOCALAPPDATA%\Programs\GIMP 3"
if not defined GIMP_ROOT if defined ProgramFiles if exist "%ProgramFiles%\GIMP 3\bin\python.exe" set "GIMP_ROOT=%ProgramFiles%\GIMP 3"
if not defined GIMP_ROOT if defined ProgramFiles(x86) if exist "%ProgramFiles(x86)%\GIMP 3\bin\python.exe" set "GIMP_ROOT=%ProgramFiles(x86)%\GIMP 3"
exit /b 0

:find_cargo
where cargo >nul 2>&1
exit /b %errorlevel%
