@echo off
setlocal enabledelayedexpansion
:: Check if an argument is provided
if "%~1"=="" (
    echo Usage: %0 test_images path tex_name texrecon_exe
    echo Example: %0 mvs-texturing\build\apps\texrecon\texrecon.exe test_images my_folder
    exit /b 1
)

if "%~2"=="" (
    echo Error: Please provide out path as second argument
    exit /b 1
)

if "%~3"=="" (
    echo Error: Please provide the path to texrecon executable as third argument
    exit /b 1
)


:: Set the PATH and run texrecon with the provided argument
set "PATH=C:\msys64\mingw64\bin;%PATH%"
%~3 %~1\texture_albedo %~1\mesh.ply %~2\tex\tex
%~3 %~1\texture_Normals %~1\mesh.ply %~2\tex\texNormals -L %~2\tex\tex_labeling.vec -D %~2\tex\tex_data_costs.spt
%~3 %~1\texture_ReflectionMap %~1\mesh.ply %~2\tex\texReflection -L %~2\tex\tex_labeling.vec -D %~2\tex\tex_data_costs.spt
:: --- Step 1: Delete unwanted files ---
echo Deleting non-TIFF Reflection/Normal files in test_images\%~1\tex\...
for %%F in ("%~2\tex\*Reflection*" "%~2\tex\*Normal*") do (
    if /i not "%%~xF"==".tiff" (
        echo Deleting: %%F
        del "%%F" /q
    )
)

:: --- Step 2: Update .mtl file ---
set "MTL_FILE=%~2\tex\tex.mtl"
if not exist "%MTL_FILE%" (
    echo Error: .mtl file not found at "%MTL_FILE%"
    exit /b 1
)

echo Updating "%MTL_FILE%"...
set "TEMP_FILE=%MTL_FILE%.tmp"

:: Variables to track current material and existing lines
set "current_material="
set "has_map_Ks="
set "has_map_Kn="

(
    for /f "tokens=*" %%L in (%MTL_FILE%) do (
        set "line=%%L"
        
        :: Detect new material section
        if "!line:~0,6!"=="newmtl" (
            set "current_material=!line!"
            set "has_map_Ks="
            set "has_map_Kn="
        )
        
        :: Check for existing map_Ks/map_Kn in current material
        if defined current_material (
            if "!line:~0,6!"=="map_Ks" set "has_map_Ks=1"
            if "!line:~0,6!"=="map_Kn" set "has_map_Kn=1"
        )
        
        :: Replace Ks values
        set "line=!line:Ks 0.000000 0.000000 0.000000=Ks 1.000000 1.000000 1.000000!"
        echo !line!
        
        :: Process map_Kd lines
        if "!line:map_Kd=!" neq "!line!" (
            set "filename=!line:map_Kd =!"
            for /f "tokens=1,2 delims=_" %%A in ("!filename!") do (
                set "prefix=%%A"
                set "rest=%%B"
            )
            
            :: Add map_Ks if not already present in this material
            if not defined has_map_Ks (
                echo map_Ks !prefix!Reflection_!rest!_map_Kd.tiff
            )
            
            :: Add map_Kn if not already present in this material
            if not defined has_map_Kn (
                echo map_Kn !prefix!Normal_!rest!_map_Kd.tiff
            )
        )
    )
) > "%TEMP_FILE%"

move /y "%TEMP_FILE%" "%MTL_FILE%" > nul
echo .mtl file updated successfully.

endlocal