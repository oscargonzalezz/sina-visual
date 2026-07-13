@echo off
cd /d "%~dp0"
echo Compilando SINA-VISUAL...
gcc -Wall -std=c99 src/main.c src/sinacore.c src/app.c -o sina_visual.exe -lraylib -lopengl32 -lgdi32 -lwinmm -lm
if %ERRORLEVEL% EQU 0 (
    echo Compilacion exitosa! Iniciando programa...
    sina_visual.exe
) else (
    echo ERROR en la compilacion.
    pause
)
