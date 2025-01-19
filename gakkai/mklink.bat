@echo off

cd /d "%~dp0"
mklink /D src ..\src
mklink /D data ..\data

pause
