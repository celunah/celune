REM SPDX-License-Identifier: MIT

@echo off
setlocal

"%~dp0celune.exe" %*
exit /b %ERRORLEVEL%
