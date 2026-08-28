REM SPDX-License-Identifier: Apache-2.0

@echo off
setlocal

"%~dp0celune.exe" %*
exit /b %ERRORLEVEL%
