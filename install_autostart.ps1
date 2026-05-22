$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ShortcutPath = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\SonFED.lnk"
$Python = (Get-Command python).Source
$WScriptShell = New-Object -ComObject WScript.Shell
$Shortcut = $WScriptShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $Python
$Shortcut.Arguments = "`"$AppDir\launcher_tray.py`""
$Shortcut.WorkingDirectory = $AppDir
$Shortcut.IconLocation = "$Python,0"
$Shortcut.Save()
Write-Host "Đã bật tự khởi động SonFED cùng Windows: $ShortcutPath"
