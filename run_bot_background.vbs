Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
currentDir = fso.GetParentFolderName(WScript.ScriptFullName)
pythonExe = currentDir & "\venv\Scripts\python.exe"
scriptFile = currentDir & "\trading_bot.py"
WshShell.Run """" & pythonExe & """ """ & scriptFile & """ --loop --interval 300", 0, False
