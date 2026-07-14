Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
currentDir = fso.GetParentFolderName(WScript.ScriptFullName)
pythonExe = currentDir & "\venv\Scripts\python.exe"
scriptFile = currentDir & "\app.py"
WshShell.Run """" & pythonExe & """ """ & scriptFile & """", 0, False
