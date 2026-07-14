Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
currentDir = fso.GetParentFolderName(WScript.ScriptFullName)

venvPython = currentDir & "\venv\Scripts\python.exe"
globalPython = "C:\Users\Administrator\AppData\Local\Programs\Python\Python314\python.exe"

If fso.FileExists(venvPython) Then
    pythonExe = venvPython
ElseIf fso.FileExists(globalPython) Then
    pythonExe = globalPython
Else
    pythonExe = "python"
End If

scriptFile = currentDir & "\app.py"
WshShell.Run """" & pythonExe & """ """ & scriptFile & """", 0, False

