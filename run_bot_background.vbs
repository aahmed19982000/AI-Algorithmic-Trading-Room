Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
currentDir = fso.GetParentFolderName(WScript.ScriptFullName)

Function FindPython()
    ' 1. Check local venv
    Dim localVenv
    localVenv = currentDir & "\venv\Scripts\python.exe"
    If fso.FileExists(localVenv) Then
        FindPython = localVenv
        Exit Function
    End If
    
    ' 2. Try AppData Programs folder
    Dim localAppData, pythonFolder
    localAppData = WshShell.ExpandEnvironmentStrings("%LocalAppData%")
    pythonFolder = localAppData & "\Programs\Python"
    If fso.FolderExists(pythonFolder) Then
        Dim parentFolder, subFolder, exePath
        Set parentFolder = fso.GetFolder(pythonFolder)
        For Each subFolder In parentFolder.SubFolders
            exePath = subFolder.Path & "\python.exe"
            If fso.FileExists(exePath) Then
                FindPython = exePath
                Exit Function
            End If
        Next
    End If
    
    ' 3. Check registry for common versions
    Dim regPath, val, ver
    On Error Resume Next
    For Each ver In Array("3.14", "3.13", "3.12", "3.11", "3.10", "3.9")
        regPath = "HKEY_CURRENT_USER\Software\Python\PythonCore\" & ver & "\InstallPath\"
        val = WshShell.RegRead(regPath)
        If Err.Number = 0 And val <> "" Then
            FindPython = val & "python.exe"
            Exit Function
        End If
        Err.Clear
    Next
    On Error GoTo 0
    
    ' 4. Fallback to system path
    FindPython = "python.exe"
End Function

pythonExe = FindPython()
scriptFile = currentDir & "\trading_bot.py"
WshShell.Run """" & pythonExe & """ """ & scriptFile & """ --loop --interval 300", 0, False

