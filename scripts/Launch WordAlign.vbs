Option Explicit

Dim shell, fso, folder, executable, command

Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
folder = fso.GetParentFolderName(WScript.ScriptFullName)
executable = fso.BuildPath(folder, "WordAlign.exe")

If Not fso.FileExists(executable) Then
    MsgBox "WordAlign.exe was not found next to this launcher.", 16, "WordAlign"
    WScript.Quit 1
End If

shell.CurrentDirectory = folder
command = """" & executable & """"
shell.Run command, 1, False
