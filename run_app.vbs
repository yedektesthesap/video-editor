Set fso = CreateObject("Scripting.FileSystemObject")
base = fso.GetParentFolderName(WScript.ScriptFullName)
pyw = base & "\\.venv\\Scripts\\pythonw.exe"
app = base & "\\main.py"

If Not fso.FileExists(pyw) Then
  MsgBox ".venv bulunamadi. Once bir kez kurulum yapin:", 48, "Video Editor"
  MsgBox "py -3 -m venv .venv" & vbCrLf & ".\\.venv\\Scripts\\python -m pip install -r requirements.txt", 64, "Kurulum"
  WScript.Quit 1
End If

cmd = Chr(34) & pyw & Chr(34) & " " & Chr(34) & app & Chr(34)
CreateObject("WScript.Shell").Run cmd, 0, False
