#define MyAppName "Video Transcribe"
#define MyAppExeName "VideoTranscribe.exe"

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

#ifndef SourceDir
  #error SourceDir must be passed to ISCC.exe
#endif

[Setup]
AppId={{B5A5A0A1-917D-4DAB-8A77-42A4B6978C61}
AppName={#MyAppName}
AppVersion={#AppVersion}
AppPublisher=SillWo
DefaultDirName={autopf}\Video Transcribe
DefaultGroupName=Video Transcribe
DisableProgramGroupPage=yes
PrivilegesRequired=admin
OutputDir={#SourceDir}\..\installer
OutputBaseFilename=VideoTranscribe-Setup-{#AppVersion}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppExeName}
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Dirs]
Name: "{localappdata}\VideoTranscribe\logs"
Name: "{localappdata}\VideoTranscribe\data"
Name: "{localappdata}\VideoTranscribe\huggingface"

[Files]
Source: "{#SourceDir}\app\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Video Transcribe"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\Video Transcribe"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch Video Transcribe"; Flags: nowait postinstall skipifsilent
