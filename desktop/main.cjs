const { app, BrowserWindow, Tray, Menu, dialog, ipcMain, nativeImage } = require('electron');
const fs = require('node:fs');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { randomUUID } = require('node:crypto');

const PROJECT = path.resolve(__dirname, '..');
const configuredRuntime = process.env.VT_SIMPLE_RUNTIME_DIR || path.join(PROJECT, '.runtime');
const RUNTIME = path.isAbsolute(configuredRuntime) ? path.resolve(configuredRuntime) : path.resolve(PROJECT, configuredRuntime);
const DESCRIPTOR = path.join(RUNTIME, 'desktop-host.json');
const requestedPort = Number.parseInt(process.env.VT_DESKTOP_PORT || '8000', 10);
app.setName('Video Transcribe');
const userDataPath = path.join(RUNTIME, 'desktop', 'user-data');
fs.mkdirSync(userDataPath, { recursive: true });
app.setPath('userData', userDataPath);
const gotLock = app.requestSingleInstanceLock({ profile: RUNTIME });
if (!gotLock) { app.exit(0); return; }

const appDataPath = path.join(RUNTIME, 'desktop', 'app-data');
const sessionDataPath = path.join(RUNTIME, 'desktop', 'session-data');
const tempPath = path.join(RUNTIME, 'tmp', 'electron');
const crashPath = path.join(RUNTIME, 'desktop', 'crash-dumps');
const logsPath = path.join(RUNTIME, 'logs', 'electron');
for (const directory of [appDataPath, sessionDataPath, tempPath, crashPath, logsPath]) fs.mkdirSync(directory, { recursive: true });
app.setPath('appData', appDataPath);
app.setPath('sessionData', sessionDataPath);
app.setPath('temp', tempPath);
app.setPath('crashDumps', crashPath);
app.setAppLogsPath(logsPath);

let mainWindow;
let tray;
let backendChild;
let host = null;
let quitting = false;
let exitInProgress = false;
const pendingFlushes = new Map();

function readDescriptor() {
  try { return JSON.parse(fs.readFileSync(DESCRIPTOR, 'utf8')); } catch { return null; }
}
async function getHostInfo(url) {
  validateBackendUrl(url);
  const response = await fetch(`${url}/api/desktop/host`, { signal: AbortSignal.timeout(1800) });
  if (!response.ok) throw new Error('Host identity check failed');
  return response.json();
}
function validateBackendUrl(value) {
  let url;
  try { url = new URL(value); } catch { throw new Error('Адрес локального обработчика некорректен.'); }
  const port = Number(url.port);
  if (url.protocol !== 'http:' || url.hostname !== '127.0.0.1' || !url.port || !Number.isInteger(port) || port < 1 || port > 65535 ||
      url.username || url.password || url.pathname !== '/' || url.search || url.hash) {
    throw new Error('Обработчик должен использовать локальный HTTP-адрес.');
  }
  return url.origin;
}
async function stopOwnedHost() {
  if (!host?.token) throw new Error('Не удалось подтвердить право на остановку обработчика.');
  const response = await fetch(`${host.backendUrl}/api/desktop/quit`, {
    method: 'POST', headers: { 'X-VT-Host-Token': host.token }, signal: AbortSignal.timeout(18000),
  });
  if (response.status !== 202) {
    let detail = `Обработчик отказался завершить работу (код ${response.status}).`;
    try { detail = (await response.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  const deadline = Date.now() + 10000;
  while (Date.now() < deadline) {
    try {
      await fetch(`${host.backendUrl}/api/desktop/host`, { signal: AbortSignal.timeout(500) });
    } catch { return; }
    await new Promise(resolve => setTimeout(resolve, 250));
  }
  throw new Error('Обработчик получил команду остановки, но не завершился вовремя.');
}
function validTrustedSender(event) {
  if (!mainWindow || event.sender !== mainWindow.webContents) return false;
  const url = event.senderFrame?.url || '';
  try { return new URL(url).origin === new URL(host.backendUrl).origin; } catch { return false; }
}
function validatePaths(paths) {
  const files = [], rejected = [];
  for (const value of Array.isArray(paths) ? paths : []) {
    const filePath = typeof value === 'string' ? value : '';
    if (!filePath) { rejected.push({ path: '', message: 'Не удалось получить локальный путь к файлу.' }); continue; }
    try {
      const stat = fs.statSync(filePath);
      if (!stat.isFile()) { rejected.push({ path: filePath, message: 'Выберите отдельный файл, а не папку.' }); continue; }
      files.push({ path: path.resolve(filePath), name: path.basename(filePath) });
    } catch { rejected.push({ path: filePath, message: 'Файл недоступен или больше не существует.' }); }
  }
  return { files, rejected };
}
ipcMain.handle('desktop:pick-files', async (event) => {
  if (!validTrustedSender(event)) throw new Error('Untrusted desktop request.');
  const result = await dialog.showOpenDialog(mainWindow, { properties: ['openFile', 'multiSelections'],
    filters: [{ name: 'Медиафайлы', extensions: ['mp4','mkv','mov','webm','avi','mp3','wav','m4a','flac','ogg','aac'] }, { name: 'Все файлы', extensions: ['*'] }] });
  return validatePaths(result.canceled ? [] : result.filePaths);
});
ipcMain.handle('desktop:dropped-files', (event, paths) => {
  if (!validTrustedSender(event)) throw new Error('Untrusted desktop request.');
  return validatePaths(paths);
});
ipcMain.on('desktop:before-quit-done', (event, result) => {
  if (!validTrustedSender(event) || !result || typeof result.requestId !== 'string') return;
  const resolve = pendingFlushes.get(result.requestId);
  if (resolve) { pendingFlushes.delete(result.requestId); resolve(result.error ? new Error(result.error) : null); }
});

function showWindow() {
  if (!mainWindow) return createWindow();
  if (mainWindow.isMinimized()) mainWindow.restore();
  mainWindow.show(); mainWindow.focus();
}
function createWindow() {
  const { width: aw, height: ah } = require('electron').screen.getPrimaryDisplay().workAreaSize;
  const width = Math.min(1440, aw), height = Math.min(900, ah);
  mainWindow = new BrowserWindow({ width, height, minWidth: 1024, minHeight: 600,
    title: 'Video Transcribe', icon: path.join(__dirname, 'assets', 'tray.png'), show: false, autoHideMenuBar: true,
    webPreferences: { preload: path.join(__dirname, 'preload.cjs'), contextIsolation: true, nodeIntegration: false, sandbox: true } });
  mainWindow.removeMenu();
  mainWindow.webContents.on('will-navigate', (event, url) => {
    try { if (new URL(url).origin !== new URL(host.backendUrl).origin) event.preventDefault(); }
    catch { event.preventDefault(); }
  });
  mainWindow.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  mainWindow.on('close', event => { if (!quitting) { event.preventDefault(); mainWindow.hide(); } });
  mainWindow.on('closed', () => { mainWindow = null; });
  mainWindow.loadURL(host.backendUrl);
  mainWindow.once('ready-to-show', () => mainWindow?.show());
}
function createTray() {
  tray = new Tray(nativeImage.createFromPath(path.join(__dirname, 'assets', 'tray.png')));
  tray.setToolTip('Video Transcribe');
  tray.setContextMenu(Menu.buildFromTemplate([{ label: 'Открыть', click: showWindow }, { type: 'separator' }, { label: 'Выход', click: requestExit }]));
  tray.on('double-click', showWindow);
}
function requestRendererFlush(timeout = 15000) {
  if (!mainWindow || mainWindow.webContents.isLoading()) return Promise.resolve(null);
  const requestId = randomUUID();
  return new Promise(resolve => {
    const timer = setTimeout(() => { pendingFlushes.delete(requestId); resolve(new Error('Не удалось сохранить изменения интерфейса вовремя.')); }, timeout);
    pendingFlushes.set(requestId, error => { clearTimeout(timer); resolve(error); });
    mainWindow.webContents.send('desktop:before-quit', requestId);
  });
}
async function requestExit() {
  if (quitting || exitInProgress) return;
  exitInProgress = true;
  let inputDisabled = false;
  try {
    if (backendChild && backendChild.exitCode !== null) {
      if (mainWindow) { mainWindow.setEnabled(false); inputDisabled = true; }
      const flushError = await requestRendererFlush();
      if (flushError) throw flushError;
      quitting = true;
      mainWindow?.removeAllListeners('close'); mainWindow?.destroy(); tray?.destroy();
      app.quit();
      return;
    }
    const info = await fetch(`${host.backendUrl}/api/simple/queue`, { signal: AbortSignal.timeout(2500) }).then(r => r.json());
    const busy = Boolean(info.activeJob || info.waitingCount || info.waitingJobs?.length);
    if (busy) {
      const answer = await dialog.showMessageBox(mainWindow || undefined, { type: 'warning', buttons: ['Выйти', 'Продолжить работу'], defaultId: 1, cancelId: 1,
        title: 'Очередь транскрибации', message: 'В очереди есть задачи. При выходе текущая задача остановится, а ожидающие останутся в очереди.' });
      if (answer.response !== 0) return;
    }
    if (mainWindow) { mainWindow.setEnabled(false); inputDisabled = true; }
    const flushError = await requestRendererFlush();
    if (flushError) throw flushError;
    await stopOwnedHost();
    quitting = true;
    mainWindow?.removeAllListeners('close'); mainWindow?.destroy(); tray?.destroy();
    app.quit();
  } catch (error) {
    if (inputDisabled) mainWindow?.setEnabled(true);
    await dialog.showMessageBox(mainWindow || undefined, { type: 'error', title: 'Не удалось завершить работу',
      message: String(error?.message || error), detail: 'Приложение и обработчик продолжают работать.' });
    showWindow();
  } finally {
    if (!quitting) exitInProgress = false;
  }
}

async function startOrAttachHost() {
  fs.mkdirSync(RUNTIME, { recursive: true });
  const record = readDescriptor();
  if (record?.appId === 'video-transcribe' && path.resolve(record.projectRoot || '') === PROJECT && Number.isInteger(record.pid)) {
    try {
      const info = await getHostInfo(record.backendUrl);
      if (info.instanceId === record.instanceId && info.projectRoot === PROJECT && info.appId === 'video-transcribe') { host = record; return; }
    } catch {}
  }
  const python = path.join(PROJECT, '.venv', 'Scripts', 'python.exe');
  fs.mkdirSync(path.join(RUNTIME, 'logs'), { recursive: true });
  const stdout = fs.openSync(path.join(RUNTIME, 'logs', 'desktop-host.out.log'), 'a');
  const stderr = fs.openSync(path.join(RUNTIME, 'logs', 'desktop-host.err.log'), 'a');
  const launchId = randomUUID();
  const hostEnv = { ...process.env, VT_SIMPLE_RUNTIME_DIR: RUNTIME, VT_DESKTOP_LAUNCH_ID: launchId };
  backendChild = spawn(python, ['-m', 'services.app_host', '--managed', '--port', String(Number.isInteger(requestedPort) ? requestedPort : 8000)], { cwd: PROJECT, env: hostEnv, windowsHide: true, stdio: ['pipe', stdout, stderr] });
  fs.closeSync(stdout); fs.closeSync(stderr);
  backendChild.on('error', error => dialog.showErrorBox('Обработчик не запустился', error.message));
  const deadline = Date.now() + 45000;
  while (Date.now() < deadline) {
    if (backendChild.exitCode !== null) throw new Error('Не удалось запустить обработчик.');
    await new Promise(resolve => setTimeout(resolve, 250));
    const current = readDescriptor();
    if (current?.appId === 'video-transcribe' && path.resolve(current.projectRoot || '') === PROJECT && current.launchId === launchId) {
      try {
        const info = await getHostInfo(current.backendUrl);
        if (info.instanceId === current.instanceId && info.projectRoot === PROJECT && info.launchId === launchId) { host = current; return; }
      } catch {}
    }
  }
  throw new Error('Обработчик не подтвердил готовность за 45 секунд. Возможно, порт 8000 занят другим приложением.');
}

app.on('second-instance', showWindow);
app.whenReady().then(async () => {
  try { await startOrAttachHost(); createWindow(); createTray(); }
  catch (error) {
    await dialog.showMessageBox({ type: 'error', title: 'Video Transcribe не запустился', message: String(error?.message || error) });
    quitting = true;
    if (backendChild?.stdin && !backendChild.stdin.destroyed) backendChild.stdin.end();
    app.quit();
  }
});
app.on('window-all-closed', () => {});
app.on('before-quit', event => { if (!quitting) event.preventDefault(); });

module.exports = { validatePaths, validateBackendUrl, requestExit, showWindow };
