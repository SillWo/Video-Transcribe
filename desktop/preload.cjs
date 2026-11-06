const { contextBridge, ipcRenderer, webUtils } = require('electron');
const beforeQuitHandlers = new Set();
ipcRenderer.on('desktop:before-quit', async (_event, requestId) => {
  const results = await Promise.allSettled(Array.from(beforeQuitHandlers, handler => handler()));
  const failure = results.find(result => result.status === 'rejected');
  ipcRenderer.send('desktop:before-quit-done', { requestId,
    error: failure ? String(failure.reason?.message || failure.reason) : undefined });
});

contextBridge.exposeInMainWorld('desktopApi', Object.freeze({
  isDesktop: true,
  version: 1,
  pickFiles: () => ipcRenderer.invoke('desktop:pick-files'),
  getDroppedFiles: (files) => ipcRenderer.invoke('desktop:dropped-files',
    Array.from(files || [], (file) => { try { return webUtils.getPathForFile(file); } catch { return ''; } })),
  onBeforeQuit(handler) {
    if (typeof handler !== 'function') return () => {};
    beforeQuitHandlers.add(handler);
    return () => beforeQuitHandlers.delete(handler);
  }
}));
