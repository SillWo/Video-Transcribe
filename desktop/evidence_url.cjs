'use strict';

const TOKEN = /^[A-Za-z0-9_-]{16,128}$/;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const MAX_EVIDENCE_SECONDS = 7 * 24 * 60 * 60;

function isAllowedEvidenceUrl(value) {
  if (typeof value !== 'string' || value.length > 2048) return false;
  const authority = /^http:\/\/127\.0\.0\.1:(\d{1,5})(?:\/|$)/i.exec(value);
  if (!authority) return false;
  const port = Number(authority[1]);
  if (!Number.isInteger(port) || port < 1 || port > 65535) return false;
  let url;
  try { url = new URL(value); } catch { return false; }
  if (url.protocol !== 'http:' || url.hostname !== '127.0.0.1' || Number(url.port || port) !== port ||
      url.username || url.password || url.hash || url.origin !== `http://127.0.0.1:${port}`) return false;
  const match = /^\/([A-Za-z0-9_-]{16,128})\/evidence\/([0-9a-f-]{36})$/i.exec(url.pathname);
  if (!match || !TOKEN.test(match[1]) || !UUID.test(match[2])) return false;
  const params = [...url.searchParams.entries()];
  if (params.length !== 1 || params[0][0] !== 't') return false;
  const rawTime = params[0][1];
  if (!/^(?:\d+)(?:\.\d+)?$/.test(rawTime)) return false;
  const time = Number(rawTime);
  return Number.isFinite(time) && time >= 0 && time <= MAX_EVIDENCE_SECONDS;
}

module.exports = { isAllowedEvidenceUrl };
