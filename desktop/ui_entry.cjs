const fs = require('node:fs');
const { createHash } = require('node:crypto');

function getUiRevision(indexPath) {
  return createHash('sha256').update(fs.readFileSync(indexPath)).digest('hex');
}

function buildUiNavigationUrl(backendUrl, indexPath) {
  const url = new URL(backendUrl);
  url.searchParams.set('_vt_ui', getUiRevision(indexPath));
  return url.toString();
}

module.exports = { getUiRevision, buildUiNavigationUrl };
