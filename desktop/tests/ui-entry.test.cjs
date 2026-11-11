const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const { getUiRevision, buildUiNavigationUrl } = require('../ui_entry.cjs');

test('UI revision is stable for unchanged index bytes and changes with referenced bundle', t => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'vt-ui-entry-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const index = path.join(directory, 'index.html');
  fs.writeFileSync(index, '<script src="/assets/index-old.js"></script>');
  const first = getUiRevision(index);
  assert.equal(getUiRevision(index), first);
  fs.writeFileSync(index, '<script src="/assets/index-new.js"></script>');
  assert.notEqual(getUiRevision(index), first);
});

test('navigation URL preserves backend origin and existing URL rules', t => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'vt-ui-entry-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const index = path.join(directory, 'index.html');
  fs.writeFileSync(index, '<html>current UI</html>');
  const result = new URL(buildUiNavigationUrl('http://127.0.0.1:8123/', index));
  assert.equal(result.origin, 'http://127.0.0.1:8123');
  assert.equal(result.pathname, '/');
  assert.equal(result.searchParams.get('_vt_ui'), getUiRevision(index));
  assert.equal([...result.searchParams.keys()].length, 1);
});
