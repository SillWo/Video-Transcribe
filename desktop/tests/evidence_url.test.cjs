const assert = require('node:assert/strict');
const test = require('node:test');
const { isAllowedEvidenceUrl } = require('../evidence_url.cjs');

const valid = 'http://127.0.0.1:43127/Bf2t_aBCdEfGhiJKlmNopQrsTuvwxy12/evidence/aa3472d9-0c6f-54e0-a417-92447b7b317b?t=12.5';

test('allows only a local tokenized evidence URL with a meeting UUID and bounded timestamp', () => {
  assert.equal(isAllowedEvidenceUrl(valid), true);
  assert.equal(isAllowedEvidenceUrl(valid.replace('t=12.5', 't=604800')), true);
  assert.equal(isAllowedEvidenceUrl(valid.replace('t=12.5', 't=604800.1')), false);
});

test('rejects malformed hosts, credentials, paths, ports, and injected query parameters', () => {
  for (const value of [
    valid.replace('127.0.0.1', 'localhost'),
    valid.replace('127.0.0.1', '127.0.0.2'),
    valid.replace('http://', 'http://user@'),
    valid.replace(':43127', ':0'),
    valid.replace(':43127', ':65536'),
    valid.replace('92447b7b317b', 'not-a-uuid'),
    `${valid}&redirect=https://example.com`,
    valid.replace('t=12.5', 't=1&t=2'),
    valid.replace('t=12.5', 't=Infinity'),
    valid.replace('t=12.5', 't=-1'),
    `${valid}#https://example.com`,
    valid.replace('/evidence/', '/other/'),
  ]) assert.equal(isAllowedEvidenceUrl(value), false, value);
});
