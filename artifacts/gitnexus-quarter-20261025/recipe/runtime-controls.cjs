'use strict';
// GRANT ONLY because Node is excluded from the preflight allowance.
// Fake DB objects test JS bounds; these results never establish native safety.
const assert = require('node:assert/strict');
const path = require('node:path');
const runtime = require(path.join(process.argv[2], 'gitnexus-quarter-runtime.cjs'));
function fake() {
  class Database { constructor(...args) { this.args = args; } }
  class Connection {}
  return { Database, Connection };
}
for (const value of [undefined, '0', 'x', String(1024 ** 3)]) {
  process.env.GITNEXUS_LBUG_BUFFER_BYTES = value;
  process.env.GITNEXUS_LBUG_MAX_DB_SIZE = String(8 * 1024 ** 3);
  assert.throws(() => runtime(fake()), /database/);
}
process.env.GITNEXUS_LBUG_BUFFER_BYTES = String(512 * 1024 ** 2);
process.env.GITNEXUS_LBUG_MAX_DB_SIZE = String(8 * 1024 ** 3);
const lbug = fake();
runtime(lbug);
const db = new lbug.Database('fixture', 0, false, true, 0, true, 7, true, true);
assert.equal(db.constructor.name, 'Database');
assert.deepEqual(db.args, ['fixture', 512 * 1024 ** 2, false, true, 8 * 1024 ** 3, true, 7, true, true]);
assert.throws(() => new lbug.Database('fixture', 1024 ** 3), /exceeds/);
assert.throws(() => new lbug.Database('fixture', -1), /exceeds/);
assert.throws(() => new lbug.Database('fixture', 64 * 1024 ** 2, false, false, 9 * 1024 ** 3), /exceeds/);
const small = new lbug.Database('fixture', 64 * 1024 ** 2, false, false, 256 * 1024 ** 2);
assert.equal(small.args[1], 64 * 1024 ** 2);
assert.equal(small.args[4], 256 * 1024 ** 2);
console.log('PASS JS resource-bound controls only; native proof is separate');
