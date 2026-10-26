'use strict';
// GRANT ONLY: real native DB, private scratch files, bounded allocation.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { createRequire } = require('node:module');
const [scripts, output] = process.argv.slice(2);
const cli = '/usr/local/share/npm-global/lib/node_modules/gitnexus/dist/cli/index.js';
// Test-side resource choice. The production writer uses its separately fixed
// 512 MiB/8 GiB settings; this bounded fixture never runs an index.
process.env.GITNEXUS_LBUG_BUFFER_BYTES = String(64 * 1024 ** 2);
process.env.GITNEXUS_LBUG_MAX_DB_SIZE = String(256 * 1024 ** 2);
const lbug = createRequire(cli)('@ladybugdb/core');
require(path.join(scripts, 'gitnexus-quarter-runtime.cjs'))(lbug);
const receipts = [];

function parseCpus(text) {
  return text.trim().split(',').flatMap(part => {
    const ends = part.split('-').map(Number);
    return Array.from({ length: ends.at(-1) - ends[0] + 1 }, (_, i) => ends[0] + i);
  }).sort((a, b) => a - b);
}
function placement(label) {
  const expected = parseCpus(process.env.FILLER_CPUS);
  const tids = fs.readdirSync('/proc/self/task');
  const masks = tids.map(tid => {
    const status = fs.readFileSync(`/proc/self/task/${tid}/status`, 'utf8');
    const mask = status.match(/^Cpus_allowed_list:\s*(\S+)/m)[1];
    return { tid: Number(tid), mask };
  });
  const receipt = { label, pid: process.pid, task_count: tids.length, tasks: masks };
  receipts.push(receipt);
  // Save the observed failure values before enforcing the unchanged gates.
  fs.writeFileSync(path.join(output, 'placement-latest.json'), JSON.stringify(receipt, null, 2));
  assert(tids.length <= expected.length, 'native DB exceeded granted CPU thread count');
  for (const task of masks) assert.deepEqual(parseCpus(task.mask), expected, `off-quarter native thread ${task.tid}`);
  return receipt;
}
async function capReadback(conn, label) {
  const result = await conn.query("CALL current_setting('threads') RETURN *;");
  try {
    const rows = await result.getAll();
    assert(rows.length === 1, 'missing actual native thread setting');
    const values = Object.values(rows[0]).map(value => String(value));
    assert(values.includes('1'), `native query cap differs: ${JSON.stringify(values)}`);
    const getter = typeof conn._connection.getMaxNumThreadForExec === 'function'
      ? conn._connection.getMaxNumThreadForExec() : null;
    if (getter !== null) assert.equal(Number(getter), 1);
    return { label, rows, native_getter: getter, placement: placement(label) };
  } finally { result.close(); }
}
async function main() {
  fs.mkdirSync(output, { recursive: true });
  placement('startup');
  const shim = path.join(scripts, 'gitnexus-lbug-cap.so');
  assert.equal(process.env.LD_PRELOAD, shim, 'not the exact owned preload');
  assert(fs.readFileSync('/proc/self/maps', 'utf8').split('\n').some(line => line.endsWith(` ${shim}`)), 'native cap not actually mapped');
  // Constructor args are supported by the installed database.js and d.ts.
  // Never use 0: it chooses a host-sized buffer pool even for an empty DB.
  const db = new lbug.Database(path.join(output, 'native-db'), 64 * 1024 ** 2,
    false, false, 256 * 1024 ** 2, true, 64 * 1024 ** 2, true, true);
  let asyncConn;
  let syncConn;
  try {
    await db.init();
    placement('database-init');
    // Keep the initialized DB/pool resident across an external supervisor poll.
    await new Promise(resolve => setTimeout(resolve, 300));
    assert.equal(db.constructor.name, 'Database');
    asyncConn = new lbug.Connection(db, 192);
    assert.equal(asyncConn.constructor.name, 'Connection');
    await asyncConn.init();
    receipts.push(await capReadback(asyncConn, 'async-init'));
    asyncConn.setMaxNumThreadForExec(192);
    receipts.push(await capReadback(asyncConn, 'oversized-public-setter'));
    syncConn = new lbug.Connection(db, 192);
    syncConn.initSync();
    receipts.push(await capReadback(syncConn, 'sync-init'));
    // A real data query also exercises QueryResult's constructor.name invariant.
    const data = await syncConn.query('RETURN 7 AS fixture;');
    try { assert.equal(Number((await data.getAll())[0].fixture), 7); }
    finally { data.close(); }
  } finally {
    if (syncConn) await syncConn.close();
    if (asyncConn) await asyncConn.close();
    await db.close();
    placement('database-close');
  }
  // Exercise the separate synchronous Database initialization route as well.
  const syncDb = new lbug.Database(path.join(output, 'native-sync-db'), 0,
    false, false, 0, true, 64 * 1024 ** 2, true, true);
  let secondConn;
  try {
    syncDb.initSync();
    placement('sync-database-init');
    await new Promise(resolve => setTimeout(resolve, 300));
    secondConn = new lbug.Connection(syncDb, 192);
    secondConn.initSync();
    receipts.push(await capReadback(secondConn, 'sync-database-query'));
  } finally {
    if (secondConn) await secondConn.close();
    await syncDb.close();
    placement('sync-database-close');
  }
  fs.writeFileSync(path.join(output, 'native-db-proof.json'), JSON.stringify({
    status: 'PASS', buffer_manager_bytes: 64 * 1024 ** 2,
    max_db_bytes: 256 * 1024 ** 2, native_getter_policy: 'SQL current_setting is required; direct Node getter recorded if exposed',
    receipts,
  }, (_, value) => typeof value === 'bigint' ? value.toString() : value, 2));
}
main().catch(error => {
  console.error(error);
  fs.mkdirSync(output, { recursive: true });
  fs.writeFileSync(path.join(output, 'native-db-failure.json'), JSON.stringify({ status: 'FAIL', error: String(error), receipts }, (_, value) => typeof value === 'bigint' ? value.toString() : value, 2));
  process.exitCode = 1;
});
