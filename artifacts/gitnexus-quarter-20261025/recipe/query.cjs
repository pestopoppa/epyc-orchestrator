'use strict';
// GRANT ONLY. Private read-only graph snapshot; no analyze command is accepted.
const path = require('node:path');
const { createRequire } = require('node:module');
const { pathToFileURL } = require('node:url');
const [scripts, command, ...args] = process.argv.slice(2);
const cli = '/usr/local/share/npm-global/lib/node_modules/gitnexus/dist/cli/index.js';
if (!['status', 'impact', 'context'].includes(command)) throw new Error('query command refused');
const lbug = createRequire(cli)('@ladybugdb/core');
process.env.GITNEXUS_LBUG_BUFFER_BYTES = String(64 * 1024 ** 2);
process.env.GITNEXUS_LBUG_MAX_DB_SIZE = String(2 * 1024 ** 3);
const BaseDatabase = lbug.Database;
// Bound even a graph read's buffer manager. This test-only override is not
// installed into GNX or exposed from the production wrapper.
lbug.Database = class Database extends BaseDatabase {
  constructor(databasePath, _buffer, compression, readOnly, _max, ...rest) {
    super(databasePath, 64 * 1024 ** 2, compression, true, 2 * 1024 ** 3, ...rest);
  }
};
// Install the production bounds after the test-side read-only class. The
// historical snapshot is 438,747,136 bytes and cannot use the empty-DB 256 MiB
// fixture file limit. Its graph-read bound remains the reviewed v2 2 GiB.
require(path.join(scripts, 'gitnexus-quarter-runtime.cjs'))(lbug);
delete process.env.LD_PRELOAD;
process.argv = [process.execPath, cli, command, ...args];
import(pathToFileURL(cli).href).catch(error => { console.error(error); process.exitCode = 1; });
