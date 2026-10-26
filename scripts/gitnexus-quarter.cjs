'use strict';
// Owned APP bootstrap, never an installed-package patch or a test target switch.
const path = require('node:path');
const { createRequire } = require('node:module');
const { spawnSync } = require('node:child_process');
const { pathToFileURL } = require('node:url');

async function main() {
  const [repo, cli, command, ...args] = process.argv.slice(2);
  const fixedCli = '/usr/local/share/npm-global/lib/node_modules/gitnexus/dist/cli/index.js';
  if (repo !== 'app' || cli !== fixedCli || command !== 'analyze') {
    throw new Error('invalid quarter bootstrap invocation');
  }
  // This synchronous subprocess reads THIS actual Node process and every task's
  // affinity before any CLI import, native DB load or analyze/index side effect.
  // The preload belongs only to this captured Node. Do not propagate it to
  // the guard's Python process or later GitNexus subprocesses. /proc/environ
  // records Node's original exec environment, which the guard verifies.
  const guardEnv = { ...process.env };
  delete guardEnv.LD_PRELOAD;
  const checked = spawnSync('/usr/bin/python3', ['-I', path.join(__dirname, 'gitnexus-guard.py'), '--repo', 'app'], { stdio: 'inherit', env: guardEnv });
  if (checked.error || checked.status !== 0) process.exit(64);
  delete process.env.LD_PRELOAD;
  const fromCli = createRequire(cli);
  const lbug = fromCli('@ladybugdb/core');
  require('./gitnexus-quarter-runtime.cjs')(lbug);
  process.argv = [process.execPath, cli, command, ...args];
  // /proc/cmdline is the immutable startup argv that the guard just checked.
  await import(pathToFileURL(cli).href);
}
main().catch(error => { console.error(`gitnexus quarter: denied: ${error.message}`); process.exitCode = 64; });
