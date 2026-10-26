'use strict';
const fs = require('node:fs');
const path = require('node:path');
const directory = process.argv[2];
fs.writeFileSync(path.join(directory, 'NODE-READY.json'), JSON.stringify({
  pid: process.pid, ppid: process.ppid,
  status: fs.readFileSync('/proc/self/status', 'utf8'),
}));
setInterval(() => {}, 1000);
