---
title: GFW Causes ElectronJS Installation Failure
layout: post
summary: npm install electron will fail if you're unable to access github (or simply forget to use a proxy).
category: Coding
author: Sida Liu (with the help of ChatGPT)
---
If you experience something like this when installing `electron`:

```bash
~/code/try-electron$ npm install electron
npm ERR! code 1
npm ERR! path /home/liusida/code/try-electron/node_modules/electron
npm ERR! command failed
npm ERR! command sh -c node install.js
npm ERR! RequestError: Socket connection timeout
npm ERR!     at ClientRequest.<anonymous> (/home/liusida/code/try-electron/node_modules/got/dist/source/core/index.js:970:111)
npm ERR!     at Object.onceWrapper (node:events:626:26)
npm ERR!     at ClientRequest.emit (node:events:523:35)
npm ERR!     at origin.emit (/home/liusida/code/try-electron/node_modules/@szmarczak/http-timer/dist/source/index.js:43:20)
npm ERR!     at TLSSocket.socketErrorListener (node:_http_client:495:9)
npm ERR!     at TLSSocket.emit (node:events:511:28)
npm ERR!     at emitErrorNT (node:internal/streams/destroy:151:8)
npm ERR!     at emitErrorCloseNT (node:internal/streams/destroy:116:3)
npm ERR!     at process.processTicksAndRejections (node:internal/process/task_queues:82:21)
npm ERR!     at new NodeError (node:internal/errors:399:5)
npm ERR!     at internalConnectMultiple (node:net:1099:20)
npm ERR!     at Timeout.internalConnectMultipleTimeout (node:net:1638:3)
npm ERR!     at listOnTimeout (node:internal/timers:575:11)
npm ERR!     at process.processTimers (node:internal/timers:514:7)

npm ERR! A complete log of this run can be found in: /home/liusida/.npm/_logs/2023-06-01T12_35_24_835Z-debug-0.log
```

Or something like this:
```bash
PS C:\code\try-electron> npm install electron
RequestError: connect ETIMEDOUT 20.205.243.166:443
    at ClientRequest.<anonymous> (C:\code\try-electron\node_modules\got\dist\source\core\index.js:970:111)
    at Object.onceWrapper (node:events:628:26)
    at ClientRequest.emit (node:events:525:35)
    at origin.emit (C:\code\try-electron\node_modules\@szmarczak\http-timer\dist\source\index.js:43:20)
    at TLSSocket.socketErrorListener (node:_http_client:502:9)
    at TLSSocket.emit (node:events:513:28)
    at emitErrorNT (node:internal/streams/destroy:151:8)
    at emitErrorCloseNT (node:internal/streams/destroy:116:3)
    at process.processTicksAndRejections (node:internal/process/task_queues:82:21)
    at TCPConnectWrap.afterConnect [as oncomplete] (node:net:1494:16)
```

Remember, it is not your fault! It is due to the GFW!

ElectronJS only stores some scripts on npmjs's server. When executin `npm install electron`, it first fetches those scripts. The `package.json` file defines `"postinstall": "node install.js"`. And in `install.js`, it downloads artifacts (a .zip file) from GitHub (the url, for example, is like ) and extract it into a subfolder called `dist`. If any problem arises, npm will delete all the folders, leaving you without an opportunity to manually rectify this issue.

To address this, you can use `npm install electron --ignore-scripts` to prevent `npm` from calling `node install.js`. Then, you can try using `node ./node_modules/electron/install.js` to see if there's any luck. Otherwise, try to download `'https://github.com/electron/electron/releases/download/v25.0.1/electron-v25.0.1-linux-x64.zip'` or `https://github.com/electron/electron/releases/download/v25.0.1/electron-v25.0.1-win32-x64.zip` manually, and extract them into a folder called `dist`, and copy that folder into the `electron` folder as `./node_modules/electron/dist/`. Finally, create a `path.txt` in the folder `.node_modules/electron/`, the content of the `path.txt` is simply the name of the electron executable, such as `electron.exe` for Windows and `electron` for Linux.

Hope this helps.

P.S. An even better solution was found here on [StackOverflow](https://stackoverflow.com/questions/60054531/how-can-i-solve-the-connection-problem-during-npm-install-behind-a-proxy) by using an additional package `cross-env` like this: `npx cross-env ELECTRON_GET_USE_PROXY=true GLOBAL_AGENT_HTTPS_PROXY=http://127.0.0.1:7890 npm install`. Cheers!
