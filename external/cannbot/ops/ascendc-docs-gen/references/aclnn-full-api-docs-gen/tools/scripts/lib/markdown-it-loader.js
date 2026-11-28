// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
'use strict';

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { spawnSync } = require('node:child_process');
const dependency = require('./markdown-it-dependency.json');

function cacheDirectory() {
    const root = process.env.CANNDOC_DEPENDENCY_CACHE || path.join(os.homedir(), '.cache', 'canndoc-api-docs-gen');
    return path.join(path.resolve(root), dependency.name, dependency.version);
}

function verifiedFile(file, expected) {
    try {
        return createHash('sha256').update(fs.readFileSync(file)).digest('hex') === expected;
    } catch {
        return false;
    }
}

function validCache(directory) {
    return dependency.files.every(file => verifiedFile(path.join(directory, file.name), file.sha256));
}

async function downloadFile(file) {
    const response = await fetch(file.url, { signal: AbortSignal.timeout(10000), redirect: 'error' });
    if (!response.ok) throw new Error(`下载失败 HTTP ${response.status}: ${file.url}`);
    return Buffer.from(await response.arrayBuffer());
}

async function prepareDependencyCache(directory = cacheDirectory(), download = downloadFile) {
    if (validCache(directory)) return directory;
    fs.mkdirSync(directory, { recursive: true });
    const staging = fs.mkdtempSync(path.join(directory, '.download-'));
    try {
        for (const file of dependency.files) {
            const bytes = await download(file);
            if (createHash('sha256').update(bytes).digest('hex') !== file.sha256) {
                throw new Error(`CDN 文件摘要不匹配: ${file.url}`);
            }
            fs.writeFileSync(path.join(staging, file.name), bytes, { flag: 'wx' });
        }
        // Publish only verified files. Concurrent preparations may complete the same cache.
        for (const file of dependency.files) {
            const target = path.join(directory, file.name);
            if (verifiedFile(target, file.sha256)) continue;
            try {
                fs.renameSync(path.join(staging, file.name), target);
            } catch (error) {
                if (!verifiedFile(target, file.sha256)) throw error;
            }
        }
        if (!validCache(directory)) throw new Error('Markdown 依赖缓存校验失败');
        return directory;
    } finally {
        for (const file of dependency.files) fs.rmSync(path.join(staging, file.name), { force: true });
        fs.rmdirSync(staging);
    }
}

function loadMarkdownIt() {
    const directory = cacheDirectory();
    if (!validCache(directory)) {
        // Keep existing synchronous validators; the child performs bounded asynchronous I/O.
        const result = spawnSync(process.execPath, [__filename, '--prepare'], {
            encoding: 'utf8', timeout: 35000, windowsHide: true
        });
        if (!validCache(directory)) {
            const reason = result.error?.message || result.stderr?.trim() || `退出码 ${result.status}`;
            throw new Error(`无法准备 Markdown 解析依赖，文档检查未执行：${reason}。请检查网络或 CANNDOC_DEPENDENCY_CACHE。`);
        }
    }
    return require(path.join(directory, dependency.files[0].name));
}

if (require.main === module) {
    if (process.argv.length !== 3 || process.argv[2] !== '--prepare') {
        console.error('用法: node markdown-it-loader.js --prepare');
        process.exitCode = 1;
    } else {
        prepareDependencyCache().then(directory => console.log(`Markdown 依赖已校验: ${directory}`)).catch(error => {
            console.error(error.message);
            process.exitCode = 1;
        });
    }
}

module.exports = { cacheDirectory, loadMarkdownIt, prepareDependencyCache };
