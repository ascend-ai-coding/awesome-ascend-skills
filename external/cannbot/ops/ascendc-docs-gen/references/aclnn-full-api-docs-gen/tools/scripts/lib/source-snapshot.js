// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { createHash } = require('node:crypto');

function identity(file) {
    const value = path.resolve(file);
    return process.platform === 'win32' ? value.toLowerCase() : value;
}

function digest(file) {
    return createHash('sha256').update(fs.readFileSync(file)).digest('hex');
}

function validateSourceSnapshot(snapshot, requiredFiles = []) {
    if (!Array.isArray(snapshot) || !snapshot.length) {
        throw new Error('缺少取证时的源码快照，请返回取证阶段；不得在最终化时补记当前摘要');
    }
    const recorded = new Set();
    for (const entry of snapshot) {
        if (!entry || typeof entry.file !== 'string' || !/^[a-f0-9]{64}$/.test(entry.sha256 || '')) {
            throw new Error('源码快照条目无效');
        }
        const key = identity(entry.file);
        if (recorded.has(key)) throw new Error('源码快照存在重复文件：' + entry.file);
        recorded.add(key);
        if (entry.realPath && identity(fs.realpathSync(entry.file)) !== identity(entry.realPath)) {
            throw new Error('源码证据文件目标已变化，请重新取证和审阅：' + entry.file);
        }
        if (digest(entry.file) !== entry.sha256) throw new Error('源码证据文件已变化，请重新取证和审阅：' + entry.file);
    }
    for (const file of requiredFiles) {
        if (!recorded.has(identity(file))) throw new Error('源码处理记录缺少取证时摘要，请先登记再核验：' + file);
    }
}

function captureSourceSnapshot(files, previous = []) {
    if (previous.length) validateSourceSnapshot(previous);
    const entries = new Map(previous.map(entry => [identity(entry.file), entry]));
    for (const file of files) {
        const key = identity(file);
        if (!entries.has(key)) entries.set(key, { file: path.resolve(file), realPath: fs.realpathSync(file), sha256: digest(file) });
    }
    return [...entries.values()].sort((a, b) => a.file.localeCompare(b.file));
}

module.exports = { captureSourceSnapshot, validateSourceSnapshot };
