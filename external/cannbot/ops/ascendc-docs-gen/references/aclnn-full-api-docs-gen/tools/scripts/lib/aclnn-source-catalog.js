// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
'use strict';

const fs = require('fs');
const path = require('path');
const { createHash } = require('crypto');
const { execFileSync } = require('child_process');
const { parseAclnnSignaturesFromContent, parseParameter } = require('../extract-prototype.js');

function identity(file) {
    const real = fs.realpathSync(file);
    return process.platform === 'win32' ? real.toLowerCase() : real;
}

function git(root, ...args) {
    try {
        return execFileSync('git', ['-C', root, ...args], { encoding: 'utf8', windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] }).trim();
    } catch {
        throw new Error(`无法核验源码 Git 修订：${root}`);
    }
}

function sourceFiles(dir) {
    if (!fs.existsSync(dir)) return [];
    return fs.readdirSync(dir, { withFileTypes: true }).flatMap(entry => {
        const file = path.join(dir, entry.name);
        if (entry.isDirectory()) return sourceFiles(file);
        return entry.isFile() && /\.(?:h|hpp|hh|c|cc|cpp|cxx)$/i.test(file) ? [file] : [];
    }).sort();
}

function externalSources(operatorDir, options) {
    if (!options.prototypeManifest) return [];
    const manifestFile = path.resolve(options.prototypeManifest);
    const manifestDir = path.dirname(manifestFile);
    const manifest = JSON.parse(fs.readFileSync(manifestFile, 'utf8'));
    if (manifest.schemaVersion !== 1 || !Array.isArray(manifest.entries) || !manifest.entries.length) {
        throw new Error('原型来源清单必须包含 schemaVersion: 1 和非空 entries');
    }
    const entries = manifest.entries.filter(entry => {
        if (!entry || typeof entry.operatorDir !== 'string' || !entry.operatorDir) {
            throw new Error('原型来源条目缺少 operatorDir');
        }
        return identity(path.resolve(manifestDir, entry.operatorDir)) === identity(operatorDir);
    });
    if (!entries.length) return [];
    const sourceRoot = git(operatorDir, 'rev-parse', '--show-toplevel');
    if (options.projectRoot && identity(git(options.projectRoot, 'rev-parse', '--show-toplevel')) !== identity(sourceRoot)) {
        throw new Error('原型来源的项目根目录与算子源码仓不一致');
    }
    const revision = git(sourceRoot, 'rev-parse', 'HEAD');
    // Metadata binds an externally supplied build result to a tracked source revision.
    // It does not attest who built the header or prove binary compatibility.
    if (git(sourceRoot, 'status', '--porcelain', '--untracked-files=no')) {
        throw new Error('外部原型来源要求源码工作树无已跟踪修改，无法按 HEAD 核验当前源码');
    }
    const sources = [];
    for (const entry of entries) {
        if (!['generated', 'installed'].includes(entry.kind) || entry.sourceRevision !== revision) {
            throw new Error('原型来源 kind 无效或 sourceRevision 与当前源码修订不一致');
        }
        if (!Array.isArray(entry.interfaces) || !entry.interfaces.length ||
            entry.interfaces.some(name => typeof name !== 'string' || !/^aclnn[A-Za-z0-9_]+$/.test(name) || name.endsWith('GetWorkspaceSize'))) {
            throw new Error('原型来源 interfaces 必须列出实际目标接口基础名');
        }
        if (!Array.isArray(entry.headers) || !entry.headers.length) throw new Error('原型来源缺少 headers');
        for (const header of entry.headers) {
            if (!header || typeof header.path !== 'string' || !/\.(?:h|hpp|hh)$/i.test(header.path) ||
                typeof header.sha256 !== 'string' || !/^[a-f0-9]{64}$/i.test(header.sha256)) {
                throw new Error('生成或安装头文件必须提供 path 和 SHA-256');
            }
            const file = fs.realpathSync(path.resolve(manifestDir, header.path));
            const bytes = fs.readFileSync(file);
            const sha256 = createHash('sha256').update(bytes).digest('hex');
            if (sha256 !== header.sha256.toLowerCase()) throw new Error(`原型头文件摘要不一致：${file}`);
            sources.push({ file, sha256, content: bytes.toString('utf8'), kind: entry.kind,
                interfaces: entry.interfaces, sourceRevision: revision, sourceRoot,
                cannVersion: typeof entry.cannVersion === 'string' ? entry.cannVersion : null,
                provenance: 'supplied-metadata-revision-and-digest-matched' });
        }
    }
    for (const entry of entries) {
        for (const name of entry.interfaces) {
            const found = sources.some(source => source.interfaces.includes(name) &&
                parseAclnnSignaturesFromContent(source.content).some(sig => sig.name === name));
            if (!found) throw new Error(`来源清单声明的目标接口未在头文件中找到：${name}`);
        }
    }
    return sources;
}

function signatureTypeKey(sig) {
    return JSON.stringify(sig.parameters.map(param => {
        const parsed = parseParameter(param);
        return (parsed.type + parsed.pointer).replace(/\s*\*\s*/g, '*');
    }));
}

function collectAclnnSources(operatorDir, options = {}) {
    const local = ['op_api', 'op_host'].flatMap(dir => sourceFiles(path.join(operatorDir, dir))).map(file => {
        const bytes = fs.readFileSync(file);
        return { file: fs.realpathSync(file), content: bytes.toString('utf8'), kind: 'source',
            sha256: createHash('sha256').update(bytes).digest('hex') };
    });
    const sources = [...externalSources(operatorDir, options), ...local];
    const signatures = new Map();
    for (const source of sources) {
        for (const sig of parseAclnnSignaturesFromContent(source.content)) {
            const base = sig.name.replace(/GetWorkspaceSize$/, '');
            if (source.interfaces && !source.interfaces.includes(base)) continue;
            const previous = signatures.get(sig.name);
            if (previous && signatureTypeKey(previous) !== signatureTypeKey(sig)) {
                throw new Error(`同名 ACLNN 原型冲突：${sig.name}；${previous.file}；${source.file}`);
            }
            if (!previous) signatures.set(sig.name, { ...sig, file: source.file, sourceKind: source.kind });
        }
    }
    return { interfaces: [...new Set([...signatures.keys()].map(name => name.replace(/GetWorkspaceSize$/, '')))].sort(),
        signatures: [...signatures.values()], sources: sources.map(({ content, ...source }) => source) };
}

function validatePrototypeSnapshot(plan, prototypeReport) {
    if (!plan.prototypeReportSha256) return;
    if (createHash('sha256').update(prototypeReport).digest('hex') !== plan.prototypeReportSha256) {
        throw new Error('原型报告与预处理来源快照不一致，请重新预处理和审阅');
    }
    for (const source of plan.prototypeSources) {
        if (createHash('sha256').update(fs.readFileSync(source.file)).digest('hex') !== source.sha256) {
            throw new Error(`原型来源文件已改变，请重新预处理：${source.file}`);
        }
        if (source.kind !== 'source' && (git(source.sourceRoot, 'rev-parse', 'HEAD') !== source.sourceRevision ||
            git(source.sourceRoot, 'status', '--porcelain', '--untracked-files=no'))) {
            throw new Error('原型来源对应的源码修订或工作树已改变，请重新预处理');
        }
    }
}

module.exports = { collectAclnnSources, validatePrototypeSnapshot };
