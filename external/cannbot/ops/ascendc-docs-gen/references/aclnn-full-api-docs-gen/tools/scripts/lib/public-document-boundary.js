// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
'use strict';

const path = require('node:path');
const fs = require('node:fs');
const { fileURLToPath, pathToFileURL } = require('node:url');
const { documentProse, mapResourceLinks } = require('./markdown-resource-links.js');

// Match whole local-root components, independently of prose punctuation and file suffixes.
const LOCAL_UNIX_PATH = /(?<![\p{L}\p{N}_./\\~-])\/(?:home|root|Users|private|tmp|var\/tmp|data|mnt|workspace)(?![\p{L}\p{N}_.-])/mu;

function absolutePathKind(value) {
    if (/^\/?[A-Za-z]:[\\/]/.test(value) || /^\\\\/.test(value)) return 'windows';
    return value.startsWith('/') ? 'unix' : null;
}

function normalizedAbsolute(value) {
    const kind = absolutePathKind(value);
    if (!kind) return null;
    const text = value.replace(/^\/(?=[A-Za-z]:[\\/])/, '').replace(/\\/g, '/');
    const normalized = path.posix.normalize(text).replace(/\/+$/, '');
    return kind === 'windows' ? normalized.toLowerCase() : normalized;
}

function isRegisteredAbsolute(value, roots) {
    const normalized = normalizedAbsolute(value);
    return normalized !== null && roots.some(root => {
        const base = normalizedAbsolute(String(root));
        return base && base.length > 1 && (normalized === base || normalized.startsWith(base + '/'));
    });
}

function assertLocalTarget(value, roots) {
    const kind = absolutePathKind(value);
    if (kind === 'windows' || (kind === 'unix' && LOCAL_UNIX_PATH.test(value))) {
        throw new Error('最终文档资源包含本机绝对路径');
    }
    if (isRegisteredAbsolute(value, roots)) throw new Error('最终文档包含当前代码仓的本机绝对路径');
}

// These are workflow artifacts, not names of arbitrary user documentation.
const STAGE_ARTIFACTS = {
    aclnn: ['0_extract_plan.json', '0_prototype.md', '0_cross_dir.md', '0_chip_evidence.md',
        '0_error_codes.md', '0_call_example.md', '1_extract.md', '2_compare.md', '3_formula.md',
        '4_generate_skeleton.md', '4_generate.md', '4_review_input.json', '5_review_patch.json', '5_review.md'],
    ascendc: ['manifest.json', 'code-evidence.md', 'reference-evidence.md', 'resolution.md',
        'temporary-api-relation-plan.json', 'temporary-code-reading-plan.json', '3_generate.md', '4_review.md']
};

function escapeRegex(value) {
    return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function withoutMath(prose) {
    // Remove closed TeX spans only for ID classification; paths and artifacts remain checked.
    return prose.replace(/\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|(?<![\\$])\$(?!\$)(?:\\.|[^$\r\n])+?(?<!\\)\$(?!\$)/g,
        match => match.replace(/[^\r\n]/g, ' '));
}

function resourceIdentity(file) {
    let resolved = path.resolve(file);
    try { resolved = fs.realpathSync.native(resolved); } catch { /* A not-yet-created target still has a lexical identity. */ }
    return process.platform === 'win32' ? resolved.toLowerCase() : resolved;
}

function fileUrlPath(target, warn) {
    let url;
    try { url = new URL(target, 'file:///'); }
    catch { warn('需审阅无法解析的文件 URL：' + target); return null; }
    if (url.hostname && url.hostname.toLowerCase() !== 'localhost') {
        throw new Error('最终文档包含网络共享路径');
    }
    try { return decodeURIComponent(url.pathname); }
    catch { warn('需审阅无法解析的文件 URL：' + target); return null; }
}

function assertPathText(document, localRoots, warn) {
    // A path-shaped URL component is not a local filesystem reference. Preserve offsets for diagnostics.
    const pathText = document.replace(/\bhttps?:\/\/[^\s<>"'`]+/giu, value => ' '.repeat(value.length));
    const normalized = pathText.replace(/\\/g, '/');
    const checks = [
        ['本机 Windows 路径', /\b[A-Za-z]:[\\/][^\s<>]+/u, pathText],
        ['本机 Unix 路径', LOCAL_UNIX_PATH, pathText],
        ['网络共享路径', /\\\\[^\\\s]+\\[^\\\s]+/u, pathText]
    ];
    for (const [reason, pattern, content = document] of checks) {
        const match = pattern.exec(content);
        if (match) throw new Error(`最终文档包含${reason}（第 ${document.slice(0, match.index).split('\n').length} 行）`);
    }
    // Markdown renderers may leave file URLs as plain text rather than clickable resources.
    for (const match of pathText.matchAll(/\bfile:[^\s<>"'`]+/giu)) {
        const targetPath = fileUrlPath(match[0], warn);
        if (targetPath !== null) assertLocalTarget(targetPath, localRoots);
    }
    for (const root of localRoots) {
        const value = String(root).replace(/\\/g, '/').replace(/\/+$/, '');
        const pattern = new RegExp('(?<![\\p{L}\\p{N}_./\\\\~-])' + escapeRegex(value) +
            '(?=$|/|[^\\p{L}\\p{N}_.~-])', process.platform === 'win32' ? 'imu' : 'mu');
        if (value.length > 1 && pattern.test(normalized)) {
            throw new Error('最终文档包含当前代码仓的本机绝对路径');
        }
    }
    return pathText;
}

function validatePublicDocumentBoundary(document, {
    localRoots = [], evidenceIds = [], workflow, internalArtifacts = [], documentPath
} = {}) {
    const warnings = new Set();
    const warn = value => warnings.add(value);
    const pathText = assertPathText(document, localRoots, warn);
    const prose = documentProse(document);
    assertPathText(prose, localRoots, warn);
    const architecturePattern = /\b(?:__NPU_ARCH__|__CCE_AICORE__|DAV_[A-Za-z0-9_]+)\b/u;
    if (architecturePattern.test(document) || architecturePattern.test(prose)) throw new Error('最终文档包含内部架构标识');
    const ordinaryProse = withoutMath(prose);
    const knownIds = new Set(evidenceIds);
    for (const match of prose.matchAll(/\b(?:K|C|S|R|D)-\d+\b/gu)) {
        const inMath = ordinaryProse.slice(match.index, match.index + match[0].length).trim() === '';
        if (knownIds.has(match[0]) && !inMath) {
            throw new Error('最终文档包含内部证据 ID：' + match[0]);
        }
        if (knownIds.has(match[0]) || !inMath) warnings.add('需审阅编号含义：' + match[0] + '（未据字符形式判定为内部引用）');
    }
    const internalPaths = new Set(internalArtifacts.map(resourceIdentity));
    const text = mapResourceLinks(document,
        target => {
            if (!target || target.startsWith('#')) return;
            if (/^[A-Za-z][A-Za-z\d+.-]*:/.test(target) && !/^file:/i.test(target) && !/^[A-Za-z]:[\\/]/.test(target)) return;
            let targetPath;
            if (/^file:/i.test(target)) {
                targetPath = fileUrlPath(target, warn);
                if (targetPath === null) return;
            } else {
                try { targetPath = decodeURIComponent(target.split(/[?#]/)[0]); }
                catch { warnings.add('需审阅无法解析的资源链接：' + target); return; }
            }
            assertLocalTarget(targetPath, localRoots);
            if (!documentPath) { warnings.add('缺少文档路径，需审阅资源目标：' + target); return; }
            let decoded;
            try {
                decoded = /^file:/i.test(target)
                    ? fileURLToPath(new URL(target, pathToFileURL(documentPath)))
                    : path.resolve(path.dirname(documentPath), decodeURIComponent(target.split(/[?#]/)[0]));
            }
            catch { warnings.add('需审阅无法解析的资源链接：' + target); return; }
            if (internalPaths.has(resourceIdentity(decoded))) {
                throw new Error('最终文档链接指向本次会话的内部阶段文件：' + target);
            }
        }, warning => warnings.add(warning));
    const artifacts = [...(STAGE_ARTIFACTS[workflow] || []), ...internalArtifacts.map(file => path.basename(file))];
    if (artifacts.length) {
        const artifactPattern = new RegExp('(?<![\\w.-])(?:' + artifacts.map(escapeRegex).join('|') + ')(?![\\w.-])', 'gu');
        for (const match of text.matchAll(artifactPattern)) warnings.add('需审阅同名阶段文件引用：' + match[0] + '（来源未确定）');
    }
    const productPattern = /\bAscend\s+910(?:[BCD]|_9[35])\w*\b/u;
    if (productPattern.test(pathText) || productPattern.test(prose)) {
        throw new Error('最终文档包含非规范产品标识');
    }
    return [...warnings];
}

function loadBoundaryContext(workflow, sessionDir, documentPath) {
    if (!Object.hasOwn(STAGE_ARTIFACTS, workflow)) throw new Error('未知文档工作流：' + workflow);
    const context = { workflow, documentPath, internalArtifacts: [], evidenceIds: [] };
    if (!sessionDir) return context;
    const names = fs.readdirSync(sessionDir, { withFileTypes: true })
        .filter(entry => entry.isFile() && (STAGE_ARTIFACTS[workflow].includes(entry.name) || /_FAILED\.md$/.test(entry.name))).map(entry => entry.name);
    context.internalArtifacts = names.map(name => path.resolve(sessionDir, name));
    const evidenceNames = new Set(['code-evidence.md', 'reference-evidence.md', 'resolution.md', '1_extract.md', '2_compare.md']);
    for (const name of names.filter(item => evidenceNames.has(item))) {
        const content = fs.readFileSync(path.resolve(sessionDir, name), 'utf8');
        // Read declared fact IDs, not every arithmetic expression occurring in evidence.
        for (const match of content.matchAll(/^[ \t]*(?:#{1,6}[ \t]+|[-*][ \t]+|\|[ \t]*)?(?:`|\*\*)?((?:K|C|S|R|D)-\d+)(?:`|\*\*)?(?=[ \t]*(?:[：:|—–]|$))/gm)) {
            context.evidenceIds.push(match[1]);
        }
    }
    context.evidenceIds = [...new Set(context.evidenceIds)];
    return context;
}

module.exports = { STAGE_ARTIFACTS, loadBoundaryContext, validatePublicDocumentBoundary };
