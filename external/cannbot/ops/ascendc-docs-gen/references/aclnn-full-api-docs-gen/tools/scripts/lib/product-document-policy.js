// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
'use strict';

const { codeBlockRanges } = require('./markdown-resource-links.js');

const PRODUCTS = [
    ['Ascend 950PR&950DT系列产品', '950'],
    ['Ascend 950PR系列产品', '950'],
    ['Ascend 950DT系列产品', '950'],
    ['Atlas A3系列产品', 'A3'],
    ['Atlas A3训练系列产品', 'A3'],
    ['Atlas A3推理系列产品', 'A3'],
    ['Atlas A2系列产品', '910b'],
    ['Atlas A2训练系列产品', '910b'],
    ['Atlas A2推理系列产品', '910b'],
    ['Atlas训练系列产品', '910'],
    ['Atlas推理系列产品AI Core', '310p'],
    ['Atlas推理系列产品Vector Core', '310p'],
    ['Atlas推理系列产品', '310p'],
    ['Atlas 200I/500 A2推理产品', '310b']
];
const NPU_VALUES = new Set(PRODUCTS.map(([, npu]) => npu));

function escapeRegex(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// Preserve source code and resource destinations; only visible prose is rewritten.
function markdownParts(content) {
    const codeRanges = codeBlockRanges(content);
    const parts = [];
    let start = 0;
    let index = 0;
    function emit(end, kind) {
        if (index > start) parts.push({ kind: 'text', text: content.slice(start, index), offset: start });
        parts.push({ kind, text: content.slice(index, end), offset: index });
        index = end;
        start = end;
    }
    while (index < content.length) {
        const codeRange = codeRanges.find(([begin]) => begin === index);
        if (codeRange) { emit(codeRange[1], 'code'); continue; }
        const rest = content.slice(index);
        if (index === 0 || content[index - 1] === '\n') {
            const fence = /^[ \t]*(`{3,}|~{3,})[^\r\n]*(?:\r?\n|$)/.exec(rest);
            if (fence) {
                const close = new RegExp('^[ \\t]*' + fence[1][0] + '{' + fence[1].length + ',}[ \\t]*\\r?$', 'gm');
                close.lastIndex = index + fence[0].length;
                const end = close.exec(content);
                emit(end ? end.index + end[0].length : content.length, 'code');
                continue;
            }
            const reference = /^[ \t]{0,3}\[[^\]\n]+\]:[^\n]*/.exec(rest);
            if (reference) { emit(index + reference[0].length, 'resource'); continue; }
        }
        if (rest.startsWith('<!--')) {
            const end = content.indexOf('-->', index + 4);
            emit(end < 0 ? content.length : end + 3, 'comment');
            continue;
        }
        const rawCode = /^<(pre|code)\b[^>]*>[\s\S]*?<\/\1\s*>/i.exec(rest);
        if (rawCode) { emit(index + rawCode[0].length, 'code'); continue; }
        if (rest[0] === '`') {
            const delimiter = /^`+/.exec(rest)[0];
            const endPattern = new RegExp('(?<!`)' + delimiter + '(?!`)', 'g');
            endPattern.lastIndex = index + delimiter.length;
            const end = endPattern.exec(content);
            if (end) { emit(end.index + delimiter.length, 'code'); continue; }
        }
        if (rest.startsWith('](')) {
            let depth = 1;
            let end = index + 2;
            for (; end < content.length && depth; end++) {
                if (content[end] === '\\') { end++; continue; }
                if (content[end] === '(') depth++;
                if (content[end] === ')') depth--;
            }
            if (!depth) { emit(end, 'resource'); continue; }
        }
        const html = /^(?:<\/?[A-Za-z][\w:-]*\b(?:"[^"]*"|'[^']*'|[^'">])*\/?>|<https?:\/\/[^>]+>)/.exec(rest);
        if (html) { emit(index + html[0].length, 'html'); continue; }
        if (rest[0] === '\\') { index += Math.min(2, rest.length); continue; }
        index++;
    }
    if (index > start) parts.push({ kind: 'text', text: content.slice(start), offset: start });
    return parts;
}

function markdownTableRanges(content) {
    const lines = content.split('\n');
    const offsets = [];
    let offset = 0;
    for (const line of lines) { offsets.push(offset); offset += line.length + 1; }
    const ranges = [];
    for (let i = 1; i < lines.length; i++) {
        const separator = lines[i].trim().replace(/^\|/, '').replace(/\|$/, '');
        if (!lines[i - 1].includes('|') || !lines[i].includes('|') ||
            !separator.split('|').every(cell => /^\s*:?-{3,}:?\s*$/.test(cell))) continue;
        let end = i + 1;
        while (end < lines.length && lines[end].trim() && lines[end].includes('|')) end++;
        ranges.push([offsets[i - 1], offsets[end] ?? content.length]);
        i = end - 1;
    }
    return ranges;
}

function normalizeProductNames(content, docType) {
    const aliases = [
        ['Ascend 950PR/Ascend 950DT', PRODUCTS[0][0]],
        ...['A3', 'A2'].flatMap(series => [
            [`Atlas ${series} 训练系列产品/Atlas ${series} 推理系列产品`, `Atlas ${series}系列产品`],
            [`Atlas ${series}训练系列产品/Atlas ${series}推理系列产品`, `Atlas ${series}系列产品`],
            [`Atlas ${series} 训练系列产品`, `Atlas ${series}训练系列产品`],
            [`Atlas ${series} 推理系列产品`, `Atlas ${series}推理系列产品`]
        ]),
        ['Ascend 950PR', 'Ascend 950PR系列产品'],
        ['Ascend 950DT', 'Ascend 950DT系列产品'],
        ['Atlas 训练系列产品', 'Atlas训练系列产品'],
        ['Atlas 200I/500 A2 推理产品', 'Atlas 200I/500 A2推理产品'],
        ['Atlas 推理系列产品AI Core', 'Atlas推理系列产品AI Core'],
        ['Atlas 推理系列产品Vector Core', 'Atlas推理系列产品Vector Core'],
        ['Atlas 推理系列产品', docType === 'ascendc'
            ? 'Atlas推理系列产品AI Core、Atlas推理系列产品Vector Core' : 'Atlas推理系列产品']
    ];
    // Match canonical names first as well: repeated normalization must be idempotent.
    const replacements = new Map([...PRODUCTS.map(([name]) => [name, name]), ...aliases]);
    const names = [...replacements.keys()].sort((a, b) => b.length - a.length);
    const pattern = new RegExp(names.map(escapeRegex).join('|'), 'g');
    return markdownParts(content).map(part => part.kind === 'text'
        ? part.text.replace(pattern, name => replacements.get(name)) : part.text).join('');
}

function supportProducts(docType) {
    return [
        PRODUCTS[0], PRODUCTS[3], PRODUCTS[6], PRODUCTS[13],
        ...(docType === 'ascendc' ? [PRODUCTS[10], PRODUCTS[11]] : [PRODUCTS[12]]),
        PRODUCTS[9]
    ];
}

function validateProductDocument(content, docType) {
    if (!['aclnn', 'ascendc'].includes(docType)) throw new Error('未知文档类型');
    if (normalizeProductNames(content, docType) !== content) {
        throw new Error('最终文档包含旧产品名称，请使用当前产品命名规范');
    }
    const parts = markdownParts(content);
    const prose = parts.map(part => ['code', 'resource'].includes(part.kind)
        ? part.text.replace(/[^\r\n]/g, ' ') : part.text).join('');
    const firstChapter = /^##\s+(.+)$/m.exec(prose);
    if (firstChapter?.[1].trim() !== '产品支持情况') {
        throw new Error('最终文档第1章必须为“产品支持情况”');
    }
    const support = /^## 产品支持情况[ \t]*\r?\n([\s\S]*?)(?=^## |$(?![\s\S]))/m.exec(prose)?.[1] || '';
    if (/<table\b|^\s*\|/im.test(support)) {
        throw new Error('产品支持情况必须使用无序列表，不能使用表格');
    }
    const productNames = new Map(PRODUCTS);
    const productPattern = new RegExp([...productNames.keys()].sort((a, b) => b.length - a.length).map(escapeRegex).join('|'), 'g');
    const usedIds = new Set();
    let active = null;
    const tableRanges = markdownTableRanges(prose);
    const supportStart = prose.indexOf(support, firstChapter.index + firstChapter[0].length);
    const supportEnd = supportStart + support.length;
    let tableDepth = 0;
    for (const part of parts) {
        if (part.kind === 'html') {
            if (/^<table\b/i.test(part.text)) tableDepth++;
            if (/^<\/table\s*>/i.test(part.text)) {
                if (!tableDepth) throw new Error('HTML 表格结束标签没有对应起始标签');
                tableDepth--;
            }
            continue;
        }
        if (part.kind === 'comment') {
            if (!/\bnpu\b|\bend\s+id/i.test(part.text)) continue;
            const start = /^<!-- npu="([^"]+)" id([1-9]\d*) -->$/.exec(part.text);
            const end = /^<!-- end id([1-9]\d*) -->$/.exec(part.text);
            if (start) {
                if (!NPU_VALUES.has(start[1])) throw new Error('产品标签 npu 值无效');
                if (active) throw new Error('产品标签不得嵌套或交叉');
                if (usedIds.has(start[2])) throw new Error('单篇文档中的产品标签 id 不得重复');
                usedIds.add(start[2]);
                active = { id: start[2], npu: start[1], hasContent: false };
            } else if (end) {
                if (!active || active.id !== end[1]) throw new Error('产品标签起止 id 不匹配');
                if (!active.hasContent) throw new Error('产品标签不得包裹空内容');
                active = null;
            } else {
                throw new Error('产品标签格式错误，必须严格使用 <!-- npu="值" id序号 --> / <!-- end id序号 -->');
            }
            continue;
        }
        if (active && part.text.trim()) active.hasContent = true;
        if (part.kind !== 'text' || tableDepth) continue;
        for (const match of part.text.matchAll(productPattern)) {
            const offset = part.offset + match.index;
            if (tableRanges.some(([start, end]) => offset >= start && offset < end)) continue;
            // The support list is explicitly product-specific. Other prose requires semantic review.
            if (!active && offset >= supportStart && offset < supportEnd) {
                throw new Error('产品支持列表缺少配对的产品标签');
            }
            if (active && productNames.get(match[0]) !== active.npu) {
                throw new Error('产品名称与 npu 标签不匹配');
            }
        }
    }
    if (tableDepth) throw new Error('HTML 表格未闭合');
    if (active) throw new Error('产品标签缺少结束注释');
    const items = support.replace(/<!--[\s\S]*?-->/g, '').replace(/<\/?term>/g, '')
        .split(/\r?\n/).map(line => line.trim()).filter(Boolean);
    const expected = supportProducts(docType);
    if (items.length !== expected.length || items.some((line, index) =>
        !new RegExp(`^- ${escapeRegex(expected[index][0])}[：:]\\s*(?:支持|不支持|√|×)$`).test(line))) {
        throw new Error('产品支持情况列表的产品名称、顺序或支持状态不符合模板');
    }
}

module.exports = {
    markdownParts, normalizeProductNames, supportProducts, validateProductDocument,
    productNames: PRODUCTS.map(([name]) => name)
};
