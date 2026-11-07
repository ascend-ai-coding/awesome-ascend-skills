// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
'use strict';

const { loadMarkdownIt } = require('./markdown-it-loader.js');
let parser;

function getParser() {
    if (!parser) {
        const MarkdownIt = loadMarkdownIt();
        parser = new MarkdownIt({ html: true, linkify: false, typographer: false });
    }
    return parser;
}

function codeBlockRanges(source) {
    const offsets = [0];
    for (const match of source.matchAll(/\n/g)) offsets.push(match.index + 1);
    return getParser().parse(source, {}).filter(token => ['fence', 'code_block'].includes(token.type) && token.map)
        .map(token => [offsets[token.map[0]], offsets[token.map[1]] ?? source.length]);
}

function codeBlocks(source) {
    return getParser().parse(source, {}).filter(token => ['fence', 'code_block'].includes(token.type))
        .map(token => ({ language: token.info.trim().split(/\s+/)[0].toLowerCase(), content: token.content }));
}

function mapOutsideCodeBlocks(source, transform) {
    let cursor = 0;
    let result = '';
    for (const [begin, end] of codeBlockRanges(source)) {
        result += transform(source.slice(cursor, begin)) + source.slice(begin, end);
        cursor = end;
    }
    return result + transform(source.slice(cursor));
}

function documentHeadings(source) {
    const tokens = getParser().parse(source, {});
    return tokens.flatMap((token, index) => token.type === 'heading_open'
        ? [{ level: Number(token.tag.slice(1)), title: tokens[index + 1].content, lines: token.map }] : []);
}

function documentProse(source) {
    return getParser().parse(source, {}).filter(token => token.type === 'inline' || token.type === 'html_block')
        .map(token => htmlAttribute(token.content)).join('\n');
}

function htmlAttribute(value) {
    // Decode HTML entities without treating literal HTML backslashes as Markdown escapes.
    return getParser().utils.unescapeAll(value.replace(/\\/g, '&#92;'));
}

function inspectSourceSet(value, inspect, warn) {
    let remaining = value;
    while ((remaining = remaining.replace(/^[\s,]+/, ''))) {
        const url = /^\S+/.exec(remaining)[0];
        remaining = remaining.slice(url.length);
        inspect(getParser().normalizeLink(url.replace(/,+$/, '')));
        if (url.endsWith(',')) continue;
        const end = remaining.indexOf(',');
        const descriptor = (end < 0 ? remaining : remaining.slice(0, end)).trim();
        if (descriptor && !/^(?:\d+w|(?:\d+(?:\.\d+)?|\.\d+)x)$/.test(descriptor)) {
            warn('需审阅无法完整解析的 HTML srcset 描述符');
        }
        remaining = end < 0 ? '' : remaining.slice(end + 1);
    }
}

function inspectHtml(source, inspect, warn) {
    let html = source.replace(/<!--[\s\S]*?-->/g, '');
    html = html.replace(/<(script|style|textarea|xmp|iframe|noembed|noframes)\b[\s\S]*?<\/\1\s*>/gi,
        () => { warn('需审阅原始 HTML 容器中的资源，未执行其中内容'); return ''; });
    if (/<(?:script|style|textarea|xmp|iframe|noembed|noframes)\b/i.test(html)) {
        warn('需审阅未闭合的原始 HTML 容器');
        return '';
    }
    const tagPattern = /<\/?[A-Za-z][A-Za-z\d:-]*(?:[^"'<>]|"[^"]*"|'[^']*')*>/g;
    let text = '';
    let cursor = 0;
    let linkDepth = 0;
    for (const match of html.matchAll(tagPattern)) {
        if (!linkDepth) text += html.slice(cursor, match.index);
        const tag = /^<([A-Za-z][A-Za-z\d:-]*)\b/.exec(match[0]);
        if (tag) {
            // HTML keeps the first occurrence of a repeated attribute.
            const attributes = new Map();
            const body = match[0].slice(tag[0].length).replace(/\/?>(?:\s*)$/, '');
            for (const attr of body.matchAll(/([^\s"'<>\/=]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g)) {
                const name = attr[1].toLowerCase();
                if (!attributes.has(name)) attributes.set(name, htmlAttribute(attr[2] ?? attr[3] ?? attr[4] ?? ''));
            }
            for (const name of ['href', 'src', 'poster', 'xlink:href', ...(tag[1].toLowerCase() === 'object' ? ['data'] : [])]) {
                if (attributes.has(name)) inspect(getParser().normalizeLink(attributes.get(name)));
            }
            if (attributes.has('srcset')) inspectSourceSet(attributes.get('srcset'), inspect, warn);
            if (/url\s*\(|@import/i.test(attributes.get('style') || '')) {
                warn('需审阅内嵌 CSS 中的资源，未解析 CSS');
            }
            if (tag[1].toLowerCase() === 'base') warn('需审阅 HTML base 对相对资源基准的影响');
            if (tag[1].toLowerCase() === 'a') {
                if (!attributes.has('href')) warn('需审阅缺少目标的 HTML 链接');
                linkDepth++;
            }
        } else if (/^<\/a\s*>/i.test(match[0])) linkDepth = Math.max(0, linkDepth - 1);
        cursor = match.index + match[0].length;
    }
    if (!linkDepth) text += html.slice(cursor);
    return text;
}

// Only parsed link/image tokens establish a resource identity. Code blocks and code
// spans are never reinterpreted as links; literal link-like text stays reviewable.
function mapResourceLinks(source, inspect, warn) {
    let text = '';
    let linkDepth = 0;
    const inspectTarget = target => { if (target != null) inspect(target); };
    function visit(tokens) {
        for (let index = 0; index < tokens.length; index++) {
            const token = tokens[index];
            if (token.type === 'link_open') {
                inspectTarget(token.attrGet('href'));
                linkDepth++;
            } else if (token.type === 'link_close') linkDepth = Math.max(0, linkDepth - 1);
            else if (token.type === 'image') inspectTarget(token.attrGet('src'));
            else if (token.type === 'html_inline' || token.type === 'html_block') {
                // Group adjacent inline HTML and text so anchor labels are not treated
                // as unclassified artifact names. Markdown link tokens are still visited.
                let html = token.content;
                while (token.type === 'html_inline' && index + 1 < tokens.length &&
                    ['html_inline', 'text', 'softbreak', 'hardbreak'].includes(tokens[index + 1].type)) {
                    const next = tokens[++index];
                    html += ['softbreak', 'hardbreak'].includes(next.type) ? '\n' :
                        (next.type === 'text' ? getParser().utils.escapeHtml(next.content) : next.content);
                }
                const visible = inspectHtml(html, inspectTarget, warn);
                if (!linkDepth) text += visible;
            } else if (token.children && token.type !== 'code_inline') visit(token.children);
            else if (token.type === 'text' && !linkDepth) {
                text += token.content;
                if (/\]\s*\(|\[[^\]\n]+\]\[[^\]\n]*\]|^\s*\[[^\]\n]+\]:/m.test(token.content)) {
                    warn('需审阅未解析为有效链接的文字：' + token.content.slice(0, 120));
                }
            } else if (['softbreak', 'hardbreak', 'paragraph_close'].includes(token.type)) text += '\n';
        }
    }
    visit(getParser().parse(source, {}));
    return text;
}

module.exports = { codeBlocks, codeBlockRanges, mapOutsideCodeBlocks, documentHeadings, documentProse, mapResourceLinks };
