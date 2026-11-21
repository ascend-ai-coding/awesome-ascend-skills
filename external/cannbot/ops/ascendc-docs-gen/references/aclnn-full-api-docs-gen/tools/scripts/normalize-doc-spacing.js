#!/usr/bin/env node
// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

/**
 * ACLNN 文档空格规范化后处理脚本
 *
 * 规则：
 * - 中文与中文之间不保留空格
 * - 中文与英文之间不保留空格
 * - 中文与数字之间不保留空格
 * - 英文与英文之间多个空格压缩为 1 个
 * - 英文与数字之间补 1 个空格，但保护数据类型、API 名、含下划线标识符等代码式 token
 *
 * 用法: node scripts/normalize-doc-spacing.js <文档路径> [--output <文件>] [--in-place]
 */

'use strict';

const fs = require('fs');
const path = require('path');
const { markdownParts, normalizeProductNames, productNames } = require('./lib/product-document-policy.js');

const CJK = '\\p{Script=Han}';

function usage() {
    console.error('用法: node scripts/normalize-doc-spacing.js <文档路径> [--output <文件>] [--in-place]');
    console.error('示例: node scripts/normalize-doc-spacing.js ./aclnnXxx.md --in-place');
}

function parseArgs(argv) {
    const positional = [];
    let outputFile = null;
    let inPlace = false;

    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        if (arg === '--output') {
            if (i + 1 >= argv.length) throw new Error('--output 需要指定输出文件路径');
            outputFile = argv[++i];
        } else if (arg === '--in-place') {
            inPlace = true;
        } else {
            positional.push(arg);
        }
    }

    if (positional.length < 1) throw new Error('缺少文档路径');
    if (outputFile && inPlace) throw new Error('--output 与 --in-place 不能同时使用');

    const inputFile = positional[0];
    return {
        inputFile,
        outputFile: inPlace ? inputFile : outputFile,
    };
}

function protectSegments(text, pattern, placeholders) {
    return text.replace(pattern, match => {
        const key = `\u0000${placeholders.length}\u0000`;
        placeholders.push(match);
        return key;
    });
}

function restoreSegments(text, placeholders) {
    return text.replace(/\u0000(\d+)\u0000/g, (_, index) => placeholders[Number(index)]);
}

function isCodeLikeToken(token) {
    if (token.includes('_')) return true;
    if (/^aclnn[A-Za-z0-9]*$/.test(token)) return true;
    if (/^[A-Z][A-Z0-9]*$/.test(token)) return true;
    if (/^[A-Z]+[a-z]+[A-Za-z0-9]*$/.test(token)) return true;
    if (/^[a-z]+[A-Z][A-Za-z0-9]*$/.test(token)) return true;
    if (/^[A-Za-z]+[0-9][A-Za-z0-9]*$/.test(token)) return true;
    return false;
}

function normalizeWordToken(token) {
    if (isCodeLikeToken(token)) return token;
    return token
        .replace(/([A-Za-z])(\d)/g, '$1 $2')
        .replace(/(\d)([A-Za-z])/g, '$1 $2');
}

function normalizePlainText(text) {
    const cjk = CJK;
    let result = text;

    result = result.replace(new RegExp(`([${cjk}])\\s+([${cjk}])`, 'gu'), '$1$2');
    result = result.replace(new RegExp(`([${cjk}])\\s+([A-Za-z])`, 'gu'), '$1$2');
    result = result.replace(new RegExp(`([A-Za-z])\\s+([${cjk}])`, 'gu'), '$1$2');
    result = result.replace(new RegExp(`([${cjk}])\\s+([0-9])`, 'gu'), '$1$2');
    result = result.replace(new RegExp(`([0-9])\\s+([${cjk}])`, 'gu'), '$1$2');
    result = result.replace(/([A-Za-z]) {2,}([A-Za-z])/g, '$1 $2');

    result = result.replace(/[A-Za-z0-9_]+/g, normalizeWordToken);

    return result;
}

function normalizeLine(line) {
    if (!line.trim()) return line;

    const placeholders = [];
    let result = line;

    result = protectSegments(result, /`[^`]*`/g, placeholders);
    result = protectSegments(result, /<term\b[^>]*>[\s\S]*?<\/term>/gi, placeholders);
    result = protectSegments(result, /!??\[[^\]]*\]\([^)]*\)/g, placeholders);
    result = protectSegments(result, /https?:\/\/[^\s)]+/g, placeholders);
    result = protectSegments(result, /<[^>]+>/g, placeholders);
    result = protectSegments(result, /&[A-Za-z0-9#]+;/g, placeholders);
    const products = new RegExp(productNames.map(name => name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|'), 'g');
    result = protectSegments(result, products, placeholders);

    result = normalizePlainText(result);
    return restoreSegments(result, placeholders);
}

function normalizeMarkdown(content) {
    let inMathBlock = false;
    return markdownParts(normalizeProductNames(content, 'aclnn')).map(part => part.kind !== 'text' ? part.text :
        part.text.split(/\r?\n/).map(line => {
            if (/^\s*\$\$\s*$/.test(line)) inMathBlock = !inMathBlock;
            return inMathBlock ? line : normalizeLine(line);
        }).join('\n')).join('');
}

function main() {
    let args;
    try {
        args = parseArgs(process.argv.slice(2));
    } catch (error) {
        usage();
        console.error(`错误: ${error.message}`);
        process.exit(1);
    }

    const inputFile = path.resolve(args.inputFile);
    if (!fs.existsSync(inputFile)) {
        console.error(`错误: 文档不存在: ${inputFile}`);
        process.exit(1);
    }

    const content = fs.readFileSync(inputFile, 'utf8');
    const normalized = normalizeMarkdown(content);

    if (args.outputFile) {
        const outputFile = path.resolve(args.outputFile);
        fs.mkdirSync(path.dirname(outputFile), { recursive: true });
        fs.writeFileSync(outputFile, normalized, 'utf8');
        console.error(`已写入: ${outputFile}`);
    } else {
        process.stdout.write(normalized, 'utf8');
    }
}

main();
