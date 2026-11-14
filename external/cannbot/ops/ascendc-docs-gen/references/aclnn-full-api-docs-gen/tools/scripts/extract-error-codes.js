#!/usr/bin/env node
// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

/**
 * ACLNN 返回值错误码静态提取脚本
 *
 * 只扫描文档允许的 4 个错误码对应的代码上下文：
 *   161001  ACLNN_ERR_PARAM_NULLPTR
 *   161002  ACLNN_ERR_PARAM_INVALID
 *   561001  ACLNN_ERR_INNER_INFERSHAPE_ERROR
 *   561002  ACLNN_ERR_INNER_TILING_ERROR
 *
 * 用法: node scripts/extract-error-codes.js <算子目录路径> [项目根目录]
 * 示例: node scripts/extract-error-codes.js <operator-dir>
 */

'use strict';

const fs   = require('fs');
const path = require('path');
const {
    trackExternalFiles,
    detectProjectRoot,
    normPath,
} = require('./lib/cross-dir-tracker');
const { loadErrorCodeConfig } = require('./lib/mapping-config');

// ========== 配置 ==========

const MAX_CONDITION_LEN = 300;

// P1：aclnn_*.cpp（主实现文件）
function isP1File(filePath) {
    const basename = path.basename(filePath).toLowerCase();
    return basename.startsWith('aclnn_') && basename.endsWith('.cpp');
}

// ========== 错误码提取 ==========

function buildRegexes(codes) {
    const symbolPattern = Object.keys(codes).join('|');
    return {
        checkCond: new RegExp(
            `CHECK_COND\\s*\\(\\s*([\\s\\S]{1,800}?),\\s*(${symbolPattern})\\s*,\\s*"([^"]*)"`,
            'g'
        ),
        opLoge: new RegExp(
            `OP_LOGE\\s*\\(\\s*(${symbolPattern})\\s*,\\s*"([^"]*)"`,
            'g'
        ),
        checkRet: new RegExp(
            `CHECK_RET\\s*\\[[\\s\\S]{0,500}?,\\s*(${symbolPattern})\\s*\\)`,
            'g'
        ),
        directReturn: new RegExp(
            `\\breturn\\s+(${symbolPattern})\\s*;`,
            'g'
        ),
    };
}

function extractFromFile(filePath, regexes, contentOverride) {
    const content = contentOverride !== undefined ? contentOverride : (() => {
        try { return fs.readFileSync(filePath, 'utf8'); } catch { return null; }
    })();
    if (!content) return [];

    const basename = path.basename(filePath);
    const priority = isP1File(filePath) ? 'P1' : 'P2';

    const clean = content
        .replace(/\/\*[\s\S]*?\*\//g, ' ')
        .replace(/\/\/[^\n]*/g, '');

    const results = [];

    function push(symbol, condition, message, pattern, index) {
        const lineNum = clean.slice(0, index).split('\n').length;
        results.push({
            symbol,
            condition: condition.trim(),
            message,
            pattern,
            file:     basename,
            line:     lineNum,
            priority,
        });
    }

    let m;
    regexes.checkCond.lastIndex = 0;
    while ((m = regexes.checkCond.exec(clean)) !== null) push(m[2], m[1], m[3], 'CHECK_COND', m.index);
    regexes.opLoge.lastIndex = 0;
    while ((m = regexes.opLoge.exec(clean)) !== null)   push(m[1], '', m[2], 'OP_LOGE', m.index);
    regexes.checkRet.lastIndex = 0;
    while ((m = regexes.checkRet.exec(clean)) !== null)  push(m[1], '', '', 'CHECK_RET', m.index);
    regexes.directReturn.lastIndex = 0;
    while ((m = regexes.directReturn.exec(clean)) !== null)     push(m[1], '', '', 'return', m.index);

    return results;
}

function dedup(results) {
    const seen = new Set();
    return results.filter(r => {
        const key = `${r.file}:${r.line}:${r.symbol}:${r.pattern}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
    });
}

// ========== 报告生成 ==========

function generateReport(results, tracking, codes, warnings, configPath) {
    const out = [];

    out.push('# 返回值错误码提取报告');
    out.push('');
    out.push('> 脚本已扫描算子代码（含跨目录追踪文件），列出实际出现的错误码符号名。');
    out.push('> 模型在代码中搜索这些符号名，根据上下文推导每个错误码的触发原因描述。');
    out.push('');
    out.push('## 错误码配置');
    out.push('');
    out.push(`- 配置文件: ${configPath || '未配置，使用默认 4 个错误码'}`);
    out.push(`- 扫描范围: ${Object.entries(codes).map(([symbol, code]) => `${symbol}=${code}`).join('、')}`);
    if (warnings.length > 0) {
        out.push('- 配置告警:');
        for (const warning of warnings) out.push(`  - ${warning}`);
    }
    out.push('');

    const foundSymbols = [...new Set(results.map(r => r.symbol))].sort();

    out.push('## 出现的错误码');
    out.push('');
    if (foundSymbols.length === 0) {
        out.push('代码中未发现 161001 / 161002 / 561001 / 561002 任何一个。');
    } else {
        for (const symbol of foundSymbols) {
            const code = codes[symbol];
            out.push(`- \`${symbol}\`（数值码：${code}）`);
        }
    }
    out.push('');

    out.push('## 搜索范围');
    out.push('');
    out.push(`- 算子目录文件数: ${tracking.operatorFiles.length}`);
    out.push(`- 跨目录追踪外部文件数: ${tracking.externalFiles.length}`);
    out.push('');

    return out.join('\n');
}

// ========== 主函数 ==========

function main() {
    const argv = process.argv.slice(2);
    const outIdx = argv.indexOf('--output');
    const outputFile = outIdx !== -1 ? path.resolve(argv.splice(outIdx, 2)[1]) : null;
    const errorCodesIdx = argv.indexOf('--error-codes-file');
    const errorCodesFile = errorCodesIdx !== -1 ? path.resolve(argv.splice(errorCodesIdx, 2)[1]) : null;
    const args = argv;

    if (args.length < 1) {
        console.error('用法: node scripts/extract-error-codes.js <算子目录路径> [项目根目录] [--output <文件>]');
        console.error('示例: node scripts/extract-error-codes.js <operator-dir> --output <output-dir>/2_error_codes.md');
        process.exit(1);
    }

    const operatorDir  = path.resolve(args[0]);
    const explicitRoot = args[1] ? path.resolve(args[1]) : null;

    if (!fs.existsSync(operatorDir)) {
        console.error(`错误: 算子目录不存在: ${operatorDir}`);
        process.exit(1);
    }

    const projectRoot = explicitRoot || detectProjectRoot(operatorDir);
    if (!projectRoot) {
        console.error('警告: 无法自动检测项目根目录，跨目录追踪将受限。');
        console.error('建议通过第二个参数指定: node scripts/extract-error-codes.js <算子目录> <项目根目录>');
    } else {
        console.error(`项目根目录: ${projectRoot}`);
    }
    console.error(`算子目录: ${operatorDir}`);

    console.error('正在执行跨目录引用追踪...');
    const tracking = trackExternalFiles(operatorDir, projectRoot || operatorDir);
    console.error(`  追踪到 ${tracking.externalFiles.length} 个外部文件`);

    const allFilesMap = new Map();
    for (const f of tracking.operatorFiles) allFilesMap.set(f, true);
    for (const f of tracking.externalFiles) allFilesMap.set(f, true);
    const allFiles = [...allFilesMap.keys()];
    console.error(`搜索范围: ${allFiles.length} 个文件`);

    const overrides = tracking.fileContentOverrides;
    const { codes, warnings } = loadErrorCodeConfig(errorCodesFile);
    const regexes = buildRegexes(codes);

    console.error('正在提取错误码...');
    const raw = [];
    for (const f of allFiles) raw.push(...extractFromFile(f, regexes, overrides.get(f)));
    const results = dedup(raw);
    console.error(`  命中（去重后）: ${results.length} 条`);

    const report = generateReport(results, tracking, codes, warnings, errorCodesFile);
    if (outputFile) {
        fs.mkdirSync(path.dirname(outputFile), { recursive: true });
        fs.writeFileSync(outputFile, report + '\n', 'utf8');
        console.error(`已写入: ${outputFile}`);
    } else {
        process.stdout.write(report + '\n', 'utf8');
    }
}

main();
