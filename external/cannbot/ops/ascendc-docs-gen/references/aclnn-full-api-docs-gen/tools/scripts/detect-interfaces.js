#!/usr/bin/env node
// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

/**
 * ACLNN 接口检测脚本
 *
 * 自动完成步骤 2-4：
 *   1. 判断给定路径是算子目录还是算子父目录
 *   2. 如为父目录，递归列出所有算子子目录
 *   3. 扫描头文件和实现文件，提取并分组所有 ACLNN 接口名
 *
 * 用法: node scripts/detect-interfaces.js <路径>
 * 示例: node scripts/detect-interfaces.js <operator-dir>
 *        node scripts/detect-interfaces.js <operator-parent>
 */

'use strict';

const fs   = require('fs');
const path = require('path');

const { collectAclnnSources } = require('./lib/aclnn-source-catalog.js');

// ========== 目录检测 ==========

const OP_SUBDIRS = ['op_host', 'op_api', 'op_kernel'];
const SKIP_DIRS = new Set([
    '.git', '.svn', '.hg',
    'build', 'cmake-build-debug', 'cmake-build-release',
    'output', 'outputs', 'out', 'install',
    'third_party', 'node_modules'
]);

function isOperatorDir(dirPath) {
    return OP_SUBDIRS.some(sub => fs.existsSync(path.join(dirPath, sub)));
}

/**
 * 在给定目录下递归查找算子目录。
 * 命中算子目录后不再向下递归，避免把 op_host/op_api 内部目录误报为算子。
 */
function findOperatorSubdirs(parentPath) {
    const results = [];

    const walk = (dirPath) => {
        let entries;
        try { entries = fs.readdirSync(dirPath, { withFileTypes: true }); }
        catch { return; }
        for (const entry of entries) {
            if (!entry.isDirectory()) continue;
            if (SKIP_DIRS.has(entry.name.toLowerCase())) continue;

            const fullPath = path.join(dirPath, entry.name);
            if (isOperatorDir(fullPath)) {
                results.push(fullPath);
                continue;
            }
            walk(fullPath);
        }
    }

    walk(parentPath);
    return [...new Set(results)].sort();
}

/**
 * 扫描算子目录下 op_api/ 和 op_host/ 的头文件，返回去重的接口基础名列表。
 */
function detectInterfacesInDir(opDir, options = {}) {
    return collectAclnnSources(opDir, options).interfaces;
}

// ========== 报告生成 ==========

function generateReport(inputPath, options = {}) {
    const lines = [];

    lines.push('# 接口检测结果');
    lines.push('');

    // 判断输入路径类型
    if (!fs.existsSync(inputPath)) {
        throw new Error(`路径不存在: ${inputPath}`);
    }

    let opDirs;
    if (isOperatorDir(inputPath)) {
        opDirs = [inputPath];
    } else {
        opDirs = findOperatorSubdirs(inputPath);
        if (opDirs.length === 0) {
            throw new Error(`未找到算子目录: ${inputPath}`);
        }
    }

    lines.push(`输入路径: \`${inputPath}\``);
    lines.push(`算子目录数: ${opDirs.length}`);
    lines.push('');
    lines.push('---');
    lines.push('');

    for (const opDir of opDirs) {
        const dirName    = path.basename(opDir);
        const interfaces = detectInterfacesInDir(opDir, options);

        lines.push(`## ${dirName}`);
        lines.push('');
        lines.push(`- 路径: \`${opDir}\``);

        if (interfaces.length === 0) {
            lines.push('- 接口: （未找到 aclnnStatus 函数声明）');
        } else {
            lines.push(`- 接口数: ${interfaces.length}`);
            lines.push('- 接口列表:');
            for (const iface of interfaces) {
                lines.push(`  - \`${iface}\``);
            }
        }
        lines.push('');
    }

    // 汇总
    if (opDirs.length > 1) {
        lines.push('---');
        lines.push('');
        lines.push('## 汇总');
        lines.push('');
        lines.push('| 算子目录 | 接口 |');
        lines.push('|---------|------|');
        for (const opDir of opDirs) {
            const dirName    = path.basename(opDir);
            const interfaces = detectInterfacesInDir(opDir, options);
            lines.push(`| \`${dirName}\` | ${interfaces.length > 0 ? interfaces.map(i => `\`${i}\``).join('、') : '无'} |`);
        }
        lines.push('');
    }

    return lines.join('\n');
}

// ========== 主函数 ==========

function main() {
    const argv = process.argv.slice(2);
    if (argv.length < 1) {
        console.error('用法: node scripts/detect-interfaces.js <路径>');
        console.error('示例: node scripts/detect-interfaces.js <operator-dir>');
        process.exit(1);
    }

    const options = {};
    for (let i = 1; i < argv.length; i++) {
        const arg = argv[i];
        const key = { '--prototype-manifest': 'prototypeManifest', '--project-root': 'projectRoot' }[arg];
        if (!key) throw new Error(`未知参数: ${arg}`);
        const value = argv[++i];
        if (!value || value.startsWith('--')) throw new Error(`${arg} 缺少参数值`);
        options[key] = value;
    }
    const inputPath = path.resolve(argv[0]);
    const report = generateReport(inputPath, options);
    process.stdout.write(report + '\n', 'utf8');
}

module.exports = { isOperatorDir, findOperatorSubdirs, detectInterfacesInDir };

if (require.main === module) {
    try { main(); } catch (error) {
        console.error(error.message);
        process.exitCode = 1;
    }
}
