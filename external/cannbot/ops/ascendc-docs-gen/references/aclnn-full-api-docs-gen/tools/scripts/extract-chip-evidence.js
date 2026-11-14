#!/usr/bin/env node
// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

/**
 * ACLNN 芯片差异证据提取脚本
 *
 * 从算子目录中自动提取芯片/产品相关代码证据。
 * 自动执行跨目录引用追踪和3条规则的grep搜索。
 *
 * 用法: node extract-chip-evidence.js <算子目录路径> [项目根目录]
 * 示例: node extract-chip-evidence.js <operator-dir>
 */

'use strict';

const fs   = require('fs');
const path = require('path');
const {
    trackExternalFiles,
    detectProjectRoot,
    normPath,
} = require('./lib/cross-dir-tracker');
const { loadProductMappingConfig, mapChipIdentifier } = require('./lib/mapping-config');



// ========== 三条规则搜索 ==========

const CONTEXT_LINES = 5;

function searchInFiles(files, patterns, contentOverrides) {
    const results = [];
    for (const filePath of files) {
        const np = normPath(filePath);
        const content = (contentOverrides && contentOverrides.has(np))
            ? contentOverrides.get(np)
            : require('fs').readFileSync(filePath, 'utf8').toString();
        if (!content) continue;

        const lines = content.split(/\r?\n/);
        for (const { name, regex, extract, needContext } of patterns) {
            for (let i = 0; i < lines.length; i++) {
                const line = lines[i];
                const globalRegex = new RegExp(regex.source, 'g');
                let match;
                while ((match = globalRegex.exec(line)) !== null) {
                    const extracted = extract(match);
                    if (extracted) {
                        const start = Math.max(0, i - 2);
                        const end   = Math.min(lines.length - 1, i + CONTEXT_LINES);
                        const context = lines.slice(start, end + 1)
                            .map((l, idx) => `${start + idx + 1}: ${l}`)
                            .join('\n');
                        results.push({
                            rule:        name,
                            identifier:  extracted.identifier,
                            type:        extracted.type,
                            configuration: extracted.configuration,
                            filePath:    np,
                            line:        i + 1,
                            matchedLine: line.trim(),
                            context,
                        });
                    }
                }
            }
        }
    }
    return results;
}

function executeRule1(files, contentOverrides) {
    return searchInFiles(files, [{
        name:    'AddConfig',
        regex:   /AddConfig\s*\(\s*"([^"]+)"\s*(?:,\s*([\w:]+)\s*)?\)/,
        extract: (m) => ({ identifier: m[1], type: 'AddConfig', configuration: m[2] || 'default' }),
    }], contentOverrides);
}

function executeRule2(files, contentOverrides) {
    return searchInFiles(files, [
        {
            name:    'NpuArch',
            regex:   /NpuArch::(DAV_\w+|ASCEND\w+)/,
            extract: (m) => ({ identifier: m[1], type: 'NpuArch' }),
        },
        {
            name:    'SocVersion',
            regex:   /SocVersion::(\w+)/,
            extract: (m) => ({ identifier: m[1], type: 'SocVersion' }),
        },
    ], contentOverrides);
}

function executeRule3(files, contentOverrides) {
    return searchInFiles(files, [
        {
            name:    '__NPU_ARCH__',
            regex:   /__NPU_ARCH__\s*(==|>=|>|!=)\s*(\d+)/,
            extract: (m) => ({ identifier: `__NPU_ARCH__${m[1]}${m[2]}`, type: '__NPU_ARCH__' }),
        },
        {
            name:    '__CCE_AICORE__',
            regex:   /__CCE_AICORE__\s*(==|>=|>|!=)\s*(\d+)/,
            extract: (m) => ({ identifier: `__CCE_AICORE__${m[1]}${m[2]}`, type: '__CCE_AICORE__' }),
        },
        {
            name:    '__DAV_macro__',
            regex:   /__DAV_([A-Z0-9_]+)__/,
            extract: (m) => ({ identifier: `__DAV_${m[1]}__`, type: '__DAV_macro__' }),
        },
        {
            name:    'IsRegBase',
            regex:   /\b(?:isRegbase|IsRegBase)\s*\(/,
            extract: () => ({ identifier: 'IsRegBase', type: 'IsRegBase' }),
        },
    ], contentOverrides);
}

// ========== 报告生成 ==========

function dedupMatches(matches) {
    const seen = new Set();
    return matches.filter(r => {
        const key = `${r.filePath}:${r.line}:${r.identifier}:${r.type}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
    });
}

function relPath(filePath, projectRoot) {
    if (projectRoot) {
        const rel = path.relative(projectRoot, filePath).replace(/\\/g, '/');
        if (!rel.startsWith('..')) return rel;
    }
    return filePath.split('/').slice(-3).join('/');
}

function generateReport(rule1, rule2, rule3, tracking, projectRoot, productConfig, configPath) {
    const out = [];

    out.push('# 芯片支持静态分析');
    out.push('');
    out.push('## 产品型号映射配置');
    out.push('');
    out.push(`- 配置文件: ${configPath || '未配置，使用默认产品映射'}`);
    if (productConfig.warnings.length > 0) {
        out.push('- 配置告警:');
        for (const warning of productConfig.warnings) out.push(`  - ${warning}`);
    }
    out.push('');

    function renderMatches(matches) {
        for (const r of matches) {
            const rp = relPath(r.filePath, projectRoot);
            out.push(`**${rp}:${r.line}** — \`${r.identifier}\``);
            out.push('');
            out.push('```cpp');
            out.push(r.context || r.matchedLine);
            out.push('```');
            out.push('');
        }
    }

    function renderMappingSummary(matches) {
        const rows = [];
        const seen = new Set();
        for (const r of dedupMatches(matches)) {
            const products = mapChipIdentifier(r.identifier, productConfig);
            const key = `${r.identifier}:${products.join('|')}`;
            if (seen.has(key)) continue;
            seen.add(key);
            rows.push([r.identifier, r.type, products]);
        }
        if (rows.length === 0) {
            out.push('未提取到可映射芯片标识。');
            out.push('');
            return;
        }
        out.push('| 原始标识 | 类型 | 映射产品型号 |');
        out.push('|---|---|---|');
        for (const [identifier, type, products] of rows) {
            out.push(`| \`${identifier}\` | ${type} | ${products.length > 0 ? products.join('、') : '映射表中无对应条目'} |`);
        }
        out.push('');
    }

    out.push('## 映射命中摘要');
    out.push('');
    renderMappingSummary([...rule1, ...rule2, ...rule3]);

    // ---- 规则1 ----
    out.push('## 规则1：AddConfig 注册声明');
    out.push('');
    const d1 = dedupMatches(rule1);
    if (d1.length === 0) {
        out.push('未找到 AddConfig 调用。');
        out.push('');
    } else {
        renderMatches(d1);
    }

    // ---- 规则2 ----
    out.push('## 规则2：运行时 NpuArch/SocVersion 分支');
    out.push('');
    const d2 = dedupMatches(rule2);
    if (d2.length === 0) {
        out.push('未找到 NpuArch/SocVersion 条件分支。');
        out.push('');
    } else {
        renderMatches(d2);
    }

    // ---- 规则3 ----
    out.push('## 规则3：编译期架构宏分支');
    out.push('');
    const d3 = dedupMatches(rule3);
    if (d3.length === 0) {
        out.push('未找到编译期架构宏分支。');
        out.push('');
    } else {
        renderMatches(d3);
    }


    // ---- 未解析项 ----
    if (tracking.unresolvedItems.length > 0) {
        out.push('## 未解析项');
        out.push('');
        out.push('以下引用未能自动解析，需手动检查：');
        out.push('');
        for (const item of tracking.unresolvedItems) {
            out.push(`- ${item}`);
        }
        out.push('');
    }

    // ---- 搜索范围 ----
    out.push('## 搜索范围');
    out.push('');
    out.push(`- 算子目录文件数: ${tracking.operatorFiles.length}`);
    out.push(`- 追踪到的外部文件数: ${tracking.externalFiles.length}`);
    out.push('');
    if (tracking.externalFiles.length > 0) {
        out.push('### 外部文件列表');
        out.push('');
        for (const f of tracking.externalFiles) {
            out.push(`- ${relPath(f, projectRoot)}`);
        }
        out.push('');
    }

    return out.join('\n');
}

// ========== 主函数 ==========

function main() {
    const argv = process.argv.slice(2);
    const outIdx = argv.indexOf('--output');
    const outputFile = outIdx !== -1 ? path.resolve(argv.splice(outIdx, 2)[1]) : null;
    const mappingIdx = argv.indexOf('--product-mapping-file');
    const productMappingFile = mappingIdx !== -1 ? path.resolve(argv.splice(mappingIdx, 2)[1]) : null;
    const args = argv;

    if (args.length < 1) {
        console.error('用法: node extract-chip-evidence.js <算子目录路径> [项目根目录] [--output <文件>]');
        console.error('示例: node extract-chip-evidence.js <operator-dir> --output <output-dir>/1_chipdiff_evidence.md');
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
        console.error('错误: 无法自动检测项目根目录，请通过第二个参数指定。');
        console.error('用法: node extract-chip-evidence.js <算子目录路径> <项目根目录>');
        process.exit(1);
    }

    console.error(`项目根目录: ${projectRoot}`);
    console.error(`算子目录: ${operatorDir}`);

    console.error('正在执行跨目录引用追踪...');
    const tracking = trackExternalFiles(operatorDir, projectRoot);
    console.error(`  追踪到 ${tracking.externalFiles.length} 个外部文件`);
    const restrictedCount = [...tracking.fileContentOverrides.values()].filter(v => v).length;
    console.error(`  其中 ${restrictedCount} 个外部文件已精简为被引用函数体`);
    if (tracking.unresolvedItems.length > 0) {
        console.error(`  ${tracking.unresolvedItems.length} 个引用未能解析`);
    }

    const allFilesMap = new Map();
    for (const f of tracking.operatorFiles) allFilesMap.set(f, true);
    for (const f of tracking.externalFiles) allFilesMap.set(f, true);
    const allFiles = [...allFilesMap.keys()];
    console.error(`搜索范围: ${allFiles.length} 个文件`);

    const overrides = tracking.fileContentOverrides;

    console.error('正在执行规则1搜索 (AddConfig)...');
    const rule1 = executeRule1(allFiles, overrides);
    console.error(`  命中: ${rule1.length}`);

    console.error('正在执行规则2搜索 (NpuArch/SocVersion)...');
    const rule2 = executeRule2(allFiles, overrides);
    console.error(`  命中: ${rule2.length}`);

    console.error('正在执行规则3搜索 (编译期宏)...');
    const rule3 = executeRule3(allFiles, overrides);
    console.error(`  命中: ${rule3.length}`);

    const productConfig = loadProductMappingConfig(productMappingFile);
    const report = generateReport(rule1, rule2, rule3, tracking, projectRoot, productConfig, productMappingFile);
    if (outputFile) {
        fs.mkdirSync(path.dirname(outputFile), { recursive: true });
        fs.writeFileSync(outputFile, report + '\n', 'utf8');
        console.error(`已写入: ${outputFile}`);
    } else {
        process.stdout.write(report + '\n', 'utf8');
    }
}

main();
