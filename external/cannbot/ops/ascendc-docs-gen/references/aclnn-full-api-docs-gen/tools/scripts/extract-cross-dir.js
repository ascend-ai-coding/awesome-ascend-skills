#!/usr/bin/env node
// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

/**
 * 跨目录引用追踪报告生成脚本
 *
 * 分析算子目录的 #include 和 OP_TYPE_REGISTER 外部引用，
 * 生成"代码分析清单"报告。报告告知模型：
 *   1. 哪些外部文件在分析范围内
 *   2. 每个外部文件是怎么被引入的（引用链）
 *   3. 外部文件中哪些函数被实际调用（引导模型精准读取）
 *
 * 用法: node scripts/extract-cross-dir.js <算子目录路径> [项目根目录]
 * 示例: node scripts/extract-cross-dir.js <operator-dir>
 *
 * 输出到 stdout（可重定向保存为文件供模型读取）。
 */

'use strict';

const path = require('path');
const fs   = require('fs');
const {
    trackExternalFiles,
    detectProjectRoot,
    normPath,
} = require('./lib/cross-dir-tracker');

// ========== 报告生成 ==========

function relPath(filePath, base) {
    if (base) {
        const rel = path.relative(base, filePath).replace(/\\/g, '/');
        if (!rel.startsWith('..')) return rel;
    }
    return filePath.split('/').slice(-4).join('/');
}

function formatSymbols(usedSymbols) {
    if (!usedSymbols || usedSymbols.size === 0) return '—';
    return [...usedSymbols].map(s => `\`${s}\``).join('、');
}

function generateReport(tracking, operatorDir, projectRoot) {
    const out   = [];
    const base  = projectRoot || operatorDir;
    const opRel = relPath(normPath(operatorDir), base);

    out.push('# 跨目录引用追踪报告');
    out.push('');
    out.push(`- 算子目录：\`${opRel}\``);
    if (projectRoot) out.push(`- 项目根目录：\`${relPath(normPath(projectRoot), null)}\``);
    out.push(`- 算子目录文件数：${tracking.operatorFiles.length}`);
    out.push(`- 追踪到外部文件数：${tracking.externalFiles.length}`);
    out.push('');

    // ---- 精读指引（最高优先级，放在大清单之前）----
    {
        // OP_TYPE_REGISTER 核心文件：_def.cpp、proto header、直属 op_tiling/ 的 *_tiling.cpp
        const seenOp = new Set();
        const registerKeyFiles = [];   // [{ opName, defCpp, protoH, tilingFiles, registeredIn }]
        for (const f of tracking.externalFiles) {
            const meta = tracking.fileMetadata.get(f);
            if (!meta || meta.origin !== 'op_type_register' || !meta.opName) continue;
            if (seenOp.has(meta.opName)) continue;
            seenOp.add(meta.opName);
            const opFiles = tracking.externalFiles.filter(ef => {
                const em = tracking.fileMetadata.get(ef);
                return em && em.origin === 'op_type_register' && em.opName === meta.opName;
            });
            const defCpp  = opFiles.find(ef => /_def\.cpp$/.test(ef));
            const protoH  = opFiles.find(ef => /_proto\.h$/.test(ef));
            // 直属 op_tiling/ 的 *_tiling.cpp（不在 arch 子目录下）
            const tilingFiles = opFiles.filter(ef =>
                /_tiling\.cpp$/.test(ef) && /[/\\]op_tiling[/\\][^/\\]+$/.test(ef)
            );
            registerKeyFiles.push({ opName: meta.opName, defCpp, protoH, tilingFiles, registeredIn: meta.registeredIn });
        }

        // #include 外部文件（含具体函数名）
        const includeKeyFiles = tracking.externalFiles.filter(f => {
            const meta = tracking.fileMetadata.get(f);
            if (!meta || meta.origin !== 'include') return false;
            if (!meta.usedSymbols || meta.usedSymbols.size === 0) return false;
            return tracking.fileContentOverrides.has(f) && tracking.fileContentOverrides.get(f);
        });

        const hasAny = registerKeyFiles.length > 0 || includeKeyFiles.length > 0;
        if (hasAny) {
            out.push('## 必读外部文件');
            out.push('');
            out.push('以下文件包含本算子的核心约束、dtype 支持和芯片分支，**必须在分析算子目录文件的同时读取**。');
            out.push('');

            if (registerKeyFiles.length > 0) {
                out.push('### OP_TYPE_REGISTER 底层算子核心文件');
                out.push('');
                out.push('算子通过 `OP_TYPE_REGISTER` 调用以下底层算子，其 `_def.cpp`（AddConfig dtype/format/芯片注册）、');
                out.push('proto header（shape 推导约束）和直属 op_tiling/ 的 tiling 文件（shape/attr 校验逻辑）');
                out.push('包含本算子真正的参数约束，必须读取。');
                out.push('');
                out.push('| 底层算子 | _def.cpp | proto header | tiling 文件 | 注册来源 |');
                out.push('|---------|----------|--------------|------------|--------|');
                for (const { opName, defCpp, protoH, tilingFiles, registeredIn } of registerKeyFiles) {
                    const defStr    = defCpp ? `\`${defCpp}\`` : '—';
                    const protoStr  = protoH ? `\`${protoH}\`` : '—';
                    const tilingStr = tilingFiles.length > 0
                        ? tilingFiles.map(tf => `\`${tf}\``).join('<br>')
                        : '—';
                    const regRel = relPath(registeredIn, base);
                    out.push(`| ${opName} | ${defStr} | ${protoStr} | ${tilingStr} | \`${regRel}\` |`);
                }
                out.push('');
            }

            if (includeKeyFiles.length > 0) {
                out.push('### #include 引入的外部文件');
                out.push('');
                out.push('使用 Grep 定位函数名，再用 Read offset/limit 精准提取函数体，不要全文读取。');
                out.push('');
                out.push('| 外部文件（绝对路径） | 需精读的函数 |');
                out.push('|---------------------|------------|');
                for (const f of includeKeyFiles) {
                    const meta = tracking.fileMetadata.get(f);
                    out.push(`| \`${f}\` | ${formatSymbols(meta.usedSymbols)} |`);
                }
                out.push('');
            }
        }
    }

    // ---- 代码分析清单（完整引用列表，供参考）----
    out.push('## 代码分析清单（完整引用列表）');
    out.push('');
    out.push('模型在后续阶段应读取以下文件。外部文件已标注引用链和实际引用的函数名，');
    out.push('读取外部文件时只需精准读取被引用的函数体，无需全文读取。');
    out.push('');
    out.push('| 文件路径 | 分析原因 |');
    out.push('|---------|---------|');

    // 算子目录文件（按子目录分组，只列关键入口）
    const aclnnFiles = tracking.operatorFiles
        .filter(f => /op_api[/\\]aclnn_/.test(f) || /op_host[/\\]aclnn_/.test(f));
    const defFiles   = tracking.operatorFiles.filter(f => /_def\.cpp$/.test(f));
    const kernelFiles = tracking.operatorFiles.filter(f => /op_kernel/.test(f) && /\.cpp$/.test(f));
    const tilingFiles = tracking.operatorFiles.filter(f => /op_tiling|_tiling\.cpp$/.test(f));

    for (const f of aclnnFiles) {
        out.push(`| \`${relPath(f, base)}\` | aclnn 接口实现文件 |`);
    }
    for (const f of defFiles) {
        out.push(`| \`${relPath(f, base)}\` | 算子注册定义（AddConfig） |`);
    }
    for (const f of tilingFiles.slice(0, 3)) {
        out.push(`| \`${relPath(f, base)}\` | tiling 实现 |`);
    }
    for (const f of kernelFiles.slice(0, 3)) {
        out.push(`| \`${relPath(f, base)}\` | kernel 实现 |`);
    }
    const otherOpFiles = tracking.operatorFiles.filter(f =>
        !aclnnFiles.includes(f) && !defFiles.includes(f) &&
        !tilingFiles.includes(f) && !kernelFiles.includes(f)
    );
    if (otherOpFiles.length > 0) {
        out.push(`| *(算子目录其余 ${otherOpFiles.length} 个文件)* | 算子目录内部文件 |`);
    }

    // 外部文件（按来源分组）
    const includeFiles = tracking.externalFiles.filter(f => {
        const meta = tracking.fileMetadata.get(f);
        return meta && meta.origin === 'include';
    });
    const registerFiles = tracking.externalFiles.filter(f => {
        const meta = tracking.fileMetadata.get(f);
        return meta && meta.origin === 'op_type_register';
    });

    if (includeFiles.length > 0) {
        // 按引用链对 include 文件去重显示（只显示头文件，cpp 作附注）
        const shownH = new Set();
        for (const f of includeFiles) {
            const meta = tracking.fileMetadata.get(f);
            if (!meta) continue;
            const ext = f.split('.').pop().toLowerCase();
            const isH = ext === 'h' || ext === 'hpp';
            if (!isH) continue;  // .cpp 在同一行描述
            shownH.add(f);

            const callerRel = relPath(meta.includedBy, base);
            const depth     = meta.includeDepth > 1 ? `（第 ${meta.includeDepth} 层）` : '';
            const syms      = formatSymbols(meta.usedSymbols);
            const reason    = `**由 \`${callerRel}\` #include 引入**${depth}；实际引用函数：${syms}`;
            out.push(`| \`${relPath(f, base)}\` | ${reason} |`);

            // 同名 .cpp 附在下一行
            const cppKey = tracking.externalFiles.find(ef => {
                const emeta = tracking.fileMetadata.get(ef);
                if (!emeta || emeta.origin !== 'include') return false;
                const efExt = ef.split('.').pop().toLowerCase();
                return (efExt === 'cpp' || efExt === 'cc' || efExt === 'cxx') &&
                       ef.replace(/\.\w+$/, '') === f.replace(/\.\w+$/, '');
            });
            if (cppKey) {
                const hasCppOverride = tracking.fileContentOverrides.has(cppKey) &&
                                       tracking.fileContentOverrides.get(cppKey);
                const cppNote = hasCppOverride
                    ? `**同上（.cpp 实现）**；精读范围：${syms} 函数体`
                    : `**同上（.cpp 实现）**；声明在 .h，.cpp 中无对应定义`;
                out.push(`| \`${relPath(cppKey, base)}\` | ${cppNote} |`);
            }
        }
        // 没有对应头文件的外部 cpp（直接被 include）
        for (const f of includeFiles) {
            const meta = tracking.fileMetadata.get(f);
            if (!meta) continue;
            const ext = f.split('.').pop().toLowerCase();
            if (ext === 'h' || ext === 'hpp') continue;
            const hasHPair = includeFiles.some(hf => {
                const hExt = hf.split('.').pop().toLowerCase();
                return (hExt === 'h' || hExt === 'hpp') &&
                       hf.replace(/\.\w+$/, '') === f.replace(/\.\w+$/, '');
            });
            if (hasHPair) continue;
            const callerRel = relPath(meta.includedBy, base);
            out.push(`| \`${relPath(f, base)}\` | **由 \`${callerRel}\` #include 引入** |`);
        }
    }

    if (registerFiles.length > 0) {
        // 按算子名去重
        const seenOp = new Set();
        for (const f of registerFiles) {
            const meta = tracking.fileMetadata.get(f);
            if (!meta || !meta.opName) continue;
            if (!seenOp.has(meta.opName)) {
                seenOp.add(meta.opName);
                const callerRel = relPath(meta.registeredIn, base);
                out.push(`| \`${relPath(f, base)}\` | **由 \`${callerRel}\` OP_TYPE_REGISTER(${meta.opName}) 引入** |`);
            } else {
                out.push(`| \`${relPath(f, base)}\` | OP_TYPE_REGISTER(${meta.opName}) 引入的算子目录文件 |`);
            }
        }
    }

    out.push('');

    // ---- 未解析项 ----
    if (tracking.unresolvedItems.length > 0) {
        out.push('## 未解析项');
        out.push('');
        out.push('以下引用未能自动解析，模型分析时可跳过：');
        out.push('');
        for (const item of tracking.unresolvedItems.slice(0, 30)) {
            out.push(`- ${item}`);
        }
        if (tracking.unresolvedItems.length > 30) {
            out.push(`- *(另有 ${tracking.unresolvedItems.length - 30} 条，已省略)*`);
        }
        out.push('');
    }

    return out.join('\n');
}

// ========== 主函数 ==========

function main() {
    const argv = process.argv.slice(2);
    const outIdx = argv.indexOf('--output');
    if (outIdx !== -1 && (!argv[outIdx + 1] || argv[outIdx + 1].startsWith('--'))) {
        throw new Error('--output 缺少文件路径');
    }
    const outputFile = outIdx !== -1 ? path.resolve(argv.splice(outIdx, 2)[1]) : null;
    const args = argv;

    if (args.length < 1) {
        console.error('用法: node scripts/extract-cross-dir.js <算子目录路径> [项目根目录] [--output <文件>]');
        console.error('示例: node scripts/extract-cross-dir.js <operator-dir> --output <output-dir>/0_cross_dir.md');
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
        console.error('警告: 无法自动检测项目根目录，建议通过第二个参数指定。');
    }

    console.error(`算子目录: ${operatorDir}`);
    if (projectRoot) console.error(`项目根目录: ${projectRoot}`);
    console.error('正在执行跨目录引用追踪...');

    const tracking = trackExternalFiles(operatorDir, projectRoot || operatorDir);

    console.error(`  算子目录文件: ${tracking.operatorFiles.length}`);
    console.error(`  外部文件: ${tracking.externalFiles.length}`);
    console.error(`  内容精简文件: ${[...tracking.fileContentOverrides.values()].filter(v => v).length}`);
    if (tracking.unresolvedItems.length > 0) {
        console.error(`  未解析引用: ${tracking.unresolvedItems.length}`);
    }

    const report = generateReport(tracking, operatorDir, projectRoot);
    if (outputFile) {
        fs.mkdirSync(path.dirname(outputFile), { recursive: true });
        fs.writeFileSync(outputFile, report + '\n', 'utf8');
        console.error(`已写入: ${outputFile}`);
    } else {
        process.stdout.write(report + '\n', 'utf8');
    }
}

main();
