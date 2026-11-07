// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
'use strict';

/**
 * 跨目录引用追踪共享模块
 *
 * 追踪算子目录的 #include 和 OP_TYPE_REGISTER 外部引用，
 * 并精准提取每个外部文件中被实际引用的函数体。
 *
 * 导出：
 *   trackExternalFiles(operatorDir, projectRoot)
 *   detectProjectRoot(operatorDir)
 *   collectFiles(dir)
 *   normPath(p)  readFile(p)
 *   escapeRegex  stripComments  extractHeaderSymbols  findUsedSymbols  extractFunctionBodies
 */

const fs   = require('fs');
const path = require('path');

// ========== 常量 ==========

const CODE_EXTS = new Set(['.h', '.hpp', '.cpp', '.cc', '.cxx']);
const SKIP_DIRS = new Set(['tests', 'test', 'examples', '.git', 'node_modules', 'build', 'cmake-build']);
const SYSTEM_INCLUDE_PREFIXES = [
    'opdev/', 'aclnn_kernels/', 'level0/', 'acl/', 'aclnn/', 'aclnnop/',
    'register/', 'graph/', 'common/op_host/', 'op_api/op_api_def.h',
];

const MAX_INCLUDE_DEPTH  = 1;
const MAX_REGISTER_DEPTH = 0;

// ========== 基础工具 ==========

function isCodeFile(p)  { return CODE_EXTS.has(path.extname(p).toLowerCase()); }
function normPath(p)    { return path.resolve(p).replace(/\\/g, '/'); }
function readFile(p)    { try { return fs.readFileSync(p, 'utf8'); } catch { return null; } }

function collectFiles(dir) {
    const results = [];
    if (!fs.existsSync(dir)) return results;
    try {
        for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
            if (entry.name.endsWith('.md')) continue;
            const full = path.join(dir, entry.name);
            if (entry.isDirectory()) {
                if (!SKIP_DIRS.has(entry.name.toLowerCase())) results.push(...collectFiles(full));
            } else if (entry.isFile() && isCodeFile(full)) {
                results.push(full);
            }
        }
    } catch { /* ignore */ }
    return results;
}

// ========== 函数体精准提取 ==========

function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function stripComments(content) {
    return content
        .replace(/\/\*[\s\S]*?\*\//g, m => m.replace(/[^\n]/g, ' '))
        .replace(/\/\/[^\n]*/g, m => ' '.repeat(m.length));
}

function extractHeaderSymbols(headerContent) {
    const symbols = new Set();
    const clean = stripComments(headerContent);
    const KEYWORDS = new Set([
        'if','for','while','do','switch','return','sizeof','typeof','alignof',
        'decltype','static_assert','catch','throw','operator','delete','new',
        'const','virtual','override','explicit','inline','template','typename',
        'class','struct','enum','namespace','using','typedef','extern','static',
        'auto','void','int','bool','char','float','double','long','short','unsigned','signed',
    ]);
    let m;
    const fnRe = /\b([A-Za-z_]\w*)\s*\(/g;
    while ((m = fnRe.exec(clean)) !== null) {
        if (!KEYWORDS.has(m[1])) symbols.add(m[1]);
    }
    const classRe = /\b(?:class|struct)\s+([A-Za-z_]\w*)/g;
    while ((m = classRe.exec(clean)) !== null) symbols.add(m[1]);
    const nsRe = /\bnamespace\s+([A-Za-z_]\w*)/g;
    while ((m = nsRe.exec(clean)) !== null) symbols.add(m[1]);
    return symbols;
}

function findUsedSymbols(exportedSymbols, callerContent) {
    const used = new Set();
    const clean = stripComments(callerContent);
    for (const sym of exportedSymbols) {
        if (new RegExp(`\\b${escapeRegex(sym)}\\b`).test(clean)) used.add(sym);
    }
    return used;
}

function extractFunctionBodies(content, usedSymbols) {
    if (!usedSymbols || usedSymbols.size === 0) return '';
    const lines      = content.split(/\r?\n/);
    const cleanLines = stripComments(content).split(/\r?\n/);
    const parts = [];
    const processed = new Set();
    for (const sym of usedSymbols) {
        if (processed.has(sym)) continue;
        const symRe = new RegExp(`\\b${escapeRegex(sym)}\\b`);
        for (let i = 0; i < cleanLines.length; i++) {
            if (!symRe.test(cleanLines[i])) continue;
            let depth = 0, hasOpenBrace = false, defEndLine = -1;
            outer:
            for (let j = i; j < Math.min(i + 40, cleanLines.length); j++) {
                for (const ch of cleanLines[j]) {
                    if (ch === '{') { depth++; hasOpenBrace = true; }
                    else if (ch === '}') {
                        depth--;
                        if (hasOpenBrace && depth === 0) { defEndLine = j; break outer; }
                    } else if (ch === ';' && !hasOpenBrace) { break outer; }
                }
            }
            if (defEndLine !== -1) {
                parts.push(lines.slice(i, defEndLine + 1).join('\n'));
                processed.add(sym);
                break;
            }
        }
    }
    return parts.join('\n\n');
}

// ========== 项目根目录检测 ==========

function detectProjectRoot(operatorDir) {
    const files = collectFiles(operatorDir);
    const multiLevelIncludes = new Set();
    for (const file of files) {
        const content = readFile(file);
        if (!content) continue;
        const regex = /^#\s*include\s+"([^"]+)"/gm;
        let m;
        while ((m = regex.exec(content)) !== null) {
            if (m[1].split(/[/\\]/).length >= 2) multiLevelIncludes.add(m[1]);
        }
    }
    if (multiLevelIncludes.size === 0) return null;
    const incList = [...multiLevelIncludes];
    let current = path.resolve(operatorDir);
    const candidates = [];
    for (let i = 0; i < 10; i++) {
        const parent = path.dirname(current);
        if (parent === current) break;
        let resolved = 0;
        for (const inc of incList) {
            if (fs.existsSync(path.join(parent, inc))) resolved++;
        }
        candidates.push({ dir: parent, resolved });
        current = parent;
    }
    if (!candidates.length) return null;
    const maxResolved = Math.max(...candidates.map(c => c.resolved));
    if (maxResolved === 0) return null;
    return candidates.find(c => c.resolved === maxResolved).dir;
}

// ========== 核心追踪函数 ==========

/**
 * 追踪算子目录的跨目录引用。
 *
 * 返回：
 *   operatorFiles      string[]         算子目录内所有代码文件
 *   externalFiles      string[]         追踪到的外部文件（绝对路径）
 *   fileContentOverrides  Map<path, string>  外部文件的精简内容（仅被引用函数体）
 *   fileMetadata       Map<path, object> 每个外部文件的元数据（来源、引用链、引用符号）
 *   unresolvedItems    string[]         未能解析的引用
 *
 * fileMetadata[path] 结构：
 *   origin: 'include' | 'op_type_register'
 *   includedBy: string | null       直接引入者的文件路径
 *   includeDepth: number            include 追踪深度
 *   opName: string | null           OP_TYPE_REGISTER 的算子名
 *   registeredIn: string | null     包含 OP_TYPE_REGISTER 调用的文件
 *   usedSymbols: Set<string>        引入者实际使用的符号名
 */
function trackExternalFiles(operatorDir, projectRoot) {
    const externalFiles      = new Set();
    const fileContentOverrides = new Map();
    const fileMetadata       = new Map();
    const unresolvedItems    = [];
    const visited            = new Set();

    const opDirNorm = normPath(operatorDir) + '/';

    // ----- 内部辅助 -----

    function resolveIncludePath(incPath, sourceFile) {
        const relToSource = path.normalize(path.join(path.dirname(sourceFile), incPath));
        if (fs.existsSync(relToSource)) {
            const isExternal = !normPath(relToSource).startsWith(opDirNorm);
            return { resolved: true, fullPath: relToSource, isExternal };
        }
        if (projectRoot) {
            const relToRoot = path.normalize(path.join(projectRoot, incPath));
            if (fs.existsSync(relToRoot)) {
                return { resolved: true, fullPath: relToRoot, isExternal: true };
            }
        }
        return { resolved: false };
    }

    function findCppImpl(hPath) {
        const dir  = path.dirname(hPath);
        const base = path.basename(hPath, path.extname(hPath));
        for (const b of [base, base.toLowerCase()]) {
            for (const ext of ['.cpp', '.cc', '.cxx']) {
                const p = path.join(dir, b + ext);
                if (fs.existsSync(p)) return p;
            }
        }
        return null;
    }

    function findOperatorDir(opName) {
        const opNorm = opName.toLowerCase();
        const searchBases = [
            path.dirname(operatorDir),
            path.dirname(path.dirname(operatorDir)),
        ];
        for (const base of searchBases) {
            if (!fs.existsSync(base)) continue;
            for (const entry of fs.readdirSync(base, { withFileTypes: true })) {
                if (!entry.isDirectory()) continue;
                if (entry.name.replace(/_/g, '').toLowerCase() !== opNorm) continue;
                const candidate = path.join(base, entry.name);
                const hasHost   = fs.existsSync(path.join(candidate, 'op_host'));
                const hasKernel = fs.existsSync(path.join(candidate, 'op_kernel'));
                if (hasHost || hasKernel) return candidate;
            }
        }
        return null;
    }

    function addFile(filePath, meta) {
        const n = normPath(filePath);
        if (visited.has(n)) return;
        visited.add(n);
        externalFiles.add(n);
        fileMetadata.set(n, meta);
    }

    // ----- #include 追踪 -----

    function trackIncludes(filePaths, depth, maxDepth) {
        if (depth > maxDepth) return;
        for (const filePath of filePaths) {
            const content = readFile(filePath);
            if (!content) continue;
            const regex = /^#\s*include\s+"([^"]+)"/gm;
            let m;
            const newFiles = [];

            while ((m = regex.exec(content)) !== null) {
                const result = resolveIncludePath(m[1], filePath);
                if (!result.resolved) {
                    if (m[1].includes('/') || m[1].includes('\\')) {
                        const isSystem = SYSTEM_INCLUDE_PREFIXES.some(
                            p => m[1].startsWith(p) || m[1].startsWith('./' + p)
                        );
                        if (!isSystem) {
                            unresolvedItems.push(
                                `#include "${m[1]}" (from ${normPath(filePath)}) — file not found`
                            );
                        }
                    }
                    continue;
                }
                if (!result.isExternal) continue;

                const hNorm    = normPath(result.fullPath);
                const hContent = readFile(result.fullPath);
                const exported = hContent ? extractHeaderSymbols(hContent) : new Set();
                const used     = hContent ? findUsedSymbols(exported, content) : new Set();

                addFile(result.fullPath, {
                    origin:       'include',
                    includedBy:   normPath(filePath),
                    includeDepth: depth,
                    opName:       null,
                    registeredIn: null,
                    usedSymbols:  used,
                });

                // .h 文件：限制为被引用函数体（处理 inline 实现）；usedSymbols 非空时必须存 override
                if (hContent && used.size > 0) {
                    fileContentOverrides.set(hNorm, extractFunctionBodies(hContent, used));
                }

                newFiles.push(result.fullPath);

                // 查找同名 .cpp，也限制为被引用函数体
                const cppImpl = findCppImpl(result.fullPath);
                if (cppImpl) {
                    const cppNorm    = normPath(cppImpl);
                    const cppContent = readFile(cppImpl);
                    addFile(cppImpl, {
                        origin:       'include',
                        includedBy:   normPath(filePath),
                        includeDepth: depth,
                        opName:       null,
                        registeredIn: null,
                        usedSymbols:  used,
                    });
                    if (cppContent) {
                        const restricted = extractFunctionBodies(cppContent, used);
                        if (restricted) fileContentOverrides.set(cppNorm, restricted);
                    }
                    newFiles.push(cppImpl);
                }
            }

            if (newFiles.length > 0) trackIncludes(newFiles, depth + 1, maxDepth);
        }
    }

    // ----- OP_TYPE_REGISTER 追踪 -----

    function trackRegisters(filePaths) {
        for (const filePath of filePaths) {
            const content = readFile(filePath);
            if (!content) continue;
            const regex = /OP_TYPE_REGISTER\s*\(\s*(\w+)\s*\)/g;
            let m;
            while ((m = regex.exec(content)) !== null) {
                const opName = m[1];
                const opDir  = findOperatorDir(opName);
                if (opDir && normPath(opDir) !== normPath(operatorDir)) {
                    const opFiles = collectFiles(opDir);
                    for (const f of opFiles) {
                        addFile(f, {
                            origin:       'op_type_register',
                            includedBy:   null,
                            includeDepth: 0,
                            opName:       opName,
                            registeredIn: normPath(filePath),
                            usedSymbols:  new Set(),
                        });
                    }
                    trackIncludes(opFiles, 1, MAX_REGISTER_DEPTH);
                } else {
                    unresolvedItems.push(
                        `OP_TYPE_REGISTER(${opName}) (from ${normPath(filePath)}) — no operator directory found`
                    );
                }
            }
        }
    }

    // ----- 主流程 -----

    const operatorFiles = collectFiles(operatorDir);

    trackIncludes(operatorFiles, 1, MAX_INCLUDE_DEPTH);
    trackRegisters(operatorFiles);

    return {
        operatorFiles:       operatorFiles.map(normPath),
        externalFiles:       [...externalFiles],
        fileContentOverrides,
        fileMetadata,
        unresolvedItems,
    };
}

module.exports = {
    trackExternalFiles,
    detectProjectRoot,
    collectFiles,
    normPath,
    readFile,
    escapeRegex,
    stripComments,
    extractHeaderSymbols,
    findUsedSymbols,
    extractFunctionBodies,
    MAX_INCLUDE_DEPTH,
    MAX_REGISTER_DEPTH,
};
