#!/usr/bin/env node
// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

/**
 * ACLNN 函数原型提取与格式化脚本
 *
 * 从算子目录中提取函数签名并格式化为文档章节
 *
 * 用法: node extract-prototype.js <算子目录路径> [目标接口名]
 * 示例: node extract-prototype.js ./op_host aclnnMatmul
 */

const fs = require('fs');
const path = require('path');

/**
 * 移除 C++ 注释
 */
function removeComments(source) {
    // 移除单行注释 //
    source = source.replace(/\/\/.*$/gm, '');
    // 移除多行注释 /* */
    source = source.replace(/\/\*[\s\S]*?\*\//g, '');
    return source;
}

/**
 * 查找匹配的括号位置
 */
function findMatchingParen(source, openParenIndex) {
    let depth = 0;
    for (let i = openParenIndex; i < source.length; i++) {
        const ch = source[i];
        if (ch === '(') depth++;
        if (ch === ')') {
            depth--;
            if (depth === 0) return i;
        }
    }
    return -1;
}

/**
 * 分割函数参数
 */
function splitFunctionParameters(paramsStr) {
    const result = [];
    let current = '';
    let depth = 0;

    for (let i = 0; i < paramsStr.length; i++) {
        const ch = paramsStr[i];
        if (ch === '(' || ch === '<') depth++;
        if (ch === ')' || ch === '>') depth--;
        if (ch === ',' && depth === 0) {
            result.push(current.trim());
            current = '';
        } else {
            current += ch;
        }
    }
    if (current.trim()) {
        result.push(current.trim());
    }
    return result;
}

/**
 * 规范化空格
 */
function normalizeSpaces(str) {
    return str.replace(/\s+/g, ' ').trim();
}

/**
 * 解析参数
 */
function parseParameter(param) {
    const cleaned = normalizeSpaces(param);
    if (!cleaned) {
        return { type: '', pointer: '', name: '' };
    }

    if (cleaned === '...') {
        return { type: '...', pointer: '', name: '' };
    }

    // 匹配参数名（最后一个标识符）
    const nameMatch = cleaned.match(/([A-Za-z_]\w*(?:\[[^\]]*\])?)$/);
    if (!nameMatch || nameMatch.index === undefined) {
        return { type: cleaned, pointer: '', name: '' };
    }

    const name = nameMatch[1];
    const beforeName = cleaned.slice(0, nameMatch.index).trim();
    if (!beforeName || /^(?:const|volatile|struct|enum|class)$/.test(beforeName) ||
        /^(?:void|bool|char|short|int|long|float|double|signed|unsigned)$/.test(name)) {
        return { type: cleaned, pointer: '', name: '' };
    }

    // 匹配指针符号
    const pointerMatch = beforeName.match(/(\*+)\s*$/);
    const pointer = pointerMatch ? pointerMatch[1] : '';
    const type = pointer
        ? beforeName.slice(0, beforeName.length - pointer.length).trim()
        : beforeName;

    return { type, pointer, name };
}

/**
 * 格式化签名（对齐）
 */
function formatAclnnSignature(name, parameters) {
    if (parameters.length === 0) {
        return `aclnnStatus ${name}()`;
    }

    const parsed = parameters.map(p => parseParameter(p));
    const maxTypeLen = parsed.reduce((max, item) => Math.max(max, item.type.length), 0);

    const lines = parsed.map((item, index) => {
        const typePart = item.type.padEnd(maxTypeLen, ' ');
        const pointerPart = item.pointer || ' ';
        const namePart = item.name;
        const suffix = index === parsed.length - 1 ? '' : ',';
        return `  ${typePart} ${pointerPart}${namePart}${suffix}`.trimEnd();
    });

    return [`aclnnStatus ${name}(`, ...lines, ')'].join('\n');
}

/**
 * 从内容中解析签名
 */
function parseAclnnSignaturesFromContent(content) {
    const source = removeComments(content);
    const results = [];
    const headRegex = /\baclnnStatus\s+(?:[A-Za-z_][A-Za-z0-9_]*\s+)*?(aclnn[A-Za-z0-9_]*)\s*\(/g;
    let match;

    while ((match = headRegex.exec(source)) !== null) {
        const name = match[1];
        const openParenIndex = headRegex.lastIndex - 1;
        const closeParenIndex = findMatchingParen(source, openParenIndex);
        if (closeParenIndex === -1) continue;

        const afterParen = source.slice(closeParenIndex + 1);
        const endMatch = afterParen.match(/^\s*([;{])/);
        if (!endMatch) continue;

        const rawParams = source.slice(openParenIndex + 1, closeParenIndex).trim();
        const parameters = splitFunctionParameters(rawParams)
            .map(p => normalizeSpaces(p))
            .filter(Boolean);

        if (parameters.length === 1 && parameters[0] === 'void') {
            results.push({ name, parameters: [] });
        } else {
            results.push({ name, parameters });
        }

        headRegex.lastIndex = closeParenIndex + 1;
    }

    return results;
}

/**
 * 检查签名是否与目标接口相关
 *
 * 规则：
 * 1. Inplace 属性必须严格一致——目标是 Inplace 接口只匹配 Inplace 签名，反之排除
 * 2. 去掉 aclnn 前缀、Inplace 标记、GetWorkspaceSize 后缀后，算子基础名必须完全相同
 *
 * 示例：targetApiName = 'aclnnAddbmm'
 *   ✓ aclnnAddbmmGetWorkspaceSize  ✓ aclnnAddbmm
 *   ✗ aclnnInplaceAddbmm          ✗ aclnnInplaceAddbmmGetWorkspaceSize
 */
function isRelatedAclnnSignature(sigName, targetApiName) {
    if (!targetApiName) return true;

    // Inplace 类型必须严格一致
    const targetIsInplace = /Inplace/.test(targetApiName);
    const sigIsInplace    = /Inplace/.test(sigName);
    if (targetIsInplace !== sigIsInplace) return false;

    // 去掉 aclnn 前缀、Inplace 标记、GetWorkspaceSize 后缀，比较算子基础名
    const stripVariants = (s) => s
        .replace(/^aclnn/, '')
        .replace(/Inplace/, '')
        .replace(/GetWorkspaceSize$/, '');

    return stripVariants(sigName) === stripVariants(targetApiName);
}

/**
 * 比较签名顺序
 * GetWorkspaceSize 接口始终排在主接口之前（无论是否为 Inplace 变体）
 */
function compareAclnnSignatureOrder(aName, bName, targetApiName) {
    const getPriority = (name) => name.includes('GetWorkspaceSize') ? 0 : 1;
    const pa = getPriority(aName);
    const pb = getPriority(bName);
    if (pa !== pb) return pa - pb;
    return aName.localeCompare(bName);
}

/**
 * 生成函数原型章节
 */
function buildStaticAclnnPrototypeSection(allSignatures, apiName) {
    const signatures = allSignatures.filter(sig => isRelatedAclnnSignature(sig.name, apiName));
    signatures.sort((a, b) => compareAclnnSignatureOrder(a.name, b.name, apiName));
    if (signatures.length === 0) {
        return null;
    }

    // 推断 opener 用的算子基础名（不含 aclnn 前缀，不含 GetWorkspaceSize 后缀）
    let opName;
    if (apiName) {
        opName = apiName.replace(/^aclnn/, '').replace(/GetWorkspaceSize$/, '');
    } else {
        // 未提供目标名时，优先取非 Inner 变体
        const preferredSig = signatures.find(s => !s.name.includes('Inner')) || signatures[0];
        const match = preferredSig.name.match(/^aclnn(.+?)(?:GetWorkspaceSize)?$/);
        opName = match ? match[1] : preferredSig.name.replace(/^aclnn/, '');
    }

    const executeInterfaceName = `aclnn${opName}`;
    const workspaceInterfaceName = `${executeInterfaceName}GetWorkspaceSize`;
    const hasWorkspaceInterface = signatures.some(sig => sig.name === workspaceInterfaceName);
    const hasExecuteInterface = signatures.some(sig => sig.name === executeInterfaceName);
    const opener = hasWorkspaceInterface && hasExecuteInterface
        ? `每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用"${workspaceInterfaceName}"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"${executeInterfaceName}"接口执行计算。`
        : `调用"${executeInterfaceName}"接口执行计算。`;

    const blocks = signatures
        .map(sig => {
            const formatted = formatAclnnSignature(sig.name, sig.parameters);
            return ['```cpp', formatted, '```'].join('\n');
        })
        .join('\n\n');

    return [opener, '', blocks].join('\n');
}

/**
 * 主函数
 */
function main() {
    const { collectAclnnSources } = require('./lib/aclnn-source-catalog.js');
    const options = {};
    const positional = [];
    const argv = process.argv.slice(2);
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        const key = { '--output': 'outputFile', '--prototype-manifest': 'prototypeManifest', '--project-root': 'projectRoot' }[arg];
        if (key) {
            const value = argv[++i];
            if (!value || value.startsWith('--')) throw new Error(`${arg} 缺少参数值`);
            options[key] = value;
        } else if (!arg.startsWith('--') && positional.length < 2) positional.push(arg);
        else throw new Error(`未知参数: ${arg}`);
    }
    if (!positional.length) throw new Error('用法: node extract-prototype.js <算子目录> [目标接口] [--prototype-manifest <json>] [--output <文件>]');
    const catalog = collectAclnnSources(path.resolve(positional[0]), options);
    const result = buildStaticAclnnPrototypeSection(catalog.signatures, positional[1] || '');
    if (!result) throw new Error('未找到可核验的函数签名；自动生成接口需要同版本原型来源清单');
    if (options.outputFile) {
        fs.mkdirSync(path.dirname(options.outputFile), { recursive: true });
        fs.writeFileSync(options.outputFile, result + '\n', 'utf8');
    } else process.stdout.write(result + '\n');
}

module.exports = { parseAclnnSignaturesFromContent, parseParameter, buildStaticAclnnPrototypeSection };

if (require.main === module) {
    try { main(); } catch (error) {
        console.error(error.message);
        process.exitCode = 1;
    }
}
