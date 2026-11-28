#!/usr/bin/env node
// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

/**
 * ACLNN 调用示例提取脚本
 *
 * 从算子目录的 examples 目录中提取代表性调用示例
 *
 * 用法: node extract-call-example.js <算子目录路径> [目标接口名]
 * 示例: node extract-call-example.js ./ aclnnMatmul
 */

const fs = require('fs');
const path = require('path');
const { loadProductMappingConfig } = require('./lib/mapping-config');
const { supportProducts } = require('./lib/product-document-policy');

/**
 * 确保字符串以换行符结尾
 */
function ensureTrailingNewline(content) {
    return content.endsWith('\n') ? content : content + '\n';
}

/**
 * 清理示例代码中的 copyright 注释
 */
function cleanExampleCode(content) {
    // Preserve source comments, including license notices and example data.
    return content.trim();
}

/**
 * 按扩展名优先级排序
 */
function extPriority(filePath) {
    const lower = filePath.toLowerCase();
    if (lower.endsWith('.cpp')) return 0;
    if (lower.endsWith('.cc')) return 1;
    if (lower.endsWith('.cxx')) return 2;
    if (lower.endsWith('.h')) return 3;
    if (lower.endsWith('.hpp')) return 4;
    return 9;
}

/**
 * 按名称优先级排序
 */
function namePriority(filePath) {
    const lower = filePath.toLowerCase();
    if (lower.includes('demo')) return 0;
    if (lower.includes('example')) return 1;
    if (lower.includes('sample')) return 2;
    if (lower.includes('test')) return 3;
    return 4;
}

/**
 * 过滤与算子相关的示例文件
 */
function filterExampleCandidatesByOperator(files, operatorName) {
    if (!operatorName) {
        return [];
    }

    const normalizedName = operatorName.trim();
    if (!normalizedName) {
        return [];
    }

    const name = normalizedName.replace(/GetWorkspaceSize$/, '');
    if (!/^aclnn[A-Za-z0-9_]+$/.test(name)) return [];
    // Filenames, comments, strings and another API version are not call evidence.
    return files.filter(file => {
        const code = (file.content || '').replace(
            /\/\*[\s\S]*?\*\/|\/\/[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/g, ' ');
        return [name, `${name}GetWorkspaceSize`].every(target =>
            new RegExp(`\\b${target}\\s*\\(`).test(code));
    });
}

/**
 * 选择代表性示例
 */
function pickRepresentativeExample(files, operatorName) {
    const matchedFiles = filterExampleCandidatesByOperator(files, operatorName);
    const candidates = operatorName ? matchedFiles : files;

    const sorted = [...candidates].sort((a, b) => {
        const extDiff = extPriority(a.path) - extPriority(b.path);
        if (extDiff !== 0) return extDiff;

        const nameDiff = namePriority(a.path) - namePriority(b.path);
        if (nameDiff !== 0) return nameDiff;

        const depthDiff = a.path.split('/').length - b.path.split('/').length;
        if (depthDiff !== 0) return depthDiff;

        return a.path.localeCompare(b.path);
    });

    return sorted[0];
}

/**
 * 读取目录下所有文件
 */
function readDirectoryRecursively(dir, extensions) {
    if (!fs.existsSync(dir)) {
        return [];
    }

    const results = [];
    const entries = fs.readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
            results.push(...readDirectoryRecursively(fullPath, extensions));
        } else if (entry.isFile()) {
            const ext = path.extname(entry.name).toLowerCase();
            if (extensions.includes(ext)) {
                const content = fs.readFileSync(fullPath, 'utf8');
                results.push({
                    path: fullPath.replace(/\\/g, '/'),  // 统一使用正斜杠
                    content: content
                });
            }
        }
    }

    return results;
}

/**
 * 构建调用示例章节
 */
function buildStaticAclnnCallExampleSection(examplesDir, operatorName, productConfig) {
    const extensions = ['.cpp', '.cc', '.cxx', '.h', '.hpp'];
    const baseDir = examplesDir.replace(/\\/g, '/');

    const allFiles = readDirectoryRecursively(examplesDir, extensions).map(file => ({
        ...file,
        path: file.path.replace(/\\/g, '/')
    }));

    if (allFiles.length === 0) {
        return '- <span style="color:#dc2626;">【需用户补齐】缺少调用示例代码。</span>';
    }

    // 情况一：examples/ 根目录直接有文件 → 所有芯片公用示例，取一个输出单个代码块
    const rootFiles = allFiles.filter(file => {
        const rel = file.path.slice(baseDir.length + 1);
        return !rel.includes('/');
    });

    if (rootFiles.length > 0) {
        const selected = pickRepresentativeExample(rootFiles, operatorName);
        if (selected) return [
            '```cpp',
            ensureTrailingNewline(cleanExampleCode(selected.content)),
            '```'
        ].join('\n');
    }

    // 情况二：examples/ 根目录无文件，只有子目录 → 每个子目录取一个文件，按芯片映射加标签
    const bySubdir = new Map();
    for (const file of allFiles) {
        const rel = file.path.slice(baseDir.length + 1);
        const subdir = rel.split('/')[0];
        if (!bySubdir.has(subdir)) bySubdir.set(subdir, []);
        bySubdir.get(subdir).push(file);
    }

    const blocks = [];
    for (const [subdir, files] of [...bySubdir.entries()].sort(([a], [b]) => a.localeCompare(b))) {
        const archLabel = buildExampleArchLabel(subdir, productConfig);
        if (!archLabel) continue;  // 映射不到文档产品的子目录跳过

        const selected = pickRepresentativeExample(files, operatorName);
        if (!selected) continue;

        for (const product of archLabel.split('、')) {
            const npu = supportProducts('aclnn').find(([name]) => name === product)?.[1];
            if (!npu) throw new Error(`调用示例产品缺少 NPU 映射: ${product}`);
            const id = 2000 + blocks.length;
            blocks.push([
                `<!-- npu="${npu}" id${id} -->`,
                `- <term>${product}</term>：`,
                '  ```cpp',
                ensureTrailingNewline(cleanExampleCode(selected.content)).split('\n').map(line => '  ' + line).join('\n'),
                '  ```',
                `<!-- end id${id} -->`
            ].join('\n'));
        }
    }

    if (blocks.length === 0) {
        return '- <span style="color:#dc2626;">【需用户补齐】未找到可提取的调用示例代码。</span>';
    }

    return blocks.join('\n\n');
}

/**
 * 从目录名提取完整 __NPU_ARCH__，精确映射产品型号标签
 * 匹配不上返回 null（调用方跳过该块，不输出）
 */
function buildExampleArchLabel(archDirName, productConfig) {
    const normalized = archDirName.toLowerCase();
    const match = normalized.match(/^arch[_-]?([0-9]+)$/);
    if (!match) return null;

    const arch = match[1];
    const products = [];
    for (const item of productConfig.npuArchMapping) {
        if (String(item.arch) === arch) {
            const product = normalizeExampleProduct(item.product);
            if (!products.includes(product)) {
                products.push(product);
            }
        }
    }
    const allowed = new Set(productConfig.finalAllowedProducts);
    return products.filter(product => allowed.has(product)).join('、') || null;
}

function normalizeExampleProduct(product) {
    const text = String(product || '');
    if (text.includes('Atlas推理系列产品AI Core') && text.includes('Atlas推理系列产品Vector Core')) {
        return 'Atlas推理系列产品';
    }
    return text;
}

/**
 * 主函数
 */
function main() {
    const args = process.argv.slice(2);

    if (args.length < 1) {
        console.error('用法: node extract-call-example.js <算子目录路径> [目标接口名] [--output <文件路径>]');
        console.error('示例: node extract-call-example.js ./ aclnnMatmul --output ./outputs/aclnnMatmul/0_call_example.md');
        process.exit(1);
    }

    const outputFlagIdx = args.indexOf('--output');
    let outputFile = null;
    let positional = args;
    if (outputFlagIdx !== -1) {
        outputFile = args[outputFlagIdx + 1];
        positional = args.filter((_, i) => i !== outputFlagIdx && i !== outputFlagIdx + 1);
    }
    const mappingFlagIdx = positional.indexOf('--product-mapping-file');
    let productMappingFile = null;
    if (mappingFlagIdx !== -1) {
        productMappingFile = path.resolve(positional[mappingFlagIdx + 1]);
        positional = positional.filter((_, i) => i !== mappingFlagIdx && i !== mappingFlagIdx + 1);
    }

    const operatorDir = positional[0];
    const operatorName = positional[1] || '';

    // 查找 examples 目录
    const examplesDir = path.join(operatorDir, 'examples');

    if (!fs.existsSync(examplesDir)) {
        const placeholder = '- <span style="color:#dc2626;">【需用户补齐】缺少调用示例代码。</span>\n';
        if (outputFile) {
            fs.mkdirSync(path.dirname(outputFile), { recursive: true });
            fs.writeFileSync(outputFile, placeholder, 'utf8');
        } else {
            console.error(`错误: 找不到 examples 目录: ${examplesDir}`);
            process.exit(1);
        }
        return;
    }

    const productConfig = loadProductMappingConfig(productMappingFile);
    const result = buildStaticAclnnCallExampleSection(examplesDir, operatorName, productConfig);
    const output = result + '\n';

    if (outputFile) {
        fs.mkdirSync(path.dirname(outputFile), { recursive: true });
        fs.writeFileSync(outputFile, output, 'utf8');
    } else {
        process.stdout.write(output, 'utf8');
    }
}

main();
