// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
'use strict';

const { documentHeadings, codeBlocks } = require('./markdown-resource-links.js');

function validateAclnnDocument(document, interfaceName, prototype) {
    const headings = documentHeadings(document);
    const titles = headings.filter(item => item.level === 1);
    if (titles.length !== 1 || titles[0].title !== interfaceName) {
        throw new Error('最终文档必须包含唯一且匹配目标 API 的一级标题');
    }
    const chapters = headings.filter(item => item.level === 2);
    const names = chapters.map(item => item.title);
    if (new Set(names).size !== names.length) throw new Error('最终文档必需章节不得重复');
    const functions = new Set([...prototype.matchAll(/\baclnnStatus\s+(aclnn\w+)\s*\(/g)].map(item => item[1]));
    const twoStage = functions.has(interfaceName + 'GetWorkspaceSize');
    if (!functions.has(interfaceName)) throw new Error('精确原型缺少目标执行接口');
    const expected = ['产品支持情况', '功能说明', '函数原型',
        ...(twoStage ? [interfaceName + 'GetWorkspaceSize', interfaceName] : []), '约束说明', '调用示例'];
    let previous = -1;
    for (const name of expected) {
        const position = names.indexOf(name);
        if (position <= previous) throw new Error('最终文档必需章节缺失或顺序错误：' + name);
        previous = position;
    }
    if (titles[0].lines[0] >= chapters[0].lines[0]) throw new Error('一级标题必须位于必需章节之前');
    const lines = document.split('\n');
    for (const [index, chapter] of chapters.entries()) {
        const body = lines.slice(chapter.lines[1], chapters[index + 1]?.lines[0] ?? lines.length).join('\n');
        if (!body.replace(/<!--[\s\S]*?-->/g, '').trim()) throw new Error('最终文档章节不能为空：' + chapter.title);
    }
    if (/\{\{ACLNN_[A-Z_]+\}\}/.test(document)) throw new Error('最终文档包含未组装的流程占位符');
    const start = chapters[names.indexOf('调用示例')].lines[1];
    const end = chapters[names.indexOf('调用示例') + 1]?.lines[0] ?? lines.length;
    const section = lines.slice(start, end).join('\n');
    const code = codeBlocks(section).filter(block => ['', 'c', 'cpp', 'c++', 'cc', 'cxx', 'h', 'hpp'].includes(block.language))
        .map(block => block.content).join('\n')
        .replace(/R"([^\s()\\]{0,16})\([\s\S]*?\)\1"|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\/\/[^\n]*|\/\*[\s\S]*?\*\//g,
            value => value.replace(/[^\n]/g, ' '))
        .replace(/\baclnnStatus\s+(?:[A-Za-z_]\w*\s+)*?aclnn\w+(?=\s*\()/g,
            value => value.replace(/[^\n]/g, ' '));
    for (const name of [interfaceName, ...(twoStage ? [interfaceName + 'GetWorkspaceSize'] : [])]) {
        if (!new RegExp('\\b' + name + '\\s*\\(').test(code)) throw new Error('调用示例缺少目标接口实际调用：' + name);
    }
}

module.exports = { validateAclnnDocument };
