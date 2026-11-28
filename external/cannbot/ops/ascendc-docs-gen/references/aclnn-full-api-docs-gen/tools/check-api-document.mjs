// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
import fs from 'node:fs/promises';
import path from 'node:path';
import productPolicy from './scripts/lib/product-document-policy.js';
import { loadBoundaryContext, validatePublicDocumentBoundary } from './scripts/lib/public-document-boundary.js';
import { validatePrototypeIntegrity } from './aclnn-finalize.mjs';

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.document) throw new Error('缺少 --document <最终文档.md>');
  const finalDoc = await fs.readFile(path.resolve(args.document), 'utf8');
  const localRoots = args.repo ? [path.resolve(args.repo)] : [];
  productPolicy.validateProductDocument(finalDoc, 'aclnn');
  const warnings = validatePublicDocumentBoundary(finalDoc, {
    ...loadBoundaryContext('aclnn', args.sessionDir, path.resolve(args.document)), localRoots
  });
  for (const warning of warnings) console.warn('审阅提示：' + warning);

  if (args.prototypeFile) {
    const prototypeDoc = await fs.readFile(path.resolve(args.prototypeFile), 'utf8');
    validatePrototypeIntegrity(prototypeDoc, finalDoc);
  }
  console.log('产品、标签和公开内容检查通过' + (warnings.length ? `，另有 ${warnings.length} 项审阅提示` : '') + '（只读，未写入文件；' +
    (args.prototypeFile ? '已比对原型' : '未提供原型依据，不代表技术事实已核验') + '）。');
}

function parseArgs(argv) {
  const options = { '--document': 'document', '--repo': 'repo', '--prototype-file': 'prototypeFile', '--session-dir': 'sessionDir' };
  const args = {};
  for (let index = 0; index < argv.length; index++) {
    const option = argv[index];
    if (!Object.hasOwn(options, option)) throw new Error('未知参数: ' + option);
    const value = argv[++index];
    if (!value || value.startsWith('--')) throw new Error(option + ' 缺少参数值');
    args[options[option]] = value;
  }

  return args;
}

main().catch(error => {
  console.error('错误: ' + (error instanceof Error ? error.message : String(error)));
  process.exitCode = 1;
});
