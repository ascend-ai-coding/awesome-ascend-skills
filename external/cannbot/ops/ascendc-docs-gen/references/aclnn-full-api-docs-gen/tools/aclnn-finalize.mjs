#!/usr/bin/env node
// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import productPolicy from './scripts/lib/product-document-policy.js';
import sourceCatalog from './scripts/lib/aclnn-source-catalog.js';
import { loadBoundaryContext, validatePublicDocumentBoundary } from './scripts/lib/public-document-boundary.js';
import { validateAclnnDocument } from './scripts/lib/aclnn-document-validator.js';
import { validateSourceSnapshot } from './scripts/lib/source-snapshot.js';
import { mapOutsideCodeBlocks } from './scripts/lib/markdown-resource-links.js';
import readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';

const { markdownParts, normalizeProductNames, validateProductDocument, productNames } = productPolicy;
const FORMULA_PLACEHOLDER = '{{ACLNN_FORMULA_BLOCK_PLACEHOLDER}}';
const PROTOTYPE_PLACEHOLDER = '{{ACLNN_PROTOTYPE_BLOCK_PLACEHOLDER}}';
const EXECUTE_INTERFACE_PLACEHOLDER = '{{ACLNN_EXECUTE_INTERFACE_BLOCK_PLACEHOLDER}}';

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.operatorDir) {
    usage();
    process.exit(1);
  }

  const operatorDir = path.resolve(args.operatorDir);
  await assertDirectory(operatorDir, '算子目录');

  const interfaceName = args.interfaceName || await chooseOutputInterface(operatorDir);
  if (!/^aclnn[A-Za-z0-9_]+$/.test(interfaceName)) throw new Error('无效的 ACLNN 接口名');
  const outputsDir = path.join(operatorDir, 'outputs', interfaceName);
  await assertDirectory(outputsDir, '阶段产物目录');

  const skeletonPath = path.join(outputsDir, '4_generate_skeleton.md');
  const formulaPath = path.join(outputsDir, '3_formula.md');
  const patchPath = path.join(outputsDir, '5_review_patch.json');
  const callExamplePath = path.join(outputsDir, '0_call_example.md');
  const prototypePath = path.join(outputsDir, '0_prototype.md');

  const skeleton = await readRequired(skeletonPath);
  const formulaReport = await readRequired(formulaPath);
  const callExample = await readRequired(callExamplePath);
  const prototypeReport = await readRequired(prototypePath);
  const planPath = path.join(outputsDir, '0_extract_plan.json');
  const plan = await fs.readFile(planPath, 'utf8').catch(error => {
    if (error.code === 'ENOENT') return null;
    throw error;
  });
  if (plan === null) throw new Error('缺少源码快照，请重新预处理并取证');
  const extractPlan = JSON.parse(plan);
  sourceCatalog.validatePrototypeSnapshot(extractPlan, prototypeReport);
  validateSourceSnapshot(extractPlan.sourceSnapshot);
  const prototypeBlock = extractPrototypeBlock(prototypeReport);
  const formulaBlock = extractFormulaBlock(formulaReport);

  const generatedDoc = assembleDocument({ skeleton, formulaBlock, prototypeBlock, interfaceName, callExample });
  const candidatePath = path.join(outputsDir, '4_generate.md');
  const candidate = `${generatedDoc.trimEnd()}\n`;
  const candidateSha256 = createHash('sha256').update(candidate).digest('hex');
  if (args.prepareOnly) {
    await fs.writeFile(candidatePath, candidate, 'utf8');
    await fs.writeFile(path.join(outputsDir, '4_review_input.json'), `${JSON.stringify({
      document: '4_generate.md', reviewedDocumentSha256: candidateSha256
    }, null, 2)}\n`, 'utf8');
    console.log('完整候选文档已组装，请审阅 4_generate.md 并在补丁中记录 4_review_input.json 的摘要。');
    return;
  }
  if (await readRequired(candidatePath) !== candidate) {
    throw new Error('候选文档与当前输入不一致，请重新 --prepare 并审阅');
  }
  const reviewResult = await buildReviewedDoc({
    patchPath,
    skeleton,
    formulaBlock,
    prototypeBlock,
    interfaceName,
    callExample,
    candidateSha256
  });
  const reviewedDoc = reviewResult.doc;

  const finalDoc = normalizeDocSpacing(normalizeProductNames(normalizeAclnnDocFormat(reviewedDoc), 'aclnn'));
  validatePrototypeIntegrity(prototypeBlock, finalDoc);
  validateProductDocument(finalDoc, 'aclnn');
  validateAclnnDocument(finalDoc, interfaceName, prototypeBlock);
  const archivePath = path.join(outputsDir, `${interfaceName}_${formatCompactMinuteTimestamp()}.md`);
  const finalPath = path.join(operatorDir, 'docs', `${interfaceName}.md`);
  const warnings = validatePublicDocumentBoundary(finalDoc, {
    ...loadBoundaryContext('aclnn', outputsDir, finalPath), localRoots: [operatorDir]
  });
  for (const warning of warnings) console.warn('审阅提示：' + warning);
  await fs.writeFile(path.join(outputsDir, '4_generate.md'), `${generatedDoc.trimEnd()}\n`, 'utf8');
  await fs.writeFile(path.join(outputsDir, '5_review.md'), `${reviewedDoc.trimEnd()}\n`, 'utf8');
  await fs.mkdir(path.dirname(finalPath), { recursive: true });
  await fs.writeFile(archivePath, `${finalDoc.trimEnd()}\n`, 'utf8');
  await fs.writeFile(finalPath, `${finalDoc.trimEnd()}\n`, 'utf8');

  console.log(`已生成 4_generate.md: ${path.join(outputsDir, '4_generate.md')}`);
  console.log(`已生成 5_review.md: ${path.join(outputsDir, '5_review.md')}`);
  console.log(`已生成最终文档${warnings.length ? `（另有 ${warnings.length} 项审阅提示）` : ''}: ${finalPath}`);
}

async function buildReviewedDoc({ patchPath, skeleton, formulaBlock, prototypeBlock, interfaceName, callExample, candidateSha256 }) {
  const patch = parseReviewPatch(await readRequired(patchPath));
  if (patch.reviewedDocumentSha256 !== candidateSha256) {
    throw new Error('review 未绑定当前完整候选文档，请重新审阅并填写 reviewedDocumentSha256');
  }
  const reviewedSkeleton = applyEdits(skeleton, patch.skeletonEdits, 'skeletonEdits', '4_generate_skeleton.md');
  const reviewedFormulaBlock = applyEdits(formulaBlock, patch.formulaBlockEdits, 'formulaBlockEdits', '3_formula.md');
  const doc = assembleDocument({ skeleton: reviewedSkeleton, formulaBlock: reviewedFormulaBlock,
    prototypeBlock, interfaceName, callExample });
  return { doc: applyEdits(doc, patch.documentEdits, 'documentEdits', '4_generate.md') };
}

function assembleDocument({ skeleton, formulaBlock, prototypeBlock, interfaceName, callExample }) {
  const composed = composeGenerateDoc(normalizeDeterministicSkeleton(skeleton),
    normalizeDeterministicFormulaBlock(formulaBlock), prototypeBlock, interfaceName);
  return normalizeDocSpacing(normalizeProductNames(
    normalizeAclnnDocFormat(appendCallExample(composed, callExample)), 'aclnn'));
}

function validatePrototypeIntegrity(prototypeBlock, finalDoc) {
  const heading = /^## 函数原型[ \t]*\r?\n/m.exec(finalDoc);
  const section = heading ? finalDoc.slice(heading.index + heading[0].length).split(/\r?\n## /, 1)[0] : '';
  const blocks = content => [...content.matchAll(/```(?:cpp|c\+\+|c)\s*\n([\s\S]*?)```/g)]
    .map(match => match[1].replace(/\r\n/g, '\n').trim());
  // The parent template omits declaration terminators; compare the same display form.
  const declarations = content => blocks(content).map(block => block.replace(/;(\s*)$/, '$1').trim());
  const expected = declarations(prototypeBlock);
  const actual = declarations(section);
  if (!expected.length || JSON.stringify(expected) !== JSON.stringify(actual)) {
    throw new Error('最终文档的函数原型与 0_prototype.md 不一致；请修正审阅补丁，不能改写精确声明');
  }
}

function parseArgs(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === '--prepare') {
      result.prepareOnly = true;
    } else if (arg === '--interface') {
      result.interfaceName = argv[++index];
      if (!result.interfaceName || result.interfaceName.startsWith('--')) throw new Error('--interface 缺少参数值');
    } else if (!result.operatorDir) {
      result.operatorDir = arg;
    } else {
      throw new Error(`未知参数: ${arg}`);
    }
  }
  return result;
}

function usage() {
  console.log('用法: node tools/aclnn-finalize.mjs <operatorDir> [--interface aclnnXxx] [--prepare]');
}

async function assertDirectory(dir, label) {
  const stat = await fs.stat(dir).catch(() => null);
  if (!stat?.isDirectory()) {
    throw new Error(`${label}不存在或不是目录: ${dir}`);
  }
}

async function chooseOutputInterface(operatorDir) {
  const outputsRoot = path.join(operatorDir, 'outputs');
  const entries = await fs.readdir(outputsRoot, { withFileTypes: true }).catch(() => []);
  const candidates = [];
  for (const entry of entries) {
    if (!entry.isDirectory()) {
      continue;
    }
    const dir = path.join(outputsRoot, entry.name);
    const hasSkeleton = await exists(path.join(dir, '4_generate_skeleton.md'));
    const hasFormula = await exists(path.join(dir, '3_formula.md'));
    const hasPatch = await exists(path.join(dir, '5_review_patch.json'));
    if (hasSkeleton || hasFormula || hasPatch) {
      candidates.push(entry.name);
    }
  }

  if (candidates.length === 0) {
    throw new Error(`未在 outputs 目录下找到可合成的接口产物: ${outputsRoot}`);
  }
  if (candidates.length === 1) {
    console.log(`检测到 1 个可合成接口: ${candidates[0]}`);
    return candidates[0];
  }

  console.log(`检测到 ${candidates.length} 个可合成接口，请选择:`);
  candidates.forEach((name, index) => console.log(`  ${index + 1}. ${name}`));
  const rl = readline.createInterface({ input, output });
  try {
    while (true) {
      const answer = await rl.question('请输入序号或接口名: ');
      const trimmed = answer.trim();
      const numeric = Number(trimmed);
      if (Number.isInteger(numeric) && numeric >= 1 && numeric <= candidates.length) {
        return candidates[numeric - 1];
      }
      if (candidates.includes(trimmed)) {
        return trimmed;
      }
      console.log('输入无效，请重新选择。');
    }
  } finally {
    rl.close();
  }
}

async function exists(filePath) {
  return fs.access(filePath).then(() => true, () => false);
}

async function readRequired(filePath) {
  try {
    return await fs.readFile(filePath, 'utf8');
  } catch {
    throw new Error(`缺少必需文件: ${filePath}`);
  }
}

function extractFormulaBlock(formulaReport) {
  const lines = formulaReport.split(/\r?\n/);
  const start = lines.findIndex(line => /^##\s+最终文档公式块\s*$/.test(line.trim()));
  if (start < 0) {
    throw new Error('3_formula.md 中未找到 ## 最终文档公式块。');
  }
  let end = lines.length;
  for (let index = start + 1; index < lines.length; index++) {
    if (/^##\s+/.test(lines[index].trim())) {
      end = index;
      break;
    }
  }
  const body = lines.slice(start + 1, end).join('\n').trim();
  if (!body) {
    throw new Error('3_formula.md 中未找到有效的 ## 最终文档公式块。');
  }
  return body;
}

function extractPrototypeBlock(prototypeReport) {
  const body = prototypeReport.trim();
  if (!body) {
    throw new Error('0_prototype.md 为空，无法拼接函数原型。');
  }
  if (!/\baclnnStatus\s+aclnn[A-Za-z0-9_]*\s*\(/.test(body)) {
    throw new Error('0_prototype.md 中未找到有效 ACLNN 函数原型。');
  }
  return body;
}

function composeGenerateDoc(skeleton, formulaBlock, prototypeBlock, interfaceName) {
  let result = replacePlaceholderLine(skeleton, PROTOTYPE_PLACEHOLDER, prototypeBlock, '函数原型');
  result = replacePlaceholderLine(result, FORMULA_PLACEHOLDER, formulaBlock, '公式');
  result = replaceExecuteInterfaceBlock(result, interfaceName, prototypeBlock);
  return result.trimEnd();
}

function normalizeDeterministicSkeleton(content) {
  let result = content.replace(/\r\n/g, '\n');
  result = removeCallExampleSection(result);
  result = removeCodeAnalysisChecklist(result);
  result = removeIsolatedConstraintConfirmMarker(result);
  return cleanupBlankWhitespace(result);
}

function normalizeDeterministicFormulaBlock(content) {
  let result = content.replace(/\r\n/g, '\n');
  result = normalizeStandaloneMathDelimiters(result);
  result = normalizeInlineMathDelimiters(result);
  return cleanupBlankWhitespace(result);
}

function removeCallExampleSection(content) {
  return content.replace(/\n## 调用示例[\s\S]*$/, '');
}

function removeCodeAnalysisChecklist(content) {
  return content.replace(/\n## 代码分析清单[\s\S]*?(?=\n## |\n# |$)/g, '');
}

function normalizeInlineMathDelimiters(content) {
  return content.replace(/^(\s*)\$\$\s*(\S[^\n]*?\S)\s*\$\$\s*$/gm, (_match, indent, formula) => {
    return `${indent}$$\n${indent}${formula}\n${indent}$$`;
  });
}

function replaceExecuteInterfaceBlock(content, interfaceName, prototypeBlock) {
  if (content.includes(EXECUTE_INTERFACE_PLACEHOLDER)) {
    const block = hasAclnnSignature(prototypeBlock, `${interfaceName}GetWorkspaceSize`)
      ? buildExecuteInterfaceBlock(interfaceName)
      : '';
    return replacePlaceholderLine(content, EXECUTE_INTERFACE_PLACEHOLDER, block, '执行接口');
  }
  const executeHeading = new RegExp(`^##\\s+${escapeRegExp(interfaceName)}\\s*$`, 'm');
  if (executeHeading.test(content)) {
    return content;
  }
  throw new Error(`4_generate_skeleton.md 中未找到执行接口占位符 ${EXECUTE_INTERFACE_PLACEHOLDER}`);
}

function buildExecuteInterfaceBlock(interfaceName) {
  const workspaceInterfaceName = `${interfaceName}GetWorkspaceSize`;
  return `## ${interfaceName}

- **参数说明**

  <table style="table-layout: fixed; width: 1000px"><colgroup>
    <col style="width: 180px">
    <col style="width: 120px">
    <col style="width: 700px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口${workspaceInterfaceName}获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- 返回值

  aclnnStatus：返回状态码，具体参见[aclnn返回码](https://gitcode.com/cann/ops-math/blob/master/docs/zh/context/aclnn_return_code.md)。`;
}

function hasAclnnSignature(content, interfaceName) {
  return new RegExp(`\\baclnnStatus\\s+${escapeRegExp(interfaceName)}\\s*\\(`).test(content);
}

function replacePlaceholderLine(content, placeholder, replacement, label) {
  if (!content.includes(placeholder)) {
    throw new Error(`4_generate_skeleton.md 中未找到${label}占位符 ${placeholder}`);
  }
  const placeholderLineRegex = new RegExp(`^([ \\t]*)${escapeRegExp(placeholder)}[ \\t]*$`, 'm');
  const match = placeholderLineRegex.exec(content);
  if (!match) {
    return content.replace(placeholder, replacement).trimEnd();
  }

  const indent = match[1] || '';
  const indentedReplacement = indentBlock(replacement, indent);
  return (content.slice(0, match.index) + indentedReplacement + content.slice(match.index + match[0].length)).trimEnd();
}

function indentBlock(content, indent) {
  return content
    .split(/\r?\n/)
    .map(line => line.trim() ? `${indent}${line}` : '')
    .join('\n');
}

function parseReviewPatch(raw) {
  const parsed = parseFirstValidJsonObject(raw);
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('5_review_patch.json 不是有效 JSON 对象。');
  }
  if (!['pass', 'fix', 'fail'].includes(parsed.status)) {
    throw new Error('5_review_patch.json status 必须为 pass、fix 或 fail。');
  }
  if (parsed.status === 'fail') {
    throw new Error(parsed.failureReason || 'review patch 标记为 fail。');
  }
  const skeletonEdits = readEdits(parsed.skeletonEdits, 'skeletonEdits');
  const formulaBlockEdits = readEdits(parsed.formulaBlockEdits, 'formulaBlockEdits');
  const documentEdits = readEdits(parsed.documentEdits, 'documentEdits');
  const fields = { skeletonEdits, formulaBlockEdits, documentEdits, reviewedDocumentSha256: parsed.reviewedDocumentSha256 };
  if (parsed.status === 'pass' && (skeletonEdits.length > 0 || formulaBlockEdits.length > 0 || documentEdits.length > 0)) {
    console.warn('警告: 5_review_patch.json status=pass 但携带编辑项，已按 status=fix 处理。');
    return { status: 'fix', ...fields };
  }
  return { status: parsed.status, ...fields };
}

function parseFirstValidJsonObject(raw) {
  const candidates = extractJsonObjectCandidates(raw);
  const errors = [];
  for (const candidate of candidates) {
    try {
      return JSON.parse(candidate);
    } catch (error) {
      errors.push(error instanceof Error ? error.message : String(error));
    }
  }
  if (errors.length > 0) {
    throw new Error(`review patch 中未找到可解析的 JSON 对象。最后一次解析错误: ${errors[errors.length - 1]}`);
  }
  throw new Error('review patch 中未找到 JSON 对象。');
}

function extractJsonObjectCandidates(raw) {
  const text = raw.trim();
  const candidates = [];

  for (const match of text.matchAll(/```(?:json)?\s*([\s\S]*?)\s*```/gi)) {
    candidates.push(...extractBalancedJsonObjects(decodeStringWrappedJson(match[1].trim())));
  }

  let candidate = decodeStringWrappedJson(text);
  candidates.push(...extractBalancedJsonObjects(candidate));

  const deduped = [];
  const seen = new Set();
  for (const item of candidates) {
    if (!seen.has(item)) {
      seen.add(item);
      deduped.push(item);
    }
  }
  return deduped;
}

function decodeStringWrappedJson(candidate) {
  if (/^"(?:[\s\S]*)"$/.test(candidate)) {
    const decoded = JSON.parse(candidate);
    if (typeof decoded !== 'string') {
      throw new Error('review patch 字符串包装内容不是 JSON 文本。');
    }
    return decoded.trim();
  }
  return candidate;
}

function extractBalancedJsonObjects(text) {
  const objects = [];
  let start = -1;
  let depth = 0;
  let inString = false;
  let escaped = false;

  for (let index = 0; index < text.length; index++) {
    const ch = text[index];

    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (ch === '\\') {
        escaped = true;
      } else if (ch === '"') {
        inString = false;
      }
      continue;
    }

    if (ch === '"') {
      inString = true;
      continue;
    }

    if (ch === '{') {
      if (depth === 0) {
        start = index;
      }
      depth++;
      continue;
    }

    if (ch === '}' && depth > 0) {
      depth--;
      if (depth === 0 && start >= 0) {
        objects.push(text.slice(start, index + 1));
        start = -1;
      }
    }
  }

  return objects;
}

function readEdits(value, fieldName) {
  if (value === undefined) {
    return [];
  }
  if (!Array.isArray(value)) {
    throw new Error(`${fieldName} 必须为数组。`);
  }
  return value.map((item, index) => {
    if (!item || typeof item !== 'object' || Array.isArray(item)) {
      throw new Error(`${fieldName} 第 ${index + 1} 项不是有效对象。`);
    }
    const { section, oldText, newText, reason } = item;
    if ([section, oldText, newText, reason].some(value => typeof value !== 'string')) {
      throw new Error(`${fieldName} 第 ${index + 1} 项必须包含字符串字段 section、oldText、newText、reason。`);
    }
    return { section, oldText, newText, reason };
  });
}

function applyEdits(content, edits, fieldName, sourceName) {
  let result = content;
  for (const [index, edit] of edits.entries()) {
    if (!edit.oldText.trim()) {
      throw new Error(`${fieldName} 第 ${index + 1} 项 oldText 为空。`);
    }
    const literalCount = countLiteralOccurrences(result, edit.oldText);
    if (literalCount === 1) {
      result = result.replace(edit.oldText, () => edit.newText);
      continue;
    }

    const scopedResult = applyEditInScopedRegion(result, edit);
    if (scopedResult.applied) {
      result = scopedResult.content;
      continue;
    }

    const flexibleMatch = literalCount === 0 ? findUniqueWhitespaceFlexibleMatch(result, edit.oldText) : null;
    if (flexibleMatch) {
      result = result.slice(0, flexibleMatch.index) + edit.newText + result.slice(flexibleMatch.index + flexibleMatch.text.length);
      continue;
    }

    if (literalCount === 0 && edit.newText.trim() && result.includes(edit.newText)) {
      continue;
    }

    const flexibleCount = flexibleMatch ? 1 : countWhitespaceFlexibleOccurrences(result, edit.oldText);
    const actualCount = literalCount === 0 ? flexibleCount : literalCount;
    throw new Error(`${fieldName} 第 ${index + 1} 项 oldText 必须在 ${sourceName} 中唯一匹配，实际匹配 ${actualCount} 次。章节：${edit.section}`);
  }
  return result;
}

function applyEditInScopedRegion(content, edit) {
  const region = findScopedRegion(content, edit.section);
  if (!region) {
    return { applied: false, content };
  }

  const scopedContent = content.slice(region.start, region.end);
  const literalCount = countLiteralOccurrences(scopedContent, edit.oldText);
  if (literalCount === 1) {
    const nextScopedContent = scopedContent.replace(edit.oldText, () => edit.newText);
    return {
      applied: true,
      content: content.slice(0, region.start) + nextScopedContent + content.slice(region.end)
    };
  }

  if (literalCount === 0) {
    const flexibleMatch = findUniqueWhitespaceFlexibleMatch(scopedContent, edit.oldText);
    if (flexibleMatch) {
      const nextScopedContent = scopedContent.slice(0, flexibleMatch.index) + edit.newText + scopedContent.slice(flexibleMatch.index + flexibleMatch.text.length);
      return {
        applied: true,
        content: content.slice(0, region.start) + nextScopedContent + content.slice(region.end)
      };
    }
  }

  return { applied: false, content };
}

function findScopedRegion(content, section) {
  const headingRegion = findHeadingScopedRegion(content, section);
  if (headingRegion) {
    return headingRegion;
  }

  const marker = getSectionMarker(section);
  if (!marker) {
    return null;
  }

  const constraintStart = content.indexOf('## 约束说明');
  const searchStart = constraintStart >= 0 ? constraintStart : 0;
  const markerIndex = content.indexOf(marker, searchStart);
  if (markerIndex < 0) {
    return null;
  }

  const start = content.lastIndexOf('\n<details>', markerIndex);
  const normalizedStart = start >= 0 ? start + 1 : content.lastIndexOf('<details>', markerIndex);
  if (normalizedStart < 0) {
    return null;
  }

  const nextTopLevelDetails = content.indexOf('\n<details>', markerIndex + marker.length);
  const end = nextTopLevelDetails >= 0 ? nextTopLevelDetails + 1 : content.length;
  if (end <= normalizedStart) {
    return null;
  }

  return { start: normalizedStart, end };
}

function findHeadingScopedRegion(content, section) {
  const headingName = getSectionHeadingName(section);
  if (!headingName) {
    return null;
  }

  const headingRegex = new RegExp(`^##\\s+${escapeRegExp(headingName)}\\s*$`, 'm');
  const match = headingRegex.exec(content);
  if (!match || match.index === undefined) {
    return null;
  }

  const nextHeadingRegex = /^##\s+/gm;
  nextHeadingRegex.lastIndex = match.index + match[0].length;
  const nextHeading = nextHeadingRegex.exec(content);
  return {
    start: match.index,
    end: nextHeading?.index ?? content.length
  };
}

function getSectionHeadingName(section) {
  const match = section.match(/\b(aclnn[A-Za-z0-9_]*(?:GetWorkspaceSize)?)\b/);
  return match?.[1] ?? null;
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function getSectionMarker(section) {
  if (/Ascend\s*950PR|950DT/i.test(section)) {
    return '<term>Ascend 950PR&950DT系列产品</term>';
  }
  if (/Atlas A3/i.test(section)) return '<term>Atlas A3系列产品</term>';
  if (/Atlas A2/i.test(section)) return '<term>Atlas A2系列产品</term>';
  if (/训练系列产品/i.test(section)) {
    return '<term>Atlas训练系列产品</term>';
  }
  if (/推理系列产品/i.test(section)) {
    return '<term>Atlas推理系列产品</term>';
  }
  return null;
}

function countLiteralOccurrences(content, search) {
  if (!search) {
    return 0;
  }
  let count = 0;
  let index = 0;
  while (true) {
    const found = content.indexOf(search, index);
    if (found < 0) {
      break;
    }
    count += 1;
    index = found + search.length;
  }
  return count;
}

function findUniqueWhitespaceFlexibleMatch(content, needle) {
  const regex = buildWhitespaceFlexibleRegex(needle);
  const matches = Array.from(content.matchAll(regex));
  if (matches.length !== 1 || matches[0].index === undefined) {
    return null;
  }
  return { index: matches[0].index, text: matches[0][0] };
}

function countWhitespaceFlexibleOccurrences(content, needle) {
  return Array.from(content.matchAll(buildWhitespaceFlexibleRegex(needle))).length;
}

function buildWhitespaceFlexibleRegex(needle) {
  const escaped = needle
    .trim()
    .split(/(\s+)/)
    .filter(part => part.length > 0)
    .map(part => /\s+/.test(part) ? '\\s+' : part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
    .join('');
  return new RegExp(escaped, 'g');
}

function appendCallExample(doc, callExample) {
  const stripped = doc.replace(/\n## 调用示例[\s\S]*$/, '');
  const exampleSection = [
    '',
    '## 调用示例',
    '',
    '示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。',
    '',
    callExample.trim().replace(/^## 调用示例[ \t]*(?:\r?\n|$)/, '').trim()
  ].join('\n');
  return `${stripped.trimEnd()}\n${exampleSection}`;
}

function normalizeAclnnDocFormat(content) {
  let text = content.replace(/\r\n/g, '\n');
  text = mapOutsideCodeBlocks(text, normalizeStandaloneMathDelimiters);
  text = indentFormulaListItemBody(text);
  text = formatPrototypeSection(text);
  text = removeIsolatedConstraintConfirmMarker(text);
  text = mapOutsideCodeBlocks(text, cleanupBlankWhitespace);
  return text.trimEnd();
}

function normalizeStandaloneMathDelimiters(content) {
  return content.replace(/^(\s*)\$(\s*)$/gm, (_match, indent, trailing) => `${indent}$$${trailing}`);
}

function indentFormulaListItemBody(content) {
  const heading = '- 计算公式<span style="color:#d97706;">【需用户确认】</span>：';
  const start = content.indexOf(heading);
  if (start < 0) {
    return content;
  }

  const nextHeading = content.indexOf('\n## ', start + heading.length);
  const end = nextHeading >= 0 ? nextHeading : content.length;
  const before = content.slice(0, start);
  const formulaSection = content.slice(start, end);
  const after = content.slice(end);
  const lines = formulaSection.split('\n');
  const normalized = lines.map((line, index) => {
    if (index === 0 || line.trim() === '') {
      return line;
    }
    if (/^\s{2,}\S/.test(line)) {
      return line;
    }
    return `  ${line}`;
  }).join('\n');
  return before + normalized + after;
}

function formatPrototypeSection(content) {
  return content.replace(/(^## 函数原型[ \t]*\n)([\s\S]*?)(?=^## |$(?![\s\S]))/m,
    (_match, heading, section) => heading + section.replace(
      /(```(?:cpp|c\+\+|c)[ \t]*\n)([\s\S]*?)(\n```)/g,
      (_block, open, declaration, close) => open + declaration.replace(/;(\s*)$/, '$1') + close));
}

function removeIsolatedConstraintConfirmMarker(content) {
  return content.replace(
    /(## 约束说明\s*\n\s*)<span style="color:#d97706;">【需用户确认】<\/span>\s*\n\s*/m,
    '$1'
  );
}

function cleanupBlankWhitespace(content) {
  return content
    .split('\n')
    .map(line => line.trim() === '' ? '' : line.replace(/[ \t]+$/g, ''))
    .join('\n')
    .replace(/\n{4,}/g, '\n\n\n');
}

function normalizeDocSpacing(content) {
  let inMathBlock = false;
  return markdownParts(content).map(part => part.kind !== 'text' ? part.text :
    part.text.split(/\r?\n/).map(line => {
      if (/^\s*\$\$\s*$/.test(line)) inMathBlock = !inMathBlock;
      return inMathBlock ? line : normalizeDocSpacingLine(line);
    }).join('\n')).join('');
}

function normalizeDocSpacingLine(line) {
  if (!line.trim()) {
    return line;
  }

  const placeholders = [];
  let result = line;
  const protect = pattern => {
    result = result.replace(pattern, match => {
      const key = `\u0000${placeholders.length}\u0000`;
      placeholders.push(match);
      return key;
    });
  };

  protect(/`[^`]*`/g);
  protect(/<term\b[^>]*>[\s\S]*?<\/term>/gi);
  protect(/!??\[[^\]]*\]\([^)]*\)/g);
  protect(/https?:\/\/[^\s)]+/g);
  protect(/<[^>]+>/g);
  protect(/&[A-Za-z0-9#]+;/g);
  protect(new RegExp(productNames.map(escapeRegExp).join('|'), 'g'));

  result = result
    .replace(/(\p{Script=Han})\s+(\p{Script=Han})/gu, '$1$2')
    .replace(/(\p{Script=Han})\s+([A-Za-z])/gu, '$1$2')
    .replace(/([A-Za-z])\s+(\p{Script=Han})/gu, '$1$2')
    .replace(/(\p{Script=Han})\s+([0-9])/gu, '$1$2')
    .replace(/([0-9])\s+(\p{Script=Han})/gu, '$1$2')
    .replace(/([A-Za-z]) {2,}([A-Za-z])/g, '$1 $2')
    .replace(/[A-Za-z0-9_]+/g, normalizeDocWordToken);

  return result.replace(/\u0000(\d+)\u0000/g, (_match, index) => placeholders[Number(index)] || '');
}

function normalizeDocWordToken(token) {
  if (
    token.includes('_') ||
    /^aclnn[A-Za-z0-9]*$/.test(token) ||
    /^[A-Z][A-Z0-9]*$/.test(token) ||
    /^[A-Z]+[a-z]+[A-Za-z0-9]*$/.test(token) ||
    /^[a-z]+[A-Z][A-Za-z0-9]*$/.test(token) ||
    /^[A-Za-z]+[0-9][A-Za-z0-9]*$/.test(token)
  ) {
    return token;
  }
  return token
    .replace(/([A-Za-z])(\d)/g, '$1 $2')
    .replace(/(\d)([A-Za-z])/g, '$1 $2');
}

function formatCompactMinuteTimestamp() {
  const now = new Date();
  const pad = value => String(value).padStart(2, '0');
  return `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}${pad(now.getHours())}${pad(now.getMinutes())}`;
}

export { validatePrototypeIntegrity };

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => {
    console.error(`错误: ${error instanceof Error ? error.message : String(error)}`);
    process.exitCode = 1;
  });
}
