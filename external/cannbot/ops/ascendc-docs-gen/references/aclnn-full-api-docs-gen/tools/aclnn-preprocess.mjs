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
import { spawn } from 'node:child_process';
import readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';
import tracker from './scripts/lib/cross-dir-tracker.js';
import sourceCatalog from './scripts/lib/aclnn-source-catalog.js';
import { captureSourceSnapshot, validateSourceSnapshot } from './scripts/lib/source-snapshot.js';

const TOOL_ROOT = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_SCRIPTS_DIR = path.join(TOOL_ROOT, 'scripts');
const DEFAULT_SKILL_ROOT = path.dirname(TOOL_ROOT);

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.operatorDir) {
    usage();
    process.exit(1);
  }

  const operatorDir = path.resolve(args.operatorDir);
  const scriptsDir = path.resolve(args.scriptsDir || args.skillDir || process.env.ACLNN_PREPROCESS_SCRIPTS_DIR || DEFAULT_SCRIPTS_DIR);
  const skillRoot = path.resolve(args.skillRoot || process.env.ACLNN_SKILL_ROOT || DEFAULT_SKILL_ROOT);
  const errorCodesFile = path.resolve(args.errorCodesFile || await resolveSkillConfigFile(skillRoot, 'error_codes.yml'));
  const productMappingFile = path.resolve(args.productMappingFile || await resolveSkillConfigFile(skillRoot, 'product_mapping.yml'));
  const projectRoot = args.projectRoot ? path.resolve(args.projectRoot) : await detectProjectRoot(operatorDir);

  await assertDirectory(operatorDir, '算子目录');
  await assertDirectory(scriptsDir, '预处理脚本目录');

  const catalog = sourceCatalog.collectAclnnSources(operatorDir, { prototypeManifest: args.prototypeManifest, projectRoot });
  const interfaces = catalog.interfaces;
  if (interfaces.length === 0) {
    throw new Error(`尚未发现可核验的 ACLNN 声明：${operatorDir}。接口可能由 opbuild 自动生成；请通过 --prototype-manifest 提供同一源码版本的生成或安装头文件来源清单，不根据 OpDef 猜测原型。`);
  }

  const interfaceName = args.interfaceName || await chooseInterface(interfaces);
  if (!interfaces.includes(interfaceName)) {
    throw new Error(`指定接口 ${interfaceName} 不在检测结果中：${interfaces.join(', ')}`);
  }

  const closure = tracker.trackExternalFiles(operatorDir, projectRoot);
  const sourceSnapshot = captureSourceSnapshot([...closure.operatorFiles, ...closure.externalFiles,
    ...tracker.collectFiles(path.join(operatorDir, 'examples')), ...catalog.sources.map(item => item.file),
    ...args.sourceFiles.map(file => path.resolve(projectRoot, file))]);

  const outputsDir = path.join(operatorDir, 'outputs', interfaceName);
  await fs.mkdir(outputsDir, { recursive: true });
  await archivePreviousRun(outputsDir);

  console.log(`算子目录: ${operatorDir}`);
  console.log(`项目根目录: ${projectRoot}`);
  console.log(`预处理脚本目录: ${scriptsDir}`);
  console.log(`Skill 根目录: ${skillRoot}`);
  console.log(`错误码配置: ${errorCodesFile}`);
  console.log(`产品映射配置: ${productMappingFile}`);
  console.log(`目标接口: ${interfaceName}`);
  console.log(`输出目录: ${outputsDir}`);

  const tasks = [
    {
      name: 'prototype',
      script: 'extract-prototype.js',
      args: [operatorDir, interfaceName, '--project-root', projectRoot,
        ...(args.prototypeManifest ? ['--prototype-manifest', path.resolve(args.prototypeManifest)] : []),
        '--output', path.join(outputsDir, '0_prototype.md')]
    },
    {
      name: 'cross_dir',
      script: 'extract-cross-dir.js',
      args: [operatorDir, projectRoot, '--output', path.join(outputsDir, '0_cross_dir.md')]
    },
    {
      name: 'chip_evidence',
      script: 'extract-chip-evidence.js',
      args: [operatorDir, projectRoot, '--product-mapping-file', productMappingFile, '--output', path.join(outputsDir, '0_chip_evidence.md')]
    },
    {
      name: 'error_codes',
      script: 'extract-error-codes.js',
      args: [operatorDir, projectRoot, '--error-codes-file', errorCodesFile, '--output', path.join(outputsDir, '0_error_codes.md')]
    },
    {
      name: 'call_example',
      script: 'extract-call-example.js',
      args: [operatorDir, interfaceName, '--product-mapping-file', productMappingFile, '--output', path.join(outputsDir, '0_call_example.md')]
    }
  ];

  const results = await Promise.all(tasks.map(task => runPreprocessTask(scriptsDir, outputsDir, task)));
  for (const result of results) {
    const label = result.ok ? 'OK' : 'FAILED';
    console.log(`[${label}] ${result.name}`);
  }

  if (results.some(result => !result.ok)) {
    throw new Error('预处理失败；不得继续 LLM 阶段或最终合成，请检查 0_*_FAILED.md。');
  }
  validateSourceSnapshot(sourceSnapshot);
  await writeExtractPlan(outputsDir, operatorDir, interfaceName, catalog, sourceSnapshot);
  console.log(`[OK] extract_plan`);
  console.log(`预处理完成: ${outputsDir}`);
}

async function archivePreviousRun(outputsDir) {
  const entries = await fs.readdir(outputsDir, { withFileTypes: true });
  const artifacts = entries.filter(entry => entry.isFile() && /^(?:[0-5]_.*\.(?:md|json)|aclnn[A-Za-z0-9_]+_\d+\.md)$/.test(entry.name));
  if (artifacts.length === 0) return;
  const archiveDir = await fs.mkdtemp(path.join(outputsDir, '.previous-'));
  for (const entry of artifacts) {
    await fs.rename(path.join(outputsDir, entry.name), path.join(archiveDir, entry.name));
  }
}

function parseArgs(argv) {
  const result = { sourceFiles: [] };
  const options = { '--interface': 'interfaceName', '--skill-dir': 'skillDir',
    '--skill-root': 'skillRoot', '--error-codes-file': 'errorCodesFile',
    '--product-mapping-file': 'productMappingFile', '--scripts-dir': 'scriptsDir',
    '--project-root': 'projectRoot', '--prototype-manifest': 'prototypeManifest' };
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === '--source-file') {
      const value = argv[++index];
      if (!value || value.startsWith('--')) throw new Error('--source-file 缺少参数值');
      result.sourceFiles.push(value);
    } else if (Object.hasOwn(options, arg)) {
      const value = argv[++index];
      if (!value || value.startsWith('--')) throw new Error(`${arg} 缺少参数值`);
      result[options[arg]] = arg === '--interface' ? normalizeInterfaceName(value) : value;
    } else if (!arg.startsWith('--') && !result.operatorDir) result.operatorDir = arg;
    else throw new Error(`未知参数: ${arg}`);
  }
  return result;
}

function usage() {
  console.log('用法: node tools/aclnn-preprocess.mjs <operatorDir> [--interface aclnnXxx] [--prototype-manifest <json>] [--project-root <dir>] [--scripts-dir <dir>] [--skill-root <dir>] [--error-codes-file <file>] [--product-mapping-file <file>] [--source-file <额外直接证据文件>]');
}

async function assertDirectory(dir, label) {
  const stat = await fs.stat(dir).catch(() => null);
  if (!stat?.isDirectory()) {
    throw new Error(`${label}不存在或不是目录: ${dir}`);
  }
}

async function resolveSkillConfigFile(skillRoot, filename) {
  const preferred = path.join(skillRoot, 'config', filename);
  if (await fileExists(preferred)) {
    return preferred;
  }
  return path.join(skillRoot, 'references', filename);
}

async function fileExists(filePath) {
  const stat = await fs.stat(filePath).catch(() => null);
  return Boolean(stat?.isFile());
}

async function detectProjectRoot(operatorDir) {
  const inferred = tracker.detectProjectRoot(operatorDir);
  if (inferred) return inferred;
  for (let dir = operatorDir; ; dir = path.dirname(dir)) {
    if (await fs.stat(path.join(dir, '.git')).catch(() => null)) return dir;
    if (path.dirname(dir) === dir) break;
  }
  // A standalone operator is valid; unresolved includes remain explicit evidence gaps.
  return operatorDir;
}

function normalizeInterfaceName(name) {
  return name.replace(/GetWorkspaceSize$/, '').trim();
}

async function chooseInterface(interfaces) {
  if (interfaces.length === 1) {
    console.log(`检测到 1 个接口: ${interfaces[0]}`);
    return interfaces[0];
  }

  console.log(`检测到 ${interfaces.length} 个接口，请选择要生成的接口:`);
  interfaces.forEach((name, index) => console.log(`  ${index + 1}. ${name}`));

  const rl = readline.createInterface({ input, output });
  try {
    while (true) {
      const answer = await rl.question('请输入序号或接口名: ');
      const trimmed = answer.trim();
      const numeric = Number(trimmed);
      if (Number.isInteger(numeric) && numeric >= 1 && numeric <= interfaces.length) {
        return interfaces[numeric - 1];
      }
      const normalized = normalizeInterfaceName(trimmed);
      if (interfaces.includes(normalized)) {
        return normalized;
      }
      console.log('输入无效，请重新选择。');
    }
  } finally {
    rl.close();
  }
}

function runPreprocessTask(scriptsDir, outputsDir, task) {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(scriptsDir, task.script);
    const child = spawn(process.execPath, [scriptPath, ...task.args], {
      cwd: scriptsDir,
      stdio: ['ignore', 'pipe', 'pipe']
    });

    let stdoutText = '';
    let stderrText = '';
    child.stdout.on('data', chunk => { stdoutText += chunk.toString(); });
    child.stderr.on('data', chunk => { stderrText += chunk.toString(); });
    child.on('error', reject);
    child.on('close', async code => {
      try {
        const outputFile = task.args[task.args.indexOf('--output') + 1];
        const stat = await fs.stat(outputFile).catch(() => null);
        if (code === 0 && stat?.isFile() && stat.size > 0) {
          resolve({ name: task.name, ok: true });
          return;
        }
        await writeFailedFile(outputsDir, task.name, scriptPath, code,
          stderrText || stdoutText || '子任务未生成非空输出文件');
        resolve({ name: task.name, ok: false });
      } catch (error) {
        reject(error);
      }
    });
  });
}

async function writeFailedFile(outputsDir, name, scriptPath, code, detail) {
  const content = [
    `# 预处理 ${name} 失败`,
    '',
    `**失败时间**：${formatMinuteTimestamp()}`,
    `**失败阶段**：预处理 — ${name}`,
    '',
    '## 失败原因',
    '',
    `脚本退出码：${code}`,
    '',
    '## 执行脚本',
    '',
    scriptPath,
    '',
    '## 错误输出摘要',
    '',
    detail.trim() || '无',
    ''
  ].join('\n');
  await fs.writeFile(path.join(outputsDir, `0_${name}_FAILED.md`), content, 'utf8');
}

async function writeExtractPlan(outputsDir, operatorDir, interfaceName, catalog, sourceSnapshot) {
  const files = {
    prototype: await readOptional(path.join(outputsDir, '0_prototype.md')),
    cross_dir: await readOptional(path.join(outputsDir, '0_cross_dir.md')),
    chip_evidence: await readOptional(path.join(outputsDir, '0_chip_evidence.md')),
    error_codes: await readOptional(path.join(outputsDir, '0_error_codes.md')),
    call_example: await readOptional(path.join(outputsDir, '0_call_example.md'))
  };
  const hits = catalog.sources
    .filter(source => catalog.signatures.some(sig => sig.file === source.file && normalizeInterfaceName(sig.name) === interfaceName))
    .map(source => ({ path: normalizePath(source.file), source: 'prototype', reason: '已核验声明来源；只证明原型，不替代行为取证' }));
  for (const [source, content] of Object.entries(files)) {
    hits.push(...collectPathHits(content, source, operatorDir));
  }
  const keyFiles = dedupe(hits, item => `${item.path}:${item.line || 0}`).slice(0, 100);
  const aroundItems = keyFiles
    .filter(item => item.line)
    .filter(item => ['chip_evidence', 'error_codes'].includes(item.source) || /aclnn/i.test(item.path))
    .sort(comparePlanPriority)
    .slice(0, 40)
    .map(item => ({ path: item.path, line: item.line, before: 60, after: 80, reason: item.reason }));
  const functionItems = buildFunctionItems(keyFiles, interfaceName, files.prototype);

  const plan = {
    schemaVersion: 1,
    kind: 'aclnn_extract_plan',
    prototypeReportSha256: createHash('sha256').update(files.prototype).digest('hex'),
    sourceSnapshot,
    prototypeSources: catalog.sources.filter(source => catalog.signatures.some(sig =>
      sig.file === source.file && normalizeInterfaceName(sig.name) === interfaceName)),
    target: {
      interfaceName,
      operatorDir: normalizePath(operatorDir)
    },
    usage: {
      purpose: '减少 extract 阶段首轮工具探索耗时',
      sourceOfTruth: '源码和原始 evidence 优先；本计划只提供读取路线，不替代源码核验',
      requiredEvidenceFiles: ['cross_dir', 'chip_evidence', 'error_codes']
    },
    keyFiles,
    recommendedToolCalls: {
      read_file_around_batch: { items: aroundItems },
      extract_function_batch: { items: functionItems }
    }
  };
  await fs.writeFile(path.join(outputsDir, '0_extract_plan.json'), `${JSON.stringify(plan, null, 2)}\n`, 'utf8');
}

async function readOptional(filePath) {
  return fs.readFile(filePath, 'utf8').catch(() => '');
}

function collectPathHits(content, source, operatorDir) {
  const hits = [];
  const lines = content.split(/\r?\n/);
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index];
    for (const match of line.matchAll(/\*\*([^*`]+?\.(?:cpp|cc|cxx|c|h|hpp|hh|md))[:：](\d+)\*\*/gi)) {
      const resolvedPath = resolveEvidencePath(match[1], operatorDir);
      hits.push({
        path: normalizePath(resolvedPath),
        line: Number(match[2]),
        source,
        sourceLine: index + 1,
        reason: classifyPath(resolvedPath)
      });
    }
    for (const match of line.matchAll(/\b[A-Za-z]:[\\/][^`'"\s|)）]+/g)) {
      const parsed = parsePathWithLine(match[0], line.slice((match.index || 0) + match[0].length));
      if (!/\.(cpp|cc|cxx|c|h|hpp|hh|md)$/i.test(parsed.path)) {
        continue;
      }
      hits.push({
        path: normalizePath(parsed.path),
        line: parsed.line,
        source,
        sourceLine: index + 1,
        reason: classifyPath(parsed.path)
      });
    }
  }
  return hits;
}

function resolveEvidencePath(filePath, operatorDir) {
  const normalized = normalizePath(filePath).replace(/^`|`$/g, '');
  if (path.isAbsolute(normalized)) {
    return normalized;
  }
  const operatorBase = path.basename(operatorDir);
  if (normalized === operatorBase || normalized.startsWith(`${operatorBase}/`)) {
    return normalizePath(path.join(path.dirname(operatorDir), normalized));
  }
  return normalizePath(path.join(operatorDir, normalized));
}

function comparePlanPriority(a, b) {
  return planPriority(a) - planPriority(b) || a.path.localeCompare(b.path) || (a.line || 0) - (b.line || 0);
}

function planPriority(item) {
  const p = item.path.toLowerCase();
  if (p.includes('/op_host/op_api/aclnn_')) return 1;
  if (p.endsWith('_def.cpp')) return 2;
  if (p.includes('infershape')) return 3;
  if (p.includes('/op_tiling/')) return 4;
  if (p.includes('/op_kernel/')) return 5;
  return 9;
}

function buildFunctionItems(keyFiles, interfaceName, prototypeReport = '') {
  const aclnnCpp = keyFiles
    .map(item => item.path)
    .find(filePath => /\/op_(api|host)\/.*aclnn.*\.cpp$/i.test(filePath));
  if (!aclnnCpp) return [];
  const items = [];
  if (hasAclnnSignature(prototypeReport, `${interfaceName}GetWorkspaceSize`)) {
    items.push({
      path: aclnnCpp,
      functionName: `${interfaceName}GetWorkspaceSize`,
      reason: '目标两段式 GetWorkspaceSize 接口实现'
    });
  }
  if (hasAclnnSignature(prototypeReport, interfaceName)) {
    items.push({
      path: aclnnCpp,
      functionName: interfaceName,
      reason: '目标 ACLNN 接口实现'
    });
  }
  return items;
}

function hasAclnnSignature(content, interfaceName) {
  return new RegExp(`\\baclnnStatus\\s+${escapeRegExp(interfaceName)}\\s*\\(`).test(content);
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function parsePathWithLine(rawPath, tail) {
  const stripped = rawPath.replace(/[，,；;。]+$/g, '');
  const withLine = stripped.match(/^(.+\.(?:cpp|cc|cxx|c|h|hpp|hh|md))[:：](\d+)\D*$/i);
  if (withLine) {
    return { path: withLine[1], line: Number(withLine[2]) };
  }
  const pathOnly = stripped.match(/^(.+\.(?:cpp|cc|cxx|c|h|hpp|hh|md))[:：]?$/i);
  const tailLine = tail.match(/^\s*[:：]?\s*(?:line|行号|行)?\s*(\d+)\b/i);
  return { path: pathOnly ? pathOnly[1] : stripped, line: tailLine ? Number(tailLine[1]) : undefined };
}

function classifyPath(filePath) {
  const normalized = normalizePath(filePath).toLowerCase();
  if (normalized.endsWith('_def.cpp')) return 'OP_TYPE_REGISTER/_def.cpp，优先用于 dtype/format/产品支持分析';
  if (normalized.includes('/op_tiling/')) return 'tiling 文件，优先用于 shape/attr/workspace 约束分析';
  if (normalized.includes('/op_kernel/')) return 'kernel 文件，优先用于芯片宏分支和实现约束分析';
  if (normalized.includes('/op_api/') && /aclnn.*\.(cpp|h|hpp)$/.test(normalized)) return 'aclnn 接口文件，优先用于原型/参数/运行时校验分析';
  if (normalized.includes('/op_host/')) return 'host/proto 文件，优先用于 infer shape、注册和 host 校验分析';
  return '跨目录/include 候选文件，按 evidence 说明核验';
}

function dedupe(items, keyFn) {
  const seen = new Set();
  const result = [];
  for (const item of items) {
    const key = keyFn(item);
    if (!seen.has(key)) {
      seen.add(key);
      result.push(item);
    }
  }
  return result;
}

function normalizePath(filePath) {
  return filePath.replace(/\\/g, '/');
}

function formatMinuteTimestamp() {
  const now = new Date();
  const pad = value => String(value).padStart(2, '0');
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())} ${pad(now.getHours())}:${pad(now.getMinutes())}`;
}

main().catch(error => {
  console.error(`错误: ${error instanceof Error ? error.message : String(error)}`);
  process.exit(1);
});
