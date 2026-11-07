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
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import discovery from './scripts/detect-interfaces.js';

const TOOL_ROOT = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_SCRIPTS_DIR = path.join(TOOL_ROOT, 'scripts');
const PREPROCESS_SCRIPT = path.join(TOOL_ROOT, 'aclnn-preprocess.mjs');
const FINALIZE_SCRIPT = path.join(TOOL_ROOT, 'aclnn-finalize.mjs');

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.mode || !args.inputDir) {
    usage();
    process.exit(1);
  }

  const inputDir = path.resolve(args.inputDir);
  await assertDirectory(inputDir, '输入目录');

  if (args.mode === 'detect') {
    const targets = await detectTargets(inputDir, args);
    printDetectedTargets(targets);
    return;
  }

  if (args.mode === 'preprocess') {
    await runBatchPreprocess(inputDir, args);
    return;
  }

  if (args.mode === 'finalize') {
    await runBatchFinalize(inputDir, args);
    return;
  }

  throw new Error(`未知模式: ${args.mode}`);
}

function parseArgs(argv) {
  const result = {
    mode: argv[0],
    inputDir: argv[1],
    include: [],
    exclude: [],
    failFast: false,
    continueOnError: true,
    passthrough: []
  };

  for (let index = 2; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === '--include') {
      result.include = splitNameList(requiredValue(argv, ++index, arg));
      if (!result.include.length) throw new Error('--include 不能为空');
    } else if (arg === '--exclude') {
      result.exclude = splitNameList(requiredValue(argv, ++index, arg));
      if (!result.exclude.length) throw new Error('--exclude 不能为空');
    } else if (arg === '--fail-fast') {
      result.failFast = true;
      result.continueOnError = false;
    } else if (arg === '--continue-on-error') {
      result.continueOnError = true;
      result.failFast = false;
    } else if (isForwardedOption(arg)) {
      result.passthrough.push(arg);
      if (optionRequiresValue(arg)) {
        const value = argv[++index];
        if (!value || value.startsWith('--')) {
          throw new Error(`${arg} 缺少参数值。`);
        }
        result.passthrough.push(path.resolve(value));
      }
    } else {
      throw new Error(`未知参数: ${arg}`);
    }
  }

  return result;
}

function requiredValue(argv, index, option) {
  const value = argv[index];
  if (!value || value.startsWith('--')) throw new Error(`${option} 缺少参数值`);
  return value;
}

function usage() {
  console.log([
    '用法:',
    '  node tools/aclnn-batch.mjs detect <operatorDir> [--include aclnnA,aclnnB] [--exclude aclnnC]',
    '  node tools/aclnn-batch.mjs preprocess <operatorDir> [options]',
    '  node tools/aclnn-batch.mjs finalize <operatorDir> [options]',
    '',
    '批量参数:',
    '  --include <names>            只处理指定接口，多个接口用逗号分隔',
    '  --exclude <names>            排除指定接口，多个接口用逗号分隔',
    '  --continue-on-error          单个接口失败时继续处理后续接口（默认）',
    '  --fail-fast                  单个接口失败时立即停止',
    '',
    'preprocess 可透传参数:',
    '  --project-root <dir>',
    '  --prototype-manifest <json>',
    '  --scripts-dir <dir>',
    '  --skill-root <dir>',
    '  --skill-dir <dir>',
    '  --error-codes-file <file>',
    '  --product-mapping-file <file>'
  ].join('\n'));
}

function splitNameList(value) {
  return value
    .split(',')
    .map(item => normalizeInterfaceName(item.trim()))
    .filter(Boolean);
}

function isForwardedOption(arg) {
  return [
    '--project-root',
    '--scripts-dir',
    '--skill-root',
    '--skill-dir',
    '--error-codes-file',
    '--product-mapping-file',
    '--prototype-manifest'
  ].includes(arg);
}

function optionRequiresValue(_arg) {
  return true;
}

async function assertDirectory(dir, label) {
  const stat = await fs.stat(dir).catch(() => null);
  if (!stat?.isDirectory()) {
    throw new Error(`${label}不存在或不是目录: ${dir}`);
  }
}

async function detectTargets(inputDir, args) {
  const scriptsDir = path.resolve(readForwardedValue(args.passthrough, '--scripts-dir') || DEFAULT_SCRIPTS_DIR);
  const detectScript = path.join(scriptsDir, 'detect-interfaces.js');
  await assertFile(detectScript, '接口检测脚本');

  const discoveryArgs = [inputDir];
  for (const option of ['--prototype-manifest', '--project-root']) {
    const value = readForwardedValue(args.passthrough, option);
    if (value) discoveryArgs.push(option, path.resolve(value));
  }
  const result = await runNodeCollect(detectScript, discoveryArgs, scriptsDir);
  if (result.code !== 0) {
    throw new Error(`接口检测失败：${result.stderr || result.stdout || `退出码 ${result.code}`}`);
  }

  const targets = parseDetectReport(result.stdout)
    .map(target => ({
      operatorDir: path.resolve(target.operatorDir),
      interfaces: filterInterfaces(target.interfaces, args)
    }))
    .filter(target => target.interfaces.length > 0);

  if (targets.length === 0) {
    throw new Error('未检测到符合条件的 ACLNN 接口。');
  }
  return targets;
}

async function assertFile(filePath, label) {
  const stat = await fs.stat(filePath).catch(() => null);
  if (!stat?.isFile()) {
    throw new Error(`${label}不存在或不是文件: ${filePath}`);
  }
}

function readForwardedValue(passthrough, optionName) {
  const index = passthrough.indexOf(optionName);
  if (index < 0) {
    return undefined;
  }
  return passthrough[index + 1];
}

function parseDetectReport(report) {
  const targets = [];
  let current = null;

  for (const line of report.split(/\r?\n/)) {
    const heading = /^##\s+(.+?)\s*$/.exec(line);
    if (heading && heading[1] !== '汇总') {
      current = { name: heading[1], operatorDir: '', interfaces: [] };
      targets.push(current);
      continue;
    }

    if (!current) {
      continue;
    }

    const pathMatch = /^-\s+路径:\s+`(.+?)`\s*$/.exec(line);
    if (pathMatch) {
      current.operatorDir = pathMatch[1];
      continue;
    }

    const interfaceMatch = /^\s+-\s+`(aclnn[A-Za-z0-9_]+)`\s*$/.exec(line);
    if (interfaceMatch) {
      current.interfaces.push(normalizeInterfaceName(interfaceMatch[1]));
    }
  }

  return targets.filter(target => target.operatorDir && target.interfaces.length > 0);
}

function normalizeInterfaceName(name) {
  return name.replace(/GetWorkspaceSize$/, '').trim();
}

function filterInterfaces(interfaces, args) {
  const include = new Set(args.include || []);
  const exclude = new Set(args.exclude || []);
  return interfaces
    .map(normalizeInterfaceName)
    .filter(name => !name.includes('GetWorkspaceSize'))
    .filter(name => include.size === 0 || include.has(name))
    .filter(name => !exclude.has(name));
}

function printDetectedTargets(targets) {
  let total = 0;
  console.log('检测到以下 ACLNN 接口:');
  for (const target of targets) {
    console.log('');
    console.log(`算子目录: ${target.operatorDir}`);
    for (const interfaceName of target.interfaces) {
      total++;
      console.log(`  - ${interfaceName}`);
    }
  }
  console.log('');
  console.log(`合计接口数: ${total}`);
}

async function runBatchPreprocess(inputDir, args) {
  await assertFile(PREPROCESS_SCRIPT, '预处理入口脚本');
  const targets = await detectTargets(inputDir, args);
  printDetectedTargets(targets);

  const results = [];
  for (const target of targets) {
    for (const interfaceName of target.interfaces) {
      const runArgs = [
        target.operatorDir,
        '--interface',
        interfaceName,
        ...args.passthrough
      ];
      const result = await runOne('preprocess', PREPROCESS_SCRIPT, runArgs, TOOL_ROOT, target.operatorDir, interfaceName);
      results.push(result);
      if (!result.ok && args.failFast) {
        printSummary(results);
        process.exit(1);
      }
    }
  }

  printSummary(results);
  if (results.some(result => !result.ok)) {
    process.exit(1);
  }
}

async function runBatchFinalize(inputDir, args) {
  await assertFile(FINALIZE_SCRIPT, '最终合成入口脚本');
  const targets = await detectFinalizeTargets(inputDir, args);
  if (targets.length === 0) {
    throw new Error('未找到可合成的接口产物。');
  }

  const results = [];
  for (const target of targets) {
    for (const interfaceName of target.interfaces) {
      const outputDir = path.join(target.operatorDir, 'outputs', interfaceName);
      const missing = await missingFinalizeFiles(outputDir);
      if (missing.length > 0) {
        results.push({
          mode: 'finalize',
          operatorDir: target.operatorDir,
          interfaceName,
          ok: false,
          skipped: true,
          detail: `缺少文件: ${missing.join(', ')}`
        });
        continue;
      }

      const runArgs = [target.operatorDir, '--interface', interfaceName];
      const result = await runOne('finalize', FINALIZE_SCRIPT, runArgs, TOOL_ROOT, target.operatorDir, interfaceName);
      results.push(result);
      if (!result.ok && args.failFast) {
        printSummary(results);
        process.exit(1);
      }
    }
  }

  printSummary(results);
  if (results.some(result => !result.ok)) {
    process.exit(1);
  }
}

async function detectFinalizeTargets(inputDir, args) {
  const operatorDirs = discovery.isOperatorDir(inputDir)
    ? [inputDir] : discovery.findOperatorSubdirs(inputDir);
  const targets = [];
  for (const operatorDir of operatorDirs) {
    const entries = await fs.readdir(path.join(operatorDir, 'outputs'), { withFileTypes: true }).catch(() => []);
    const interfaces = filterInterfaces(entries.filter(entry => entry.isDirectory())
      .map(entry => entry.name).filter(name => /^aclnn[A-Za-z0-9_]+$/.test(name)), args);
    if (interfaces.length) targets.push({ operatorDir, interfaces });
  }
  return targets;
}

async function missingFinalizeFiles(outputDir) {
  const required = [
    '0_prototype.md',
    '5_review_patch.json',
    '3_formula.md',
    '4_generate_skeleton.md',
    '0_call_example.md'
  ];
  const missing = [];
  for (const filename of required) {
    const stat = await fs.stat(path.join(outputDir, filename)).catch(() => null);
    if (!stat?.isFile()) {
      missing.push(filename);
    }
  }
  return missing;
}

async function runOne(mode, scriptPath, scriptArgs, cwd, operatorDir, interfaceName) {
  console.log('');
  console.log(`>>> ${mode}: ${interfaceName}`);
  console.log(`算子目录: ${operatorDir}`);
  const result = await runNodeInherited(scriptPath, scriptArgs, cwd);
  if (result.code === 0) {
    console.log(`[OK] ${mode}: ${interfaceName}`);
    return { mode, operatorDir, interfaceName, ok: true };
  }
  console.log(`[FAILED] ${mode}: ${interfaceName}，退出码 ${result.code}`);
  return { mode, operatorDir, interfaceName, ok: false, detail: `退出码 ${result.code}` };
}

function runNodeInherited(scriptPath, args, cwd) {
  return new Promise(resolve => {
    const child = spawn(process.execPath, [scriptPath, ...args], {
      cwd,
      stdio: 'inherit'
    });
    child.on('close', code => resolve({ code: code ?? 1 }));
    child.on('error', error => {
      console.error(error.message);
      resolve({ code: 1 });
    });
  });
}

function runNodeCollect(scriptPath, args, cwd) {
  return new Promise(resolve => {
    const child = spawn(process.execPath, [scriptPath, ...args], {
      cwd,
      stdio: ['ignore', 'pipe', 'pipe']
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', chunk => { stdout += chunk.toString(); });
    child.stderr.on('data', chunk => { stderr += chunk.toString(); });
    child.on('close', code => resolve({ code: code ?? 1, stdout, stderr }));
    child.on('error', error => resolve({ code: 1, stdout, stderr: error.message }));
  });
}

function printSummary(results) {
  const ok = results.filter(result => result.ok).length;
  const skipped = results.filter(result => result.skipped).length;
  const failed = results.filter(result => !result.ok && !result.skipped).length;

  console.log('');
  console.log('批量执行汇总:');
  console.log(`  成功: ${ok}`);
  console.log(`  跳过: ${skipped}`);
  console.log(`  失败: ${failed}`);

  for (const result of results) {
    const label = result.ok ? 'OK' : result.skipped ? 'SKIP' : 'FAILED';
    const detail = result.detail ? ` - ${result.detail}` : '';
    console.log(`  [${label}] ${result.interfaceName}${detail}`);
  }
}

main().catch(error => {
  console.error(`错误: ${error instanceof Error ? error.message : String(error)}`);
  process.exit(1);
});
