// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
'use strict';

const fs = require('fs');

const DEFAULT_ERROR_CODES = {
    ACLNN_ERR_PARAM_NULLPTR: '161001',
    ACLNN_ERR_PARAM_INVALID: '161002',
    ACLNN_ERR_INNER_INFERSHAPE_ERROR: '561001',
    ACLNN_ERR_INNER_TILING_ERROR: '561002',
};

const DEFAULT_CHIP_MAPPING = [
    { chip: 'Ascend 910D', product: 'Ascend 950PR&950DT系列产品' },
    { chip: 'Ascend 910_95', product: 'Ascend 950PR&950DT系列产品' },
    { chip: 'Ascend 950', product: 'Ascend 950PR&950DT系列产品' },
    { chip: 'Ascend 910C', product: 'Atlas A3系列产品' },
    { chip: 'Ascend 910_93', product: 'Atlas A3系列产品' },
    { chip: 'Ascend 910B*', product: 'Atlas A2系列产品' },
    { chip: 'Ascend 310B*', product: 'Atlas 200I/500 A2推理产品' },
    { chip: 'Ascend 310P AIC', product: 'Atlas推理系列产品' },
    { chip: 'Ascend 310P AIV', product: 'Atlas推理系列产品' },
    { chip: 'Ascend 310P*', product: 'Atlas推理系列产品' },
    { chip: 'Ascend 910', product: 'Atlas训练系列产品' },
    { chip: 'Ascend 910A', product: 'Atlas训练系列产品' },

];

const DEFAULT_NPU_ARCH_MAPPING = [
    { arch: '3510', product: 'Ascend 950PR&950DT系列产品' },
    { arch: '3101', product: 'Ascend 950PR&950DT系列产品' },
    { arch: '2201', product: 'Atlas A3系列产品' },
    { arch: '2201', product: 'Atlas A2系列产品' },
    { arch: '3002', product: 'Atlas 200I/500 A2推理产品' },
    { arch: '2002', product: 'Atlas推理系列产品' },
    { arch: '1001', product: 'Atlas训练系列产品' },

];

const DEFAULT_ALLOWED_PRODUCTS = [
    'Ascend 950PR&950DT系列产品',
    'Atlas A3系列产品',
    'Atlas A2系列产品',
    'Atlas 200I/500 A2推理产品',
    'Atlas推理系列产品',
    'Atlas训练系列产品',
];

function loadYamlFile(filePath) {
    if (!filePath || !fs.existsSync(filePath)) {
        return { data: null, warning: '' };
    }
    try {
        const data = parseSimpleYaml(fs.readFileSync(filePath, 'utf8')) || {};
        return { data, warning: '' };
    } catch (error) {
        return { data: null, warning: `配置文件解析失败，使用默认配置: ${filePath}; ${error.message}` };
    }
}

function parseSimpleYaml(content) {
    const root = {};
    let currentKey = '';
    let currentItem = null;

    for (const rawLine of content.replace(/\r\n/g, '\n').split('\n')) {
        const line = stripYamlComment(rawLine);
        if (!line.trim()) continue;

        const topLevelMatch = line.match(/^([A-Za-z0-9_]+):\s*(.*)$/);
        if (topLevelMatch) {
            currentKey = topLevelMatch[1];
            currentItem = null;
            const value = topLevelMatch[2].trim();
            if (value === '[]') {
                root[currentKey] = [];
            } else if (value === '{}') {
                root[currentKey] = {};
            } else if (value) {
                root[currentKey] = parseYamlScalar(value);
            } else {
                root[currentKey] = currentKey === 'codes' ? {} : [];
            }
            continue;
        }

        if (!currentKey) continue;

        if (currentKey === 'codes') {
            const pair = line.match(/^\s+([A-Za-z0-9_]+):\s*(.+)$/);
            if (pair) {
                root.codes[pair[1]] = parseYamlScalar(pair[2]);
            }
            continue;
        }

        if (currentKey === 'chip_mapping' || currentKey === 'npu_arch_mapping') {
            const itemStart = line.match(/^\s*-\s+([A-Za-z0-9_]+):\s*(.+)$/);
            if (itemStart) {
                currentItem = {};
                currentItem[itemStart[1]] = parseYamlScalar(itemStart[2]);
                root[currentKey].push(currentItem);
                continue;
            }
            const itemField = line.match(/^\s+([A-Za-z0-9_]+):\s*(.+)$/);
            if (itemField && currentItem) {
                currentItem[itemField[1]] = parseYamlScalar(itemField[2]);
            }
            continue;
        }

        // 仅读取 codes、chip_mapping 和 npu_arch_mapping；其余键不参与配置合并。
    }

    return root;
}

function stripYamlComment(line) {
    let quote = '';
    for (let i = 0; i < line.length; i += 1) {
        const char = line[i];
        if ((char === '"' || char === "'") && line[i - 1] !== '\\') {
            quote = quote === char ? '' : quote || char;
            continue;
        }
        if (char === '#' && !quote) {
            return line.slice(0, i);
        }
    }
    return line;
}

function parseYamlScalar(value) {
    const trimmed = String(value || '').trim();
    if ((trimmed.startsWith('"') && trimmed.endsWith('"')) || (trimmed.startsWith("'") && trimmed.endsWith("'"))) {
        return trimmed.slice(1, -1);
    }
    return trimmed;
}

function loadErrorCodeConfig(filePath) {
    const { data, warning } = loadYamlFile(filePath);
    const codes = { ...DEFAULT_ERROR_CODES };
    const warnings = warning ? [warning] : [];
    if (data && data.codes && typeof data.codes === 'object') {
        for (const [symbol, value] of Object.entries(data.codes)) {
            if (Object.prototype.hasOwnProperty.call(DEFAULT_ERROR_CODES, symbol)) {
                codes[symbol] = String(value);
            } else if (/^ACLNN_ERR_[A-Z0-9_]+$/.test(symbol)) {
                warnings.push(`忽略非文档允许错误码: ${symbol}`);
            }
        }
    }
    return { codes, warnings };
}

function loadProductMappingConfig(filePath) {
    const { data, warning } = loadYamlFile(filePath);
    const warnings = warning ? [warning] : [];
    const chipMapping = mergeByKey(DEFAULT_CHIP_MAPPING, normalizeArray(data?.chip_mapping), 'chip');
    const npuArchMapping = mergeNpuArch(DEFAULT_NPU_ARCH_MAPPING, normalizeArray(data?.npu_arch_mapping));
    const finalAllowedProducts = DEFAULT_ALLOWED_PRODUCTS;
    return { chipMapping, npuArchMapping, finalAllowedProducts, warnings };
}

function normalizeArray(value) {
    return Array.isArray(value) ? value.filter(item => item && typeof item === 'object') : [];
}

function mergeByKey(defaultItems, userItems, key) {
    const map = new Map(defaultItems.map(item => [normalizeKey(item[key]), { ...item }]));
    for (const item of userItems) {
        if (!item[key] || !item.product) continue;
        map.set(normalizeKey(item[key]), { ...item, [key]: String(item[key]), product: String(item.product) });
    }
    return Array.from(map.values());
}

function mergeNpuArch(defaultItems, userItems) {
    const map = new Map();
    for (const item of defaultItems) {
        const list = map.get(String(item.arch)) || [];
        list.push({ ...item });
        map.set(String(item.arch), list);
    }
    const userMap = new Map();
    for (const item of userItems) {
        if (!item.arch || !item.product) continue;
        const arch = String(item.arch);
        const list = userMap.get(arch) || [];
        list.push({ arch, product: String(item.product) });
        userMap.set(arch, list);
    }
    for (const [arch, list] of userMap.entries()) {
        map.set(arch, list);
    }
    return Array.from(map.values()).flat();
}

function mapChipIdentifier(identifier, productConfig) {
    const products = new Set();
    const normalized = normalizeChipIdentifier(identifier);
    if (!normalized) return [];
    if (/isregbase/i.test(identifier)) {
        return ['Ascend 950PR&950DT系列产品'];
    }
    const npuArch = extractArchValue(identifier);
    if (npuArch) {
        for (const item of productConfig.npuArchMapping) {
            if (String(item.arch) === npuArch) products.add(item.product);
        }
    }
    const exact = [];
    const prefix = [];
    for (const item of productConfig.chipMapping) {
        const key = normalizeChipIdentifier(item.chip);
        if (!key) continue;
        if (String(item.chip).endsWith('*')) {
            const prefixKey = key.replace(/\*$/, '');
            if (normalized.startsWith(prefixKey)) prefix.push({ len: prefixKey.length, product: item.product });
        } else if (normalized === key) {
            exact.push(item.product);
        }
    }
    for (const product of exact) products.add(product);
    prefix.sort((a, b) => b.len - a.len);
    for (const item of prefix) products.add(item.product);
    return Array.from(products);
}

function extractArchValue(identifier) {
    const npu = identifier.match(/(?:DAV_|__NPU_ARCH__[=!><]+)(\d+)/i);
    if (npu) return npu[1];
    const davMacro = identifier.match(/__DAV_C?(\d{3})/i);
    if (davMacro) return `${davMacro[1]}1`;
    return null;
}

function normalizeChipIdentifier(value) {
    return String(value || '')
        .replace(/^SocVersion::/i, '')
        .replace(/^NpuArch::/i, '')
        .replace(/^ASCEND/i, 'Ascend')
        .replace(/\*/g, '*')
        .replace(/[^a-zA-Z0-9*]/g, '')
        .toLowerCase();
}

function normalizeKey(value) {
    return String(value || '').trim().toLowerCase();
}

module.exports = {
    DEFAULT_ERROR_CODES,
    DEFAULT_ALLOWED_PRODUCTS,
    loadErrorCodeConfig,
    loadProductMappingConfig,
    mapChipIdentifier,
};
