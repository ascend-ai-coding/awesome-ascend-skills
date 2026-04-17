# CANN Version Guide

当前版本只检查，不安装，不切换，不修复。

## 目录优先级

1. 优先使用使用者提供的 CANN 目录。
2. 如果使用者未提供，再检索常见路径：
   - `CANN < 8.5` 常见路径：`/usr/local/Ascend/ascend-toolkit/latest`
   - `CANN >= 8.5` 常见路径：`/usr/local/Ascend/cann/latest`

## 使用者提供目录时的处理规则

- 使用者可以直接提供到 `latest` 这一层，例如 `x/x/latest`。
- 如果提供的是 `.../latest`，实际加载环境时应执行 `source x/x/set_env.sh`，也就是对 `latest` 的父目录执行 `set_env.sh`。
- 如果提供的是 toolkit/cann 根目录，也应优先用该目录判断版本并加载环境。

## 版本判断规则

对选中的 CANN 根目录按下面顺序判断：

1. 先确定布局属于哪一类：
   - `ascend-toolkit/latest`
   - `cann/latest`
2. 优先检查 `latest/compiler` 是否为软链接。
3. 如果 `latest/compiler` 的链接目标目录名中包含版本号，例如 `8.2.RC1`，则直接认定当前引用版本为该版本。
4. 如果 `latest` 下没有可用于识别版本的软链接，则返回 `latest` 同层级的其他目录名，说明“当前机器可能存在这些版本，需要使用者自行确认”。
5. 如果两类常见路径都不存在，返回 `unknown`，并说明未发现可识别的 CANN 布局。

## 返回要求

返回时至少说明：

- 实际采用的 CANN 根目录
- 是用户提供还是自动检索得到
- `latest` 路径及其布局类型
- `compiler` 软链接判断结果
- 若无法确定版本，则列出 `latest` 同层级候选版本目录并要求使用者确认
