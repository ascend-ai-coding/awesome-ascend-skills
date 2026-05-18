# 算子工程目录结构（交付物落盘）

本文只描述 **目录树、文件职责、骨架生成后的自检路径**；不写具体的实现方式，详细可参考其他references。

---

## 目录树

```
ops/<算子名>/                    # 算子根目录（USER_OP_PROJECT）
├── op_host/
│   ├── *_tiling.h               # Tiling 数据结构
│   ├── *_tiling.cpp             # Tiling 实现（TilingFunc）
│   ├── *_def.cpp                # OpDef 注册
│   ├── *_infershape.cpp         # 形状推导 + 数据类型推导
│   └── CMakeLists.txt           # 由 opgen 生成
├── op_kernel/
│   ├── *.cpp                    # Kernel 实现
│   ├── *.h / *_impl.h           # Kernel 模板定义（可含 BlockEpilogue 组装）
│   ├── custom_epilogue/         # 可选：自定义 Tile Epilogue 头文件（见 custom-epilogue.md）
│   └── CMakeLists.txt           # catlass include + 可选 custom_epilogue -I
├── examples/
│   └── test_aclnn_<op>.cpp      # 可调用测试示例（路径与命名固定）
└── CMakeLists.txt               # 由 opgen 生成
```

---

## 文件职责与细则索引

| 路径模式 | 职责（一句话） | 编写细则 |
|----------|----------------|----------|
| `op_host/*_tiling.h` | Tiling 数据结构 | [tiling-rules.md](./tiling-rules.md) |
| `op_host/*_tiling.cpp` | TilingFunc、TilingKey、Workspace | [tiling-rules.md](./tiling-rules.md) |
| `op_host/*_def.cpp` | OpDef 仅原型与芯片配置 | [def-rules.md](./def-rules.md) |
| `op_host/*_infershape.cpp` | InferShape / InferDataType | [infershape-rules.md](./infershape-rules.md) |
| `op_kernel/*.{h,cpp}` | Catlass Kernel、`TILING_KEY_IS` | [kernel-rules.md](./kernel-rules.md) |
| `op_kernel/custom_epilogue/*.hpp` | 自定义 Tile Epilogue（无现成 catlass 组件时） | [custom-epilogue.md](./custom-epilogue.md) |
| `examples/test_aclnn_<op>.cpp` | aclnn 两段式调用测试程序 | [example-rules.md](./example-rules.md) |

---

## 生成骨架与目录自检

在 `OPS_PROJECT_ROOT` 下：

```bash
bash build.sh --genop=ops/<op_name>
```

`<op_name>` 须含 `catlass` 子串。

```bash
ls ops/<op_name>/op_host/    # 应有 *_def.cpp, *_tiling.h, *_tiling.cpp, *_infershape.cpp, CMakeLists.txt
ls ops/<op_name>/op_kernel/  # 应有 *.cpp, *.h
ls ops/<op_name>/examples/   # 应有 test_aclnn_*.cpp
```

`op_kernel/CMakeLists.txt` 若骨架未带齐 catlass 选项，须按 [kernel-rules.md](./kernel-rules.md) 在 `scripts/kernel/binary_config/ascendc_config.json` 中配置编译选项。

---

## 结构级检查（仅「有没有、放哪」）

- [ ] **OPS_PROJECT_ROOT** 下存在 `catlass/include`（与 `ops/` 同级，不在 `ops/<op>/` 内）；克隆细则见 [kernel-rules.md](./op_kernel/kernel-rules.md)
- [ ] 算子目录由 `build.sh --genop=ops/<op_name>` 生成，非手搓空目录
- [ ] `scripts/kernel/binary_config/ascendc_config.json` 中已配置算子编译选项（见 [compile-options.md](./compile-options.md)）
- [ ] `op_host` 下 tiling / infershape / def 三逻辑分文件存在，且与 `CMakeLists.txt` 一致
- [ ] `op_kernel` 下源码与 `CMakeLists.txt` 存在
- [ ] `examples/test_aclnn_<op_name>.cpp` 路径与命名符合工程约定

实现质量、设计文档对齐、陷阱与端到端步骤见根目录 **SKILL.md**；测试程序写法见 **example-rules.md**。
