# PyPTO 错误码排查流程

⚠️ **前提：先确认报错是否包含错误码（格式 `Errcode: Fxxxxx!` 或 `ErrCode: Fxxxxx!`）**
- **有错误码** → 继续本流程
- **无错误码** → 跳过，直接分析报错信息或日志定位问题

---

## 1. 错误信息获取途径

| 途径 | 适用场景 | 特点 |
|------|----------|------|
| stderr（运行异常） | 前端参数校验错误 | 直接抛 RuntimeError，包含 `Errcode: Fxxxxx!` |
| 日志文件 | 编译/运行阶段深层错误 | `pypto-log*.log` 中 `[ERROR] ... ErrCode: Fxxxxx!` |

**关键区别**：
- 前端参数校验错误（如 FC0000 dtype 不匹配）不写入日志文件，直接从异常信息获取
- 编译/运行阶段错误写入日志文件，需开启日志排查

---

## 2. 排查流程

1. **识别错误码**：从 stderr 或日志中搜索 `Errcode: F` / `ErrCode: F`
2. **定位组件文档**：根据错误码前缀查 `docs/trouble_shooting/README.md` → 对应子文档
3. **查阅排查建议**：打开子文档，找到错误码定义与排查步骤
4. **执行排查与修复**：按文档建议逐项检查并修复

---

## 3. 开启日志与排查手段

详见各组件文档：
- `docs/trouble_shooting/codegen.md` - 编译阶段日志开启方法（搜索 "设置日志级别"）
- `docs/trouble_shooting/vector.md` - 排查步骤与典型场景（搜索 "排查手段"）

---

## 4. 常见错误码索引

**详见对应文档，以下仅为快速定位索引：**

| 错误码 | 组件 | 文档位置 |
|--------|------|----------|
| FC0000 | VECTOR | `docs/trouble_shooting/vector.md` - 搜索 "FC0000 ERR_PARAM_INVALID" |
| FC0001 | VECTOR | `docs/trouble_shooting/vector.md` - 搜索 "FC0001 ERR_PARAM_DTYPE_UNSUPPORTED" |
| FC1000 | VECTOR | `docs/trouble_shooting/vector.md` - 搜索 "FC1000 ERR_CONFIG_TILE" |
| FC1001 | VECTOR | `docs/trouble_shooting/vector.md` - 搜索 "FC1001 ERR_CONFIG_ALIGNMENT" |
| F62014 | CODEGEN | `docs/trouble_shooting/codegen.md` - 搜索 "F62014 SYMBOL_NOT_FOUND" |
| F63001 | CODEGEN | `docs/trouble_shooting/codegen.md` - 搜索 "F63001 COMPILE_CODE_FAILED" |

---

## 5. 排查注意事项

1. **错误码前缀决定组件归属**：F7XXXX 必查 `machine.md`，FC0XXX 必查 `vector.md`
2. **前端校验错误无需看日志**：直接从异常信息获取错误码和描述
3. **编译问题需串行编译**：见 `docs/trouble_shooting/codegen.md`（搜索 "并行编译"）

---

## 6. 关联 Skill

某些错误码在文档中标注了关联 Skill，可直接加载对应技能进行排查：

| 错误类型 | 关联 Skill |
|----------|------------|
| 环境问题 | `pypto-environment-setup` |
| 精度问题 | `pypto-precision-debug` / `pypto-precision-compare` |
| AICore Error | `pypto-aicore-error-locator` |
| Pass 问题 | `pypto-pass-error-fixer` |
