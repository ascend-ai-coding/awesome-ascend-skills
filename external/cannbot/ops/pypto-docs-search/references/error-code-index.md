# PyPTO 错误码 → 排障文档索引

报错带错误码（`Errcode: Fxxxxx!` / `ErrCode: Fxxxxx!`）时，按前缀定位组件排障文档。
按错误码前缀在下表查到组件，`<doc>` 取该行"排障文档"列的值。**缓存在场**读 `$PYPTO_DEVKIT_DIR/docs/trouble_shooting/<doc>.md`；**无则在线** `https://pypto.gitcode.com/_sources/trouble_shooting/<doc>.md.txt`。
例：错误码 `F70123` 前缀 `F7` → MACHINE → `<doc>`=`machine` → 本地 `$PYPTO_DEVKIT_DIR/docs/trouble_shooting/machine.md` / 在线 `.../_sources/trouble_shooting/machine.md.txt`。
范围映射权威源 `trouble_shooting/README`，结构变动以其为准。

| 错误码前缀 | 组件 | 排障文档 |
|---|---|---|
| `F0XXXX` | 外部写法问题（检查算子写法，无专文） | — |
| `F1XXXX` | 框架内部公共 | — |
| `F2`–`F3XXXX` | FUNCTION | `function` |
| `F4`–`F5XXXX` | PASS | `pass` |
| `F6XXXX` | CODEGEN | `codegen` |
| `F7`–`F8XXXX` | MACHINE | `machine` |
| `F9XXXX` | SIMULATION | `simulation` |
| `FAXXXX` | DISTRIBUTED | `distributed` |
| `FBXXXX` | VERIFY | `verify` |
| `FCXXXX` | OPERATION | `operation` |
| `FC0`–`FC2XXX` | OPERATION · VECTOR 子类 | `vector` |
| `FC3`–`FC5XXX` | OPERATION · MATMUL 子类 | `matmul` |
| `FC6`–`FC8XXX` | OPERATION · CONV 子类 | `conv` |
| `FC9XXX` | OPERATION · 视图类 OP 子类 | `view_op` |

无错误码（`FFFFF` / `UNKNOWN` / 无码报错）：从报错信息与日志入手，排查教程见 [`doc-index.md`](doc-index.md) 的 `tutorials/debug`。
