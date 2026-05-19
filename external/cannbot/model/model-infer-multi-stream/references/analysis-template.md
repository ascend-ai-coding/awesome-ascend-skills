# <network_or_case_name> 多流分析

## 1. 分析范围

- 模型/网络：
- 阶段：`prefill` / `decode`
- 分析对象：
- 说明：

## 2. 整网模块清单

| module_id | module_name | module_type | inputs | outputs | side_effect | resource_hint |
| --- | --- | --- | --- | --- | --- | --- |

## 3. 整网模块依赖清单

| from | to | dependency_type | reason |
| --- | --- | --- | --- |

## 4. 整网模块 DAG

```mermaid
flowchart LR
```

## 5. 模块级并行性结论

### 5.1 主串行链

- 

### 5.2 可并行模块组

- 组 A：
- 组 B：

### 5.3 待验证模块组

- 

### 5.4 建议流分组

- `Stream0` / `Main Path`：
- `Stream1` / `Side Path`：
- `Stream2` / `Comm Path`：

## 6. 模块拆解结论

- 哪些模块必须串行：
- 哪些模块可以并行：
- 哪些模块并行性待验证：
- 推荐优先检查的多流切入点：

## 7. 模块内算子拆解

### 7.1 <module_name>

#### 算子清单

| op_id | op_name | op_type | inputs | outputs | side_effect | resource_hint |
| --- | --- | --- | --- | --- | --- | --- |

#### 算子依赖清单

| from | to | dependency_type | reason |
| --- | --- | --- | --- |

#### 算子 DAG

```mermaid
flowchart LR
```

#### 算子并行性结论

##### 模块内主串行链

- 

##### 模块内可并行算子组

- 组 A：
- 组 B：

##### 模块内待验证算子组

- 

##### 模块内建议流分组

- `Stream0` / `Main Path`：
- `Stream1` / `Side Path`：
- `Stream2` / `Comm Path`：

#### 模块内结论

- 模块内必须串行的部分：
- 模块内可以并行的部分：
- 模块内待验证的部分：
- 可能的流切换点 / 同步点：

### 7.2 <module_name>

#### 算子清单

| op_id | op_name | op_type | inputs | outputs | side_effect | resource_hint |
| --- | --- | --- | --- | --- | --- | --- |

#### 算子依赖清单

| from | to | dependency_type | reason |
| --- | --- | --- | --- |

#### 算子 DAG

```mermaid
flowchart LR
```

#### 算子并行性结论

##### 模块内主串行链

- 

##### 模块内可并行算子组

- 组 A：
- 组 B：

##### 模块内待验证算子组

- 

##### 模块内建议流分组

- `Stream0` / `Main Path`：
- `Stream1` / `Side Path`：
- `Stream2` / `Comm Path`：

#### 模块内结论

- 模块内必须串行的部分：
- 模块内可以并行的部分：
- 模块内待验证的部分：
- 可能的流切换点 / 同步点：

## 8. 最终结论

- 整网最适合做多流优化的位置：
- 模块级并行的主结论：
- 算子级并行的主结论：
- 后续建议：
