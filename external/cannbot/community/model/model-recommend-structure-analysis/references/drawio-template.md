# drawio 结构图集规范（推荐模型版）

[structure-analysis](../SKILL.md) 的 drawio 产物规范：**单文件多页**——P1 主题结构图（模型级拓扑）+ P2..Pn 各模块详细展开图。

## 1. 文件组织

```
docs/{model_name}/model-analysis/diagrams/
└── {model_name}.drawio      # 单文件，多 <diagram> 页
```

页按模型实际结构裁剪（无特征交互模块的纯 DNN 模型无 P3 页；单任务模型无多任务输出页）：

| 页名 | 内容 | 裁剪条件 |
|------|------|---------|
| `P1-主题结构图` | 模型级模块拓扑 | 必备 |
| `P2-特征输入与Embedding` | 稀疏/稠密特征输入、查表、flatten/concat 展开 | 存在查表输入时 |
| `P3-特征交互` | cross/interaction/attention 类交互模块展开 | 存在交互模块时 |
| `P4-DNN-MLP` | 全连接塔/残差块展开 | 必备 |
| `P5-输出层` | 多任务 head（多目标输出、sigmoid/softmax）展开 | 存在多任务输出时 |
| `P6+-自定义` | 非标准大参数模块（超大 embedding 表、自定义算子子图） | 存在时必备 |

## 2. 风格与图例

配色采用「柔和粉彩填充 + 高饱和描边 + 深色文字」（Material 风格），全局统一，图例以同色样块自演示放每页左下角：

| 类别 | fillColor | strokeColor | 示例节点 |
|------|-----------|-------------|---------|
| AIC/Cube 类 | `#FFF3E0` | `#E65100` | MatMul/Gemm/Conv、MLP 全连接层、交互点积 |
| AIV/Vector 类 | `#E8F5E9` | `#2E7D32` | Add/Mul/Relu/Sigmoid、Reduce、LayerNorm、Cast |
| 访存类 | `#FFFDE7` | `#F57F17` | Gather/GatherNd、Embedding 查表、SparseToDense、Concat |
| 控制类（小算子串） | `#FCE4EC` | `#C2185B` | Shape/Reshape/Transpose/Slice/Where/If/Loop |
| 输入/输出 | `#E3F2FD` | `#1565C0` | 特征输入（ids/values）、多任务打分输出 |
| 容器（模块/块） | `#ECEFF1` | `#607D8B`（实线，顶部加粗标题） | Feature Interaction Block、MLP Tower ×N、多任务 Head |
| 信息框 | `#FAFAFA`（shape=note） | `#90A4AE` | 页首参数信息、说明 |

节点基础样式：`rounded=1;arcSize=12;whiteSpace=wrap;html=1;shadow=1;fontSize=13;fontColor=#263236;`

**布局规则（防连线穿越）**：

- 节点水平间距 ≥ 50px、垂直间距 ≥ 60px，**禁止紧贴**（间距 <30px 视为布局缺陷）
- 主数据流**左 → 右**；并行分支（多特征路、多任务 head）**垂直堆叠**后再汇合
- 连线必须带显式锚点：水平链 `exitX=1;exitY=0.5;entryX=0;entryY=0.5`，垂直链 `exitX=0.5;exitY=1;entryX=0.5;entryY=0`，分支汇合用分数锚点（如 `entryY=0.3/0.7`）分层入边
- 旁路长连线（如 residual、直连稠密特征）用**航点**（`<Array as="points"><mxPoint x y/></Array>`）沿空白泳道绕行，禁止穿越其他框体
- 每页顶部加标题（`text;strokeColor=none;fillColor=none;fontSize=22;fontStyle=1;`）

**连线样式**：

| 类型 | 样式 |
|------|------|
| 数据流 | `strokeColor=#455A64;strokeWidth=1.5;`（深蓝灰） |
| 旁路 / residual / 稠密特征直连 | `dashed=1;strokeColor=#90A4AE;` |
| 融合算子候选（条目 10 输出回标） | `strokeColor=#E65100;strokeWidth=2.5;`（粗橙） |
| 边标签 | `fontSize=11;labelBackgroundColor=#FFFFFF;` |

其他约定：

- 多特征路（field_a/field_b/...）：并行分支显式画出，边上标注特征名或输入名，**不合并省略**
- 多层重复：容器 + `×N` 标注，不逐层画
- 节点文本换行：样式含 `whiteSpace=wrap;html=1;`
- 并行支路（多路特征汇聚、多任务 head）：上下分行布局，汇聚节点收边

**换行的生成方式（重要坑位）**：生成代码（Python 等）中在 value 里写**真实换行符**，由 XML 序列化器转义为 `&#10;`（Python `xml.etree.ElementTree.tostring` 对属性内 `\n` 会输出 `&#10;`）。**两个禁止**：
1. 禁止在生成代码里写字面 `&#10;` 字符串——会被序列化器二次转义为 `&amp;#10;`，drawio 将其渲染为乱文本（所有标签挤成超长单行）
2. 禁止用 `minidom.toprettyxml()` 等二次格式化——会把属性内的 `&#10;` 还原为字面换行，再被 XML 属性规范化为空格，换行丢失

## 3. XML 骨架

```xml
<mxfile host="app.diagrams.net">
  <diagram id="p1" name="P1-主题结构图">
    <mxGraphModel dx="800" dy="600" grid="1" page="1" pageWidth="1600" pageHeight="900" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        <mxCell id="p1_info" value="模型: {name}&#10;op 总数: {N}  AIC/AIV: {aic}/{aiv}&#10;输入数: {I}  输出数: {O}&#10;权重总量: {weights}  dtype: {dtype}" style="shape=note;whiteSpace=wrap;html=1;size=16;fillColor=#f5f5f5;strokeColor=#666666;align=left;" vertex="1" parent="1">
          <mxGeometry x="40" y="40" width="260" height="120" as="geometry"/>
        </mxCell>
        <mxCell id="p1_n_in" value="特征输入&#10;ids [B,F] / values [B,F]" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="1">
          <mxGeometry x="40" y="420" width="150" height="60" as="geometry"/>
        </mxCell>
        <mxCell id="p1_n_embed" value="Embedding 查表&#10;Gather ×E（见 P2）" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;" vertex="1" parent="1">
          <mxGeometry x="250" y="420" width="150" height="60" as="geometry"/>
        </mxCell>
        <mxCell id="p1_n_block" value="推荐网络主体&#10;━━━━━━━━&#10;特征交互 (见 P3)&#10;MLP Tower ×N (见 P4)&#10;多任务 Head (见 P5)" style="rounded=1;whiteSpace=wrap;html=1;dashed=1;fillColor=none;strokeColor=#666666;" vertex="1" parent="1">
          <mxGeometry x="460" y="320" width="320" height="260" as="geometry"/>
        </mxCell>
        <mxCell id="p1_n_out" value="输出&#10;scores ×T" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="1">
          <mxGeometry x="860" y="420" width="140" height="60" as="geometry"/>
        </mxCell>
        <mxCell id="p1_e1" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;" edge="1" parent="1" source="p1_n_in" target="p1_n_embed">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="p1_e2" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;" edge="1" parent="1" source="p1_n_embed" target="p1_n_block">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="p1_e3" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;" edge="1" parent="1" source="p1_n_block" target="p1_n_out">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
  <!-- P2..Pn 各模块展开页，<diagram> 结构同上；页内节点/边 id 加页前缀 -->
</mxfile>
```

XML 规则：

- 每页必须含 `<mxCell id="0"/>` 与 `<mxCell id="1" parent="0"/>` 两个根节点，业务节点 `parent="1"`
- 节点：`vertex="1"` + `<mxGeometry x y width height as="geometry"/>`；边：`edge="1"` + `source`/`target` 引用节点 id + `<mxGeometry relative="1" as="geometry"/>`
- **业务节点 id 页内唯一**（`0`/`1` 根节点每页必备、跨页重复属正常）；建议按页前缀命名（`p1_n_*` / `p1_e_*` / `p2_n_*` …），边的 `source`/`target` 只能引用同页节点
- **`as="geometry"` 必须存在**（drawio 靠它把坐标挂到节点；缺失时所有图形渲染在原点堆叠成一团文字）

**生成代码坑位（Python ElementTree）**：
1. `as` 是 Python 关键字，**禁止**用 `ET.SubElement(..., as_="geometry")`——ET 不会去掉关键字参数的尾部下划线，会写出字面量 `as_="geometry"`（drawio 不识别）。必须用字典传参：`ET.SubElement(c, "mxGeometry", {"x": ..., "as": "geometry"})`
2. value 内换行用**真实换行符** `\n`，ET 序列化属性时自动转义为 `&#10;`；禁止写字面 `&#10;` 字符串（会被二次转义为 `&amp;#10;`，drawio 渲染为乱文本）
3. 禁止 `minidom.toprettyxml()` 二次格式化——会把属性内 `&#10;` 还原为字面换行，被 XML 属性规范化为空格

## 4. 页清单与必备内容

### P1 主题结构图

采用 **top-down 或 left-right 布局**（按图规模选择），主体模块可先用容器概括（内部子模块以副标签指向展开页）；节点数少于 30 的模型建议直接全展开：

```
特征输入（稀疏 ids / 稠密 values）→ Embedding 查表（见 P2）
  → 特征交互（见 P3）→ MLP Tower ×N（见 P4）→ 多任务 Head（见 P5）→ 输出 scores
```

- 容器标签含 `×N` 重复说明；稠密特征直连路径用虚线旁路画出
- 页首信息框必备字段：模型名、op 总数、AIC/AIV 算子数、输入/输出数、权重总量、主要 dtype

### P2 特征输入与 Embedding 展开

多路特征（field_a/field_b/...）并行 → 各自 Gather/GatherNd 查表（标注表名、表形状、dtype）→ flatten/reshape → Concat/Elementwise 合并；超大 embedding 表（GB 级）节点加体积副标签；bool/int 类型转换（Cast）显式画出。

### P3 特征交互展开

按实际模块展开：Cross（点积/外积交互，标注 AIC 归属）、Interaction（bit-wise/field-wise）、Attention 类交互（标注 Q/K/V 形状与 softmax 归属 AIV）；中间大张量（B × F × D 级）标注体积。

### P4 DNN-MLP 展开

`Linear → 激活（Relu/Sigmoid）` ×N（并行上下支路）→ 残差相加（dashed 旁路边，若有）；标注各 Linear 的 [in, out] 形状与 AIC 归属；AIV 类 Norm/Dropout 显式画出。

### P5 输出层（多任务）展开

共享底层 → 各任务 head 并行（上下分行）→ Sigmoid/Softmax → 各任务输出（输出名与 `io_config.outputs[]` 一致）。

## 5. 生成后自校验（必做）

```bash
python3 - <<'EOF'
import sys, xml.etree.ElementTree as ET
path = sys.argv[1] if len(sys.argv) > 1 else "docs/{model_name}/model-analysis/diagrams/{model_name}.drawio"
raw = open(path).read()
assert "as_=" not in raw, "found as_= attribute (geometry will not mount!)"
assert "&amp;#10;" not in raw, "found double-escaped &#10;"
t = ET.parse(path)
problems = []
pages = list(t.iter("diagram"))
assert pages, "no <diagram> pages"
for pg in pages:
    cells = list(pg.iter("mxCell"))
    all_ids = [c.get("id") for c in cells if c.get("id")]
    dup = {i for i in all_ids if all_ids.count(i) > 1}
    ids = set(all_ids)
    dangling = [f"{c.get('id')} -> {r}" for c in cells
                for r in (c.get("source"), c.get("target")) if r and r not in ids]
    if dup: problems.append((pg.get("name"), "dup", dup))
    if dangling: problems.append((pg.get("name"), "dangling", dangling))
    if "0" not in ids or "1" not in ids:
        problems.append((pg.get("name"), "missing root cells 0/1", None))
    verts = [c for c in cells if c.get("vertex") == "1"]
    xs, ys = [], []
    for v in verts:
        geo = [g for g in v if g.tag == "mxGeometry" and g.get("as") == "geometry"]
        if not geo:
            problems.append((pg.get("name"), "vertex missing geometry", v.get("id")))
            continue
        xs.append(float(geo[0].get("x", 0))); ys.append(float(geo[0].get("y", 0)))
    if verts and (max(xs) - min(xs) < 100 and max(ys) - min(ys) < 100):
        problems.append((pg.get("name"), "all nodes clustered (geometry not spread)", None))
assert not problems, problems
print("OK:", len(pages), "pages")
EOF
```

再人工核对：

- [ ] 页齐全（对照结构档案模块类型）
- [ ] 关键字段（op 总数、AIC/AIV、输入输出数、权重总量）与 `model_config.json` 一致
- [ ] 图例块存在于每页
- [ ] 多特征路/多任务分支已显式标注
- [ ] drawio（app.diagrams.net / drawio desktop）可正常打开
