# PR 描述：性能分析技能模块化重构

## 标题
Refactor performance-analysis into modular skills structure

## 主要更改

本PR将原有的`performance-analysis`技能重构为模块化结构，以便更好地组织代码和功能。重构后，技能分为一个主技能和三个子技能，分别用于不同类型的性能瓶颈分析。

### 目录结构变化

**重构前：**
```
performance-analysis/
  ├── Scripts/
  │   └── performance_analysis_main_process.py
  └── SKILL.md
```

**重构后：**
```
profiling-analysis/
  ├── profiling-main/
  │   ├── SKILL.md          # 主分析技能
  │   └── scripts/
  │       └── performance_analysis_main_process.py
  ├── profiling-hostbound/  # 下发瓶颈分析子技能
  │   ├── SKILL.md
  │   └── scripts/
  ├── profiling-computing/  # 计算瓶颈分析子技能
  │   ├── SKILL.md
  │   └── scripts/
  └── profiling-communication/  # 通信瓶颈分析子技能
      ├── SKILL.md
      └── scripts/
```

### 功能改进

1. **模块化设计**：将不同类型的性能瓶颈分析分离为独立的子技能，提高代码可维护性和扩展性
2. **清晰的调用关系**：主技能负责整体分析并根据瓶颈类型调用相应的子技能
3. **符合仓库规范**：所有SKILL.md文件都包含正确的YAML frontmatter
4. **更新的子Skill调用**：主程序中的子Skill调用路径已更新为正确的格式（如 `/profiling-hostbound-skill`）

### 关键文件修改

- **profiling-main/SKILL.md**：添加了"如果遇到xx问题，请调用子Agent运行 /profiling-xx-skill"的内容
- **profiling-main/scripts/performance_analysis_main_process.py**：更新了子Skill调用逻辑，将输出从文件名改为正确的子Skill路径
- **所有新创建的SKILL.md文件**：包含了详细的技能描述和使用说明

## 后续步骤

1. 审核PR并合并到main分支
2. 测试重构后的技能是否正常工作
3. 可以根据需要为子技能添加具体的分析脚本

## 注意事项

- 旧的`performance-analysis`目录已被删除
- 所有现有的功能都已保留并进行了模块化重构
- 技能调用方式保持不变