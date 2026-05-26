---
name: external-gitcode-ascend-run-mindspeed-llm-test
description: 运行MindSpeed-LLM项目的测试用例。当需要运行测试用例、扫描项目代码覆盖率时调用此技能
original-name: run-mindspeed-llm-test
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# MindSpeed-LLM 测试执行器

执行MindSpeed-LLM项目的单元测试用例。

## 运行环境
运行mindspeed-llm测试用例时请严格按照以下运行环境的要求进行
- **Docker容器**：`q4_master`
- **默认环境配置**：'source /root/.bashrc'
- **默认工作目录**：当前项目工作目录，记为`WORK_DIR`
- **NPU环境配置**：'source /usr/local/Ascend/ascend-toolkit/set\_env.sh && source /usr/local/Ascend/nnal/atb/set\_env.sh'

## 输入

用户输入执行脚本（如run\_coverage.sh脚本）或者指定测试用例目录

## 执行流程

1.进入docker容器

```
docker exec -it q4_master /bin/bash
```

2.执行默认环境配置

```
source /root/.bashrc
```

3.进入项目工作目录

```
cd ${WORK_DIR}
```

4.运行任务
执行测试命令或脚本

```
# 执行全量代码覆盖率扫描任务
bash tests/run_coverage.sh all

# 执行pipeline所有用例
bash tests/pipeline/pipe_run.sh

# 执行ut测试用例
pytest -s tests/ut/inference/test_inference.py
```

5.分析运行结果

## 输出

输出测试报告或覆盖率报告，以markdown文件的形式至`TEST`目录下

## 示例

### 执行全量代码覆盖率扫描任务

```bash
docker exec -it q4_master bash -c "source /root/.bashrc && cd ${WORK_DIR} && bash tests/run_coverage.sh all"
```

### 运行pipeline所有用例

```bash
docker exec -it q4_master bash -c "source /root/.bashrc && cd ${WORK_DIR} && bash tests/pipeline/pipe_run.sh"
```

### 运行UT测试用例

```bash
# 运行一个ut测试用例文件中的所有用例
docker exec -it q4_master bash -c "source /root/.bashrc && cd ${WORK_DIR} && pytest -s tests/pipeline/ut/model_module/test_attention.py"

# 运行一个ut测试用例中的特定测试用例
docker exec -it q4_master bash -c "source /root/.bashrc && cd ${WORK_DIR} && pytest -s tests/pipeline/ut/model_module/test_attention.py::TestAttention::test_alibi_seq8192_bs2_bf16"
```

### 运行ST测试用例

```bash
docker exec -it q4_master bash -c "source /root/.bashrc && cd ${WORK_DIR} && bash tests/pipeline/st/llama2/llama2_tp8_pp1_coc_ptd.sh"
```