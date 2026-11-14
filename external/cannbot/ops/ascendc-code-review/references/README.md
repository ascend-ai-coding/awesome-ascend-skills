# 规则集总览

> **声明**：本表**仅供人读**，用于快速了解全部规则文件的分类、定位与条款规模。机器路由（规则加载、侧别过滤、触发匹配）依赖各规则文件自身的 `<适用>` 声明头，**不依赖本表**。任何检视流程不得以「确认文件清单」为由读取本表；主题文件清单以各文件 `<适用>` 声明为准，本概览仅导航不参与判定。新增、修改或删除规则文件时，须按 `docs/rule-writing-standard.md` SYNC-1 同步更新本表。

## 规则集概览

<table>
  <thead>
    <tr><th>#</th><th>规则分类</th><th>规则文件</th><th>条款数(总计:194)</th><th>规则描述</th><th>规则缩写</th><th>适用语言</th><th>适用侧别</th></tr>
  </thead>
  <tbody>
    <tr><td>1</td><td rowspan="2">AscendC红线和topN</td><td>ascendc-red-line.md</td><td>10</td><td>生产环境总结的红线问题清单</td><td>RED</td><td>C++</td><td>All</td></tr>
    <tr><td>2</td><td>ascendc-topk.md</td><td>12</td><td>检视实践中沉淀的topN风险点清单</td><td>TOPK</td><td>C++</td><td>All</td></tr>
    <tr><td>3</td><td rowspan="5">AscendC通用规则</td><td>ascendc-kernel-rules.md</td><td>13</td><td>AscendC Kernel侧代码规则</td><td>KERNEL</td><td>C++</td><td>Kernel</td></tr>
    <tr><td>4</td><td>ascendc-host-rules.md</td><td>4</td><td>AscendC Host侧代码规则</td><td>HOST</td><td>C++</td><td>Host</td></tr>
    <tr><td>5</td><td>ascendc-log.md</td><td>2</td><td>Host 侧日志上报规范</td><td>LOG</td><td>C++</td><td>Host</td></tr>
    <tr><td>6</td><td>ascendc-error-code.md</td><td>4</td><td>错误码使用与上报规范</td><td>ERR</td><td>C++</td><td>All</td></tr>
    <tr><td>7</td><td>ascendc-sunset-api.md</td><td>2</td><td>官方废弃机制驱动的 API 兼容性约束</td><td>SUNSET</td><td>C++</td><td>All</td></tr>
    <tr><td>8</td><td rowspan="3">C++通用规则</td><td>cpp-secure.md</td><td>15</td><td>C++安全编码规范</td><td>SEC</td><td>C++</td><td>All</td></tr>
    <tr><td>9</td><td>cpp-general.md</td><td>39</td><td>C++通用编码规范</td><td>GEN</td><td>C++</td><td>All</td></tr>
    <tr><td>10</td><td>cpp-style.md</td><td>16</td><td>C++代码风格规范</td><td>STYLE</td><td>C++</td><td>All</td></tr>
    <tr><td>11</td><td rowspan="3">领域专项规则</td><td>mc2-specific.md</td><td>41</td><td>MC2通信计算领域规则</td><td>MC2</td><td>不限</td><td>All</td></tr>
    <tr><td>12</td><td>ascendc-simt.md</td><td>15</td><td>AscendC SIMT领域规范</td><td>SIMT</td><td>C++</td><td>Kernel</td></tr>
    <tr><td>13</td><td>blaze-specific.md</td><td>21</td><td>Blaze矩阵乘框架领域规则</td><td>BLAZE</td><td>C++</td><td>Host, Kernel</td></tr>
  </tbody>
</table>
