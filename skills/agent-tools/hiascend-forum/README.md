# hiascend-forum

基于华为[昇腾社区论坛](https://www.hiascend.com/forum/all)上开发者反馈的内容，识别特定领域的易用性问题，用于做产品改进分析。

---

## skills介绍

**hiascend-forum-fetcher** 支持根据给定的时间范围，读取昇腾社区论坛在该范围内的帖子，并输出一个excel文档。文档内容包含帖子ID、所属板块、分类、标题、发布时间、论坛链接。

**hiascend-forum-analyzer** 支持基于hiascend-forum-fetcher获取的文档，根据给定的内容，识别论坛上开发者反馈的相关问题帖子，并输出一个excel文档。文档内容包含帖子ID、所属板块、分类、标题、发布时间、论坛链接。

---

## 使用指南

### hiascend-forum-fetcher

在安装好hiascend-forum-fetcher skills后，可以使用类似“使用hiascend-forum-fetcher skills获取26年3月的昇腾社区论坛贴”，让AI编程工具输出具体时间范围内的昇腾社区论坛贴（输出格式为hiascend_topics_20260301_20260331.xlsx）。

### hiascend-forum-analyzer

在安装好hiascend-forum-analyzer skills后，可以使用类似“使用hiascend-forum-analyzer skills读取hiascend_topics_20260301_20260331.xlsx文档，并整理算子相关问题”，让AI编程工具输出昇腾社区论坛中具体内容的问题贴。

---

## 许可证

MIT License

Copyright (c) 2024 Ascend AI Coding

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.