#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
"""从 YAML 检视发现生成行内 PR 评论正文（格式固化，不依赖 LLM）。

输入：单条 YAML 发现文件（子 Agent 产出，含 clause_id/clause_title/severity/problem_desc/fix_suggestion）
     + 可选 --review-time（阶段3 统计的检视耗时，如 "16 分 28 秒"）
输出：统一格式的评论正文（品牌区 + 三段结构 + 可选尾部耗时标识，直接用于 POST body）

用法：
  python3 workflow.format_inline_comment.py --yaml {finding.yaml}
  python3 workflow.format_inline_comment.py --yaml {finding.yaml} --review-time "16 分 28 秒"
  python3 workflow.format_inline_comment.py --json '{"clause_id":"LOG-2",...}'
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

# ─── 共享内容逻辑：直接复用报告渲染器 assemble_report.py 的唯一实现 ──────────
# severity 兜底与 SUSPICIOUS 尾注只此一份，两侧调用同一函数对象，物理消灭内容漂移。
# 注：文件名带点（workflow.xxx.py）不可直接 import，用 importlib 按路径加载；
# 上游若重命名/重构这些函数，本脚本会在调用处响亮报错（AttributeError）——
# 静默漂移变成显式失败，属预期安全行为。
_AR_PATH = Path(__file__).resolve().parent / "workflow.assemble_report.py"
_ar_spec = importlib.util.spec_from_file_location("assemble_report", _AR_PATH)
assemble_report = importlib.util.module_from_spec(_ar_spec)
_ar_spec.loader.exec_module(assemble_report)


_LOGO_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAOsAAAB4CAIAAACcmpJvAABLQ0lEQVR42u19d9xUxfX3OWfm3i1P4aEL0i0o1iSK"
    "LaKILaKJBYMKUQExGmOLNdHE3hW7wRaJBUUjamLFLjY0FkQURUAUhacBT9ly75055/1jdvepPDyALb93z2c/uuw+"
    "e2fuzLlnTv0ekDwZY9ybuXPnXnrppbvssktFRQUiIiIUqUg/Bjn2q6io2GWXXS655JK5c+e24lURQfc/a63Wuqam"
    "5rzzzrv//vvT6TQAaK2Li1iknwIfR1EEAMlkcty4cZdffnmPHj2MMTn+FJEwDEXkrbfeGjRoUIHxiwtXpJ+gPAaA"
    "wYMHv/nmmyISRZGIoDFGKfXKK6/sv//+QRAkEolMJhOLxfbZZ58RI0ZstNFGxbUr0o9IIrJixYrZs2e/8MIL2Ww2"
    "mUym02nf92c9N2uPPfew1gIzL1u2rHfv3gAQi8UAYMyYMR999JEUqUg/JZo3b96hhx4KAL7vI2Lfvn2XLVvGzGit"
    "Pfzww2fOnOm4+9xzz73iiisAwForIkUZUKSfghhGRKf1nnvuuVdddZXj1TFjxsyYMQOeffZZAPA8DwAmTJggIkEQ"
    "OPYtUpF+OmStDYJARCZMmFDg2FmzZikimjt3LgB07979scceSyQSRERExUe/SD9NS27XXXedNm1aOp0WESJSy5Yt"
    "C8OQmSdNmjRmzBhrrVKquF5F+mkysbW2rKzs66+/njNnjtZ6yZIltGrVKsfae++9t1M4iitVpJ+4WvyrX/3KvVm9"
    "ejW5CAcAbLrppkVPcJF+4kREiOgCF9ZaRKSCw8FaW1ygIv1PUIFXRYSaKxnFpSnS/4pC3CSVi8tRpP9tvaK4BEUq"
    "cnCRilTk4CIVqcjBRSpycJGKVOTgIhWpyMFFKlKRg4tU5OAiFanIwUUq0vdK61ZPX0iYX7/BELHj3HkREeZWP8F1"
    "SbcXEW55hc6Mu9YruJQoAGDmtrdf+LaT1DaJqjDDdkdfj3XufJrh+o3Ydkl/gMywdjexsxzsdk4ptYEJQCKCgND6"
    "GgICIoiE2Da/XgQ6vRmI2G6GPjN3hok7uIL7dk0XEWHE7+D6HXy7rpRLPlzzXTvGJaL1HrF5PcQPUxhhrW11R3qd"
    "JlpZueKLLxaurlutSEkzoYyAAgCCICLAAoBAbvkQRQAIlVIqHo/tssuuitrjMBEBSNWs+Pzttz3yARkQwyjaaJMh"
    "/YdtZ0XU2pjYicYlS5Z88sknWuvC9lhrt9xyy8GDB6+1/MQx0Mrale++964iFBAQQFRBEPz85z/v06cPALz//vs1"
    "NdVEaK0FBASybLfbdtu+fft1prwFEY0xb7/9diaTAQQQQEITmQEDBgwbNgwAKisr33//faWUmwyLzQlIcT8XcE8/"
    "AgIiIbOAMCCCAIuAiFKqZ8/egwYN6t69eysma3dPv/nmmy+++CKVami+jFJ4LsWJFnRiBgEFRGs9aODgYcO2Ukox"
    "MyKmUqlXXnklmUwWHoz8xawIAiAzAwiAIBIAIQIggLCI21YEYBYLQPmxAVHciO4cjkKzww479OrVa42LO2/ePFdP"
    "16rCziH8vP/++2N/O7Znzx4b8gBdc821rSCDChRZKyIPnfCHCwHuAP/vQLdDbArARX371a1YYZiFueNKQId/cf31"
    "17cdd9DAQUuWLGn37tre6Ztvvtn2Co899pj7m4MPPqTtt9tuu3Vl5QpruePru3Ms1Zhqi8JxzDHHuL954oknNlxQ"
    "IeJGG/U5+ndHz58/v90Fd2v19ttvH3TQr7t27bYeQ/i+t9NOO0+bNs3dVzqd3muvvb5X6du3b9+qqiq3wvPmzeus"
    "DHbYPrffcfupp5waBIHyMJbwiAriEJvUgNzLvc9/joKstKcb6jN7jtzzzDPPaPc0F2ZN9O38jyrve+hg3VWjAIhF"
    "BNVl9rfLXr/9jgP+9ldrLGnKSZ8OV1YplSiJWxs4saGV/+XSL8ccPubll18qLSlbq6TUWhNRLK6REASIdCad1dpz"
    "3yaTcSJKlsassQIiIlp7H3308dixY5999lnP8zuhTkgymSBFpaU+MyutG+qyrvIWADxPE1EiEWOI2iwygGD+iMPc"
    "uedkWY5xQdycWKprVtx7372PP/747XdMPeKII601SumC9NVa33zzTX/60xnGmNyeIorbLzecgAgIcGFcQCAkAAFE"
    "JwvmzHl7zpy3X3zpxTtuvz2RSDw4/cHhOw3/9ttvfV+zGEHOT6rAIS0m21qNbPqwGf8IgRAiptOZqVOn9uzZMwxD"
    "3/c764twt/r3qX8/4fcnWLCxkrjyiMEYjvKvMP+KDBvD1rA1zPk31lgRpDC0JaUlN95wgzuh2mrS7vMXr5jSK7VK"
    "IbGxbJgiiyYzVMU/uP32+soVRGTFCnRgRIo796y1LMZNw7INTTZR5r333/cmTprQ8oxrR3QVLBsGYzliiUSstVaE"
    "87qjZWYWYyWyYhhtaLLxktgrr7w6adJEp7Ss1dBltixsha1YK9zSOnSi2h06zMKWjSm8JDJsLBsrxkhkJbISNb3h"
    "yErEYERZFcNkeSwdpMaNH//Ms08rpZ2lZS0rpa6//rpTTjlVeZgo08pDxvzV8gNFHBoJrZjcC9xAoZHQcGAlQiXx"
    "Up0si993733HHjvBWturd69rr702iiIWw2Iss2W2BU5gzjMGt+CQ1t+24B/LTITpdGbMmDEHHXRQE1ZaZziYmYnU"
    "vHnzTj/tdC+mlVLWRMwWckqt1lprrfKv5tT0iac9T3nZTPDX88/bZpttjbFtBbBlJqW+fPedykce2USXs7U5ZkIw"
    "YHppr+zbb2ZPvRMJwUKHNjMW/iOSe9YFUQCsNYky/18PzzzrrLM8z3Mwcp2x5wXA8W7hqWOWglTKDwLMYSIZe+CB"
    "By++5EKtPXdGr2Wm0kK+Fv6+oIZKM78PEeVWVrVaat36c2dqCwiLNVE87rHl008/PZ1Ou6dLKXr77bfOPOvsWMxX"
    "Co2xIgJSGCK3oZ6ntef2T3te84E0kcppuNYaGybL4jNmzJg2bRoAjBkz5pBDf5NJR7FYTCmtWpLWWmtqJr6ksNJa"
    "+1p7hQm4kbX2PM+LItO9R7cpU67rwIxuXw92ytPYsWMBIFEa1zGtY0rHwE+odXUi7zh8hyDIGmPaUWVZjLUifM+h"
    "h04H/NCvmI/lH2HZx1T+sSp7z4vN9ZNPK++KPv0bqquYOTRRx3rwLbfe5CbsJdBLoE6iTqKXQC9B8ZI4ANxxxx2F"
    "P26LqSEic+bMAYB4iYqXqFiSkiUeADz11FPubw4//HAASJZ5XtOV0UtgLK4TyTgAPDD9fhGJonDNenDjoMEDAKGk"
    "3E+U6rKKGABMnDjR/c1TT/8HABLJmJ+/so6vm/OHPPAT5CUwlqR4UieTcQCYOXNmAeJx9IGjAaC0LOnHKTf5Eg3r"
    "MkgsqQs3Hi/1lKJhw4ZlMhkRWbjws0Qi3vH0VAzIB+UD+aBia2em2++YWmDIzurBznH27bfLnn32GaXJcpgznEWF"
    "Gdunb+8RI/YsKy1DRBFhy0hO30FnWiIKAAMqEAiC4IwzzvD9mLW2pfogDCyRKF9/8erLK//97196FdpYAUZAZBEE"
    "BcoibRQrWbL86zfvumvfc/+MjNAJpw1JTsyhNPfJGc/XJ5100uabb7rHHiPXpBDnpaBg+0dTXnK0FLKMFlC0T5Mm"
    "Turfr9/uu+9hTKi134HCw8zMbBnbKoTMFhCd5c4R7LXXHptuuqkxpuDlJSIBYGEQJHTeSEaEdDr7n//8uzHV4MU1"
    "s0EQpTQivvTSy4cccojneV9+ufiVV14mD0MbiPNsCAUpM2jQoBEjRmitc+dfbqtYAEFy/3QoqO+8884nn3zix7Wg"
    "BQBmozz69NNP586du9NOO2266eY33njTK6+8Gov5bYDLJAiDp558MgiypEEEFCkT8NZbb/2LX/zCMhM5IwcLsjmb"
    "zQ4aNGjycZPdZgkAt1Eb9JpUCKXUBx9+WFdXH09qKxZAlFZhmnfZZeeHH364X7/+6+oGbssuKAAEIDDniilbGCnV"
    "BGJQQABYE1gmIREhK5tT4sM77t79hOP9Lt06dg934K5mZq29MIjGjRv/xptvDhwwsAMn8Zq0gIJC3EojQABhVp4K"
    "s+ERRxz5+uuvDx48pJNO6LbPT4GUUqEJj598/NgjxnbyCk899dRhhx3GbJyLTYRFpKqq0n37ySefphrTsRItYhGA"
    "iIKM3XH48CeffLJXz56duX5jY+Nvfzv22WefjpVop/QrIhG7YMGnO+20UxRFkydPnjx58hosq2jAwIHLv12eTPrW"
    "stJemMmMHj36yiuv7MhsEC5EtXCdosp1q1cXfoOICCgil112ab9+/YMwa6211hhrbOGdce9DayP3iSPnMmzLbGKF"
    "tF44e/bqF2b10+XGRhYjQJtRXGkaA7IoqAQxkl5emVqyaM790xVixwEkQspb0s1YLD84W47HY9988+0RY8c6lO92"
    "FFZp5qIWcKMV5i9i3TWbXd/5ZhERmW0s4X377fIjjjyisbEBEaXNbPNu5sJlseWgORO+eexz1apVxpggCIwxkTGh"
    "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"
    "HESWeeMBg/ccc+gCNnUeNSKkCNKEWRPWLPvKkKTBponThFVRpEpL/3DWmbKm7hoCABCCNAA0KkwROndvBgCaI78j"
    "WoQMQBohBZgCzAAagDAbRVEUZLPGGGuMMcZaY40xzRpqGGMKPWbCMDTGtAM4RGStHTZs2J57joxCab9RqQAKgsDE"
    "CRObtREnABg3fnwsFguzObAzAKuUygP1AyKGofF9/7jJx8EGtxT4qXOwk0lCto4kxdIo2Ai4CqUGbD22RrBsPUsU"
    "AOBkohEgDVE9RimUjBAB1C37RnueF09o7SvtKe0pTyvd9PK052nP92JP/WfmJx9/5Md8z9f5GlNsK7FEZMLJf8R4"
    "vNZGdch1wI1gI7aRwhTZECXLkPX0kojHHHvskM02a7cGgXLaC1jEECANkEEJADOKsk0RDQfMqqzSGYAMQApsBiVU"
    "CAD19XWe58Xice2662mtlFZaa6ULjS201lrrR/71r2XLlvm+747+dm0PADjmmAmQL6Fuyj4TARYCCjJmyJAhhx12"
    "WCEn3UWGhw4duu+++7ERDzWi43RLwESACETaRDJy5Mjttt3uB8hE+0lYcgxYB1CDDCjIxABpglRejOV60rbiBkRr"
    "GQiSG/VcDtAdMBRAtIEEXSn24QMz3t7tl5vuOVJbzklrV8CLKHmowDAyD/7j7iv/cn5JRflBRx513B9Pdpjsrctr"
    "AZCIOdpq25+POujAJx75V9+4Z03kGDHX1EPAKpWOuKSs7LQ/nd5REZhLsRVIAaQIDRKDMFoBKKCdWraKVO/+A96d"
    "Ny+hlAFjUYBtTNNVV189bOttNtlkk/aumxuxsbFhypQpN910c++Nek2aeNzk4yYPGjwI8hWRLbQjgNGjRw8ePOjL"
    "pV/qOEFLUa0UBWwnTJiQTCabm2KO9X9//PH/+c+/2YX0W02FGQAmTZrUdtD/uxwskBKqBSYBRGaANEIDikBTxlPb"
    "HXN2yfYH/erFW27uAuyDZQBh9tEvr1l952+PhJ49rEgkECJECBZUABwRhAQWIIy4qqa2SwkFQePtf586bfqD+48e"
    "fe011/Tp06cNE+f6nxxz6mkPPjpzJbBWhAwRula6YATAUysaglNO+P0mg4es1fQWgRAgg2JRTM7jksuuBOdDJDjw"
    "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"
)


def brand_lines():
    """品牌签名区——**全仓唯一定义**，行内评论统一使用；后续如有其他消费方
    （如汇总评论）也应复用本函数，防止样式漂移。

    结构：[logo] 署名文字 + 空行 + 水平线（署名与 --- 之间留空行，防 Setext 标题渲染）
    终版样式（用户确认，GitCode 渲染白名单实测）：
    - logo 置左 40px 显示（源图降采样 235x120 PNG，base64 常量内嵌于 _LOGO_B64——
      仓库 check-pr/OAT 门禁不允许 skill 目录与新增提交携带图片二进制，
      故以文本常量随脚本分发，skill 安装后自包含；_LOGO_B64 即仓内唯一真源，
      需要图片文件时可由该常量 base64 解码生成）
    - vertical-align:middle 与署名文字居中，margin-right 8px 间距
    - 无外层 CSS 边框/圆角——图自带边框唯一呈现（外层画框会形成双框差异）
    - 署名：斜体、保持原有大小写、深炭灰 #42453f、0.3px 轻微字距
    - _LOGO_B64 为空时降级为纯文字署名
    """
    logo_tag = (f'<img src="data:image/png;base64,{_LOGO_B64}" height="40" '
                f'style="vertical-align: middle; margin-right: 8px;" '
                f'alt="cannbot-review"/>') if _LOGO_B64 else ""
    brand_text = ('<span style="font-style: italic; '
                  'letter-spacing: 0.3px; color: #42453f;">'
                  'Powered by cannbot-review</span>')
    return [f"{logo_tag} {brand_text}".strip(), "", "---", ""]


def timing_badge_lines(review_time: str) -> list:
    """检视耗时标识——**全仓唯一定义**，行内评论尾部右对齐小字。

    样式（用户确认）：`<div align="right"><sub>⏱ 检视耗时：{值}</sub></div>`，
    右对齐 + sub 小字号，置于正文末尾（隔一个空行）。
    review_time 为空时返回空列表——不输出标识，行为与旧版完全一致（向后兼容）。
    值来源：阶段3 报告「检视耗时」章节的纯流程耗时（无等待时即总耗时，如 "16 分 28 秒"），
    由 workflow 阶段4 传入 steps 后经 --review-time 带入，脚本不自行计时。
    """
    text = str(review_time or "").strip()
    if not text:
        return []
    return ["", f'<div align="right"><sub>⏱ 检视耗时：{text}</sub></div>']


def format_comment(d: dict) -> str:
    """按统一三段格式输出评论正文（确定性，无 LLM）。

    内容逻辑（severity 兜底、SUSPICIOUS 尾注）复用 assemble_report 的实现，
    本函数只负责评论侧样式（品牌区 + 粗体三段标题行）。
    """
    clause_id = d.get("clause_id", "")
    title = str(d.get("clause_title", "") or d.get("title", "") or "")
    severity = assemble_report.resolve_severity(d)

    # problem_desc 缺失时回退 content 字段，再交共享渲染器统一处理 SUSPICIOUS 尾注
    d2 = dict(d)
    if not str(d2.get("problem_desc", "") or ""):
        d2["problem_desc"] = str(d2.get("content", "") or "")
    desc = assemble_report.render_problem_desc(d2)
    suggestion = str(d.get("fix_suggestion", "") or "")

    # 品牌签名区（唯一定义，行内评论与汇总评论共用）
    lines = brand_lines()
    if clause_id:
        header = f"**【{clause_id}】{title}**｜严重级别：{severity}"
    else:
        header = f"**{title}**｜严重级别：{severity}"
    lines += [
        header,
        "",
        f"**问题分析**：{desc}",
    ]
    if suggestion:
        lines += ["", f"**修改建议**：{suggestion}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--yaml", help="YAML 发现文件路径")
    group.add_argument("--json", help="JSON 格式的单条发现")
    ap.add_argument("--review-time", default="",
                    help='检视耗时（阶段3 统计的纯流程耗时值，如 "16 分 28 秒"；'
                         "报告无该行（无等待）时即总耗时）；缺省则评论尾部不带耗时标识")
    args = ap.parse_args()

    if args.yaml:
        import yaml
        with open(args.yaml, encoding="utf-8") as f:
            d = yaml.safe_load(f)
    else:
        d = json.loads(args.json)

    # 输入防护——与 assemble_report.py 的非 dict 防护口径一致（空文件/列表输入
    # 报语义错误退出，而非 AttributeError traceback）
    if not isinstance(d, dict):
        print(f"错误: 输入必须是 mapping（yaml/json 对象），实际类型: {type(d).__name__}",
              file=sys.stderr)
        return 2

    # 正文 = 品牌区 + 三段结构 + 可选尾部耗时标识（唯一定义见 timing_badge_lines）
    print("\n".join([format_comment(d)] + timing_badge_lines(args.review_time)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
