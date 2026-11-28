/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <acl/acl.h>
#include <platform/platform_ascendc.h>
#include <adv_api/sort/sort_tiling_intf.h>
#include <iostream>
int main()
{
    if (aclInit(nullptr) || aclrtSetDevice(0))
        return 1;
    auto p = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (!p)
        return 2;
    uint64_t ub = 0;
    p->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    std::cout << "{\"soc\":\"" << aclrtGetSocName() << "\",\"ub\":" << ub << ",\"aiv\":" << p->GetCoreNumAiv()
              << ",\"aic\":" << p->GetCoreNumAic() << ",\"system_workspace\":" << p->GetLibApiWorkSpaceSize()
              << ",\"concat\":{";
    bool first = true;
    for (int n : {32, 64, 128, 256, 288, 1024, 2048, 4096, 4128, 8192})
        for (int size : {2, 4}) {
            if (!first)
                std::cout << ",";
            first = false;
            std::cout << "\"" << n << "-" << size << "\":" << AscendC::GetConcatTmpSize(*p, n, size);
        }
    std::cout << "}}\n";
    aclrtResetDevice(0);
    aclFinalize();
}
