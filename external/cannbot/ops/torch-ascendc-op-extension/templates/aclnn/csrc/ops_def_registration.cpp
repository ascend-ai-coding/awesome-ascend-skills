
#include <torch/extension.h>
#include <torch/library.h>

// 在 custom 命名空间里注册所有自定义 aten IR，每新增一个算子就在这里加一条 m.def。
// 全工程只能有这一处 TORCH_LIBRARY(custom, ...)，impl 文件一律用 TORCH_LIBRARY_IMPL。
// step1, 为新增自定义算子添加定义
TORCH_LIBRARY(custom, m) {
    m.def("npu_x_custom_op(Tensor x1, Tensor x2, Tensor! out) -> Tensor");
}

// 通过pybind将c++接口和python接口绑定，这里绑定的是接口不是算子
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}
