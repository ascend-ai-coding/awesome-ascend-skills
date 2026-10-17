
#include "ops_common.h"

const std::vector<std::string> g_custom_lib_path = get_custom_lib_path();
const std::vector<std::string> g_default_custom_lib_path = get_default_custom_lib_path();

void *GetOpApiFuncAddrFromFeatureLib(const char *api_name)
{
    GET_OP_API_FUNC_FROM_FEATURE_LIB(ops_infer_handler, "libaclnn_ops_infer.so", api_name);
    GET_OP_API_FUNC_FROM_FEATURE_LIB(ops_train_handler, "libaclnn_ops_train.so", api_name);
    GET_OP_API_FUNC_FROM_FEATURE_LIB(math_handler, "libaclnn_math.so", api_name);
    GET_OP_API_FUNC_FROM_FEATURE_LIB(sparse_handler, "libaclnn_sparse.so", api_name);
    GET_OP_API_FUNC_FROM_FEATURE_LIB(fft_handler, "libaclnn_fft.so", api_name);
    GET_OP_API_FUNC_FROM_FEATURE_LIB(rand_handler, "libaclnn_rand.so", api_name);
    return nullptr;
}

c10::SmallVector<int64_t, SIZE> array_to_small_vector(c10::IntArrayRef shape)
{
    c10::SmallVector<int64_t, SIZE> shape_small_vec;
    for (uint64_t i = 0; i < shape.size(); i++) {
        shape_small_vec.emplace_back(shape[i]);
    }

    return shape_small_vec;
}
