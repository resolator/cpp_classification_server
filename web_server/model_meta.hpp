#pragma once

#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>



struct ModelMeta {
    std::vector<int64_t>* input_dims;
    std::vector<int64_t>* output_dims;
    Ort::MemoryInfo* mem_info;
    Ort::Session* sess;
    std::vector<const char*>* input_names;
    std::vector<const char*>* output_names;
    std::vector<std::string>* labels;
};
