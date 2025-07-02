#pragma once

#include <vector>
#include <string>

// 当PT只有一个segment时，在GPU上生成猜测的函数
// values: 模型中一个segment的所有value
// guesses: 用于存储生成结果的向量
void generate_on_gpu_single_segment(const std::vector<std::string>& values, std::vector<std::string>& guesses);

// 当PT有多个segment时，在GPU上生成猜测的函数
// prefix: 由前几个segment拼接成的固定前缀
// suffixes: 最后一个segment的所有value，作为后缀
// guesses: 用于存储生成结果的向量
void generate_on_gpu_multi_segment(const std::string& prefix, const std::vector<std::string>& suffixes, std::vector<std::string>& guesses);