#include "gpu_generate.h"
#include <iostream>
#include <numeric> // For std::accumulate

// 辅助函数：将字符串向量扁平化为GPU可处理的格式
static void flatten_strings(const std::vector<std::string>& vec_str,
                            std::vector<char>& flat_chars,
                            std::vector<int>& offsets,
                            std::vector<int>& lengths) {
    long long total_chars = 0;
    for (const auto& s : vec_str) {
        total_chars += s.length();
    }
    flat_chars.reserve(total_chars);
    offsets.reserve(vec_str.size());
    lengths.reserve(vec_str.size());

    int current_offset = 0;
    for (const auto& s : vec_str) {
        flat_chars.insert(flat_chars.end(), s.begin(), s.end());
        offsets.push_back(current_offset);
        lengths.push_back(s.length());
        current_offset += s.length();
    }
}

// 辅助函数：将扁平化的GPU结果转换回字符串向量
static void unflatten_strings(const std::vector<char>& flat_chars,
                              const std::vector<int>& offsets,
                              const std::vector<int>& lengths,
                              std::vector<std::string>& vec_str) {
    vec_str.reserve(vec_str.size() + lengths.size());
    for (size_t i = 0; i < lengths.size(); ++i) {
        if (offsets[i] + lengths[i] <= flat_chars.size()) {
            vec_str.emplace_back(&flat_chars[offsets[i]], lengths[i]);
        }
    }
}

// CUDA 核函数：用于单segment情况，在GPU上执行并行复制
__global__ void copy_kernel(const char* d_input, char* d_output, const int* d_offsets, const int* d_lengths, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const char* src = d_input + d_offsets[i];
    char* dst = d_output + d_offsets[i]; // 输出和输入的偏移/长度结构相同
    
    for(int k=0; k < d_lengths[i]; ++k) {
        dst[k] = src[k];
    }
}


// CUDA 核函数：用于多segment情况，在GPU上执行并行字符串拼接
__global__ void generate_kernel_multi(
    const char* d_prefix, int prefix_len,
    const char* d_suffixes_flat, const int* d_suffix_offsets, const int* d_suffix_lengths,
    char* d_output_flat, const int* d_output_offsets, int num_suffixes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_suffixes) return;

    // 定位到当前线程应该写入的输出位置
    char* output_start = d_output_flat + d_output_offsets[i];
    // 定位到当前线程需要读取的后缀
    const char* suffix_start = d_suffixes_flat + d_suffix_offsets[i];
    int suffix_len = d_suffix_lengths[i];

    // 复制前缀
    for (int k = 0; k < prefix_len; ++k) {
        output_start[k] = d_prefix[k];
    }

    // 复制后缀
    for (int k = 0; k < suffix_len; ++k) {
        output_start[prefix_len + k] = suffix_start[k];
    }
}

void generate_on_gpu_single_segment(const std::vector<std::string>& values, std::vector<std::string>& guesses) {
    // 1. 在主机端（CPU）扁平化数据
    std::vector<char> h_flat_values;
    std::vector<int> h_offsets;
    std::vector<int> h_lengths;
    flatten_strings(values, h_flat_values, h_offsets, h_lengths);
    size_t num_values = values.size();
    size_t flat_size = h_flat_values.size();

    // 2. 在设备端（GPU）分配内存
    char *d_flat_values, *d_output_flat;
    int *d_offsets, *d_lengths;
    cudaMalloc((void**)&d_flat_values, flat_size * sizeof(char));
    cudaMalloc((void**)&d_output_flat, flat_size * sizeof(char));
    cudaMalloc((void**)&d_offsets, num_values * sizeof(int));
    cudaMalloc((void**)&d_lengths, num_values * sizeof(int));

    // 3. 将数据从主机拷贝到设备
    cudaMemcpy(d_flat_values, h_flat_values.data(), flat_size * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets.data(), num_values * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, h_lengths.data(), num_values * sizeof(int), cudaMemcpyHostToDevice);

    // 4. 配置并启动核函数
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_values + threadsPerBlock - 1) / threadsPerBlock;
    copy_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_flat_values, d_output_flat, d_offsets, d_lengths, num_values);

    // 5. 将结果从设备拷贝回主机
    std::vector<char> h_output_flat(flat_size);
    cudaMemcpy(h_output_flat.data(), d_output_flat, flat_size * sizeof(char), cudaMemcpyDeviceToHost);

    // 6. 释放设备内存
    cudaFree(d_flat_values);
    cudaFree(d_output_flat);
    cudaFree(d_offsets);
    cudaFree(d_lengths);

    // 7. 将扁平化的结果转换回字符串向量并存入guesses
    unflatten_strings(h_output_flat, h_offsets, h_lengths, guesses);
}


void generate_on_gpu_multi_segment(const std::string& prefix, const std::vector<std::string>& suffixes, std::vector<std::string>& guesses) {
    // 1. 扁平化后缀数据
    std::vector<char> h_flat_suffixes;
    std::vector<int> h_suffix_offsets;
    std::vector<int> h_suffix_lengths;
    flatten_strings(suffixes, h_flat_suffixes, h_suffix_offsets, h_suffix_lengths);
    size_t num_suffixes = suffixes.size();
    size_t flat_suffixes_size = h_flat_suffixes.size();

    // 2. 计算输出数据的结构
    int prefix_len = prefix.length();
    std::vector<int> h_output_lengths(num_suffixes);
    std::vector<int> h_output_offsets(num_suffixes);
    long long total_output_size = 0;
    int current_offset = 0;
    for (size_t i = 0; i < num_suffixes; ++i) {
        h_output_lengths[i] = prefix_len + h_suffix_lengths[i];
        h_output_offsets[i] = current_offset;
        current_offset += h_output_lengths[i];
    }
    total_output_size = current_offset;

    // 3. 在设备端分配内存
    char *d_prefix, *d_flat_suffixes, *d_output_flat;
    int *d_suffix_offsets, *d_suffix_lengths, *d_output_offsets;
    cudaMalloc((void**)&d_prefix, prefix_len * sizeof(char));
    cudaMalloc((void**)&d_flat_suffixes, flat_suffixes_size * sizeof(char));
    cudaMalloc((void**)&d_suffix_offsets, num_suffixes * sizeof(int));
    cudaMalloc((void**)&d_suffix_lengths, num_suffixes * sizeof(int));
    cudaMalloc((void**)&d_output_flat, total_output_size * sizeof(char));
    cudaMalloc((void**)&d_output_offsets, num_suffixes * sizeof(int));

    // 4. 将数据从主机拷贝到设备
    cudaMemcpy(d_prefix, prefix.c_str(), prefix_len * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flat_suffixes, h_flat_suffixes.data(), flat_suffixes_size * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_suffix_offsets, h_suffix_offsets.data(), num_suffixes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_suffix_lengths, h_suffix_lengths.data(), num_suffixes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_offsets, h_output_offsets.data(), num_suffixes * sizeof(int), cudaMemcpyHostToDevice);

    // 5. 配置并启动核函数
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_suffixes + threadsPerBlock - 1) / threadsPerBlock;
    generate_kernel_multi<<<blocksPerGrid, threadsPerBlock>>>(
        d_prefix, prefix_len, d_flat_suffixes, d_suffix_offsets, d_suffix_lengths, 
        d_output_flat, d_output_offsets, num_suffixes);
    
    // 6. 将结果从设备拷贝回主机
    std::vector<char> h_output_flat(total_output_size);
    cudaMemcpy(h_output_flat.data(), d_output_flat, total_output_size * sizeof(char), cudaMemcpyDeviceToHost);
    
    // 7. 释放设备内存
    cudaFree(d_prefix);
    cudaFree(d_flat_suffixes);
    cudaFree(d_suffix_offsets);
    cudaFree(d_suffix_lengths);
    cudaFree(d_output_flat);
    cudaFree(d_output_offsets);

    // 8. 将扁平化的结果转换回字符串向量并存入guesses
    unflatten_strings(h_output_flat, h_output_offsets, h_output_lengths, guesses);
}