#include "PCFG.h"
#include <cuda_runtime.h> 
#include <iostream>
#include <numeric>

using namespace std;

// 设置一个合理的阈值来决定何时使用GPU
#define GPU_THRESHOLD 100000

// --- CUDA C++ 内核函数 ---
// 这个内核函数在GPU上执行，负责将前缀和后缀拼接起来
__global__ void ConcatKernel(const char* prefix, int prefix_len, const char* flat_suffixes, const int* offsets, const int* lengths, int task_size, char* flat_output, int* output_offsets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < task_size) {
        // 计算当前线程输出的起始位置
        int output_start_pos = output_offsets[idx];

        // 拷贝前缀
        for (int i = 0; i < prefix_len; ++i) {
            flat_output[output_start_pos + i] = prefix[i];
        }

        // 拷贝后缀
        int suffix_start_pos = offsets[idx];
        int suffix_len = lengths[idx];
        for (int i = 0; i < suffix_len; ++i) {
            flat_output[output_start_pos + prefix_len + i] = flat_suffixes[suffix_start_pos + i];
        }
    }
}


// --- CPU 辅助函数 (将 vector<string> 扁平化) ---
void Flatten(const vector<string>& suffixes, char** flat_data, int** offsets, int** lengths, int& total_chars) {
    total_chars = 0;
    for (const auto& s : suffixes) {
        total_chars += s.length();
    }

    *flat_data = new char[total_chars];
    *offsets = new int[suffixes.size()];
    *lengths = new int[suffixes.size()];
    
    int current_pos = 0;
    for (size_t i = 0; i < suffixes.size(); ++i) {
        (*offsets)[i] = current_pos;
        (*lengths)[i] = suffixes[i].length();
        memcpy(*flat_data + current_pos, suffixes[i].c_str(), suffixes[i].length());
        current_pos += suffixes[i].length();
    }
}

// --- CPU 辅助函数 (将扁平化的 char* 还原为 vector<string>) ---
void Unflatten(const char* flat_output, const int* output_offsets, const int* lengths, int prefix_len, int task_size, vector<string>& guesses) {
    guesses.reserve(guesses.size() + task_size);
    for (int i = 0; i < task_size; ++i) {
        int start_pos = output_offsets[i];
        int total_len = prefix_len + lengths[i];
        guesses.emplace_back(flat_output + start_pos, total_len);
    }
}


void PriorityQueue::CalProb(PT &pt)
{
    // ... (这部分代码保持不变) ...
    pt.prob = pt.preterm_prob;
    int index = 0;
    for (int idx : pt.curr_indices)
    {
        if (pt.content[index].type == 1)
        {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index += 1;
    }
}

void PriorityQueue::init()
{
    // ... (这部分代码保持不变) ...
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1) pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            if (seg.type == 2) pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            if (seg.type == 3) pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        CalProb(pt);
        priority.emplace_back(pt);
    }
}

void PriorityQueue::PopNext()
{
    // ... (这部分代码保持不变) ...
    Generate(priority.front());
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        CalProb(pt);
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }
    priority.erase(priority.begin());
}

vector<PT> PT::NewPTs()
{
    // ... (这部分代码保持不变) ...
    vector<PT> res;
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        int init_pivot = pivot;
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            curr_indices[i] += 1;
            if (curr_indices[i] < max_indices[i])
            {
                pivot = i;
                res.emplace_back(*this);
            }
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }
    return res;
}

// 这个函数是PCFG并行化算法的主要载体
void PriorityQueue::Generate(PT pt)
{
    // 计算PT的概率，这里主要是给PT的概率进行初始化
    CalProb(pt);

    string prefix;
    segment* last_seg_model_ptr;
    int task_size;

    // 准备任务：确定前缀(prefix)和最后一个段的信息
    if (pt.content.size() == 1) {
        prefix = "";
        task_size = pt.max_indices[0];
        if (pt.content[0].type == 1) last_seg_model_ptr = &m.letters[m.FindLetter(pt.content[0])];
        if (pt.content[0].type == 2) last_seg_model_ptr = &m.digits[m.FindDigit(pt.content[0])];
        if (pt.content[0].type == 3) last_seg_model_ptr = &m.symbols[m.FindSymbol(pt.content[0])];
    } else {
        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外），构成前缀
        int seg_idx = 0;
        for (int idx : pt.curr_indices) {
            if (pt.content[seg_idx].type == 1) prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            if (pt.content[seg_idx].type == 2) prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            if (pt.content[seg_idx].type == 3) prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            seg_idx++;
            if (seg_idx == pt.content.size() - 1) break;
        }
        
        // 准备最后一个段的信息
        int last_seg_idx = pt.content.size() - 1;
        task_size = pt.max_indices[last_seg_idx];
        if (pt.content[last_seg_idx].type == 1) last_seg_model_ptr = &m.letters[m.FindLetter(pt.content[last_seg_idx])];
        if (pt.content[last_seg_idx].type == 2) last_seg_model_ptr = &m.digits[m.FindDigit(pt.content[last_seg_idx])];
        if (pt.content[last_seg_idx].type == 3) last_seg_model_ptr = &m.symbols[m.FindSymbol(pt.content[last_seg_idx])];
    }
    
    // --- 动态调度器 ---
    int device_count;
    cudaGetDeviceCount(&device_count);
    bool gpu_available = (device_count > 0);

    if (task_size > GPU_THRESHOLD && gpu_available) {
        // --- GPU 加速路径 (真实 CUDA 代码) ---
        // 1. 扁平化后缀数据
        char *h_flat_suffixes; int *h_offsets, *h_lengths; int total_suffix_chars;
        Flatten(last_seg_model_ptr->ordered_values, &h_flat_suffixes, &h_offsets, &h_lengths, total_suffix_chars);

        // 2. 准备输出缓冲区信息
        int prefix_len = prefix.length();
        long long total_output_chars = (long long)prefix_len * task_size;
        for(int i=0; i < task_size; ++i) total_output_chars += h_lengths[i];

        char* h_flat_output = new char[total_output_chars];
        int* h_output_offsets = new int[task_size];
        int current_pos = 0;
        for(int i=0; i < task_size; ++i) {
            h_output_offsets[i] = current_pos;
            current_pos += prefix_len + h_lengths[i];
        }

        // 3. 在GPU上分配内存
        char *d_prefix, *d_flat_suffixes, *d_flat_output;
        int *d_offsets, *d_lengths, *d_output_offsets;
        cudaMalloc((void**)&d_prefix, prefix_len * sizeof(char));
        cudaMalloc((void**)&d_flat_suffixes, total_suffix_chars * sizeof(char));
        cudaMalloc((void**)&d_offsets, task_size * sizeof(int));
        cudaMalloc((void**)&d_lengths, task_size * sizeof(int));
        cudaMalloc((void**)&d_flat_output, total_output_chars * sizeof(char));
        cudaMalloc((void**)&d_output_offsets, task_size * sizeof(int));

        // 4. 将数据从CPU拷贝到GPU
        cudaMemcpy(d_prefix, prefix.c_str(), prefix_len * sizeof(char), cudaMemcpyHostToDevice);
        cudaMemcpy(d_flat_suffixes, h_flat_suffixes, total_suffix_chars * sizeof(char), cudaMemcpyHostToDevice);
        cudaMemcpy(d_offsets, h_offsets, task_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lengths, h_lengths, task_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_output_offsets, h_output_offsets, task_size * sizeof(int), cudaMemcpyHostToDevice);

        // 5. 启动拼接内核
        int threadsPerBlock = 256;
        int numBlocks = (task_size + threadsPerBlock - 1) / threadsPerBlock;
        ConcatKernel<<<numBlocks, threadsPerBlock>>>(d_prefix, prefix_len, d_flat_suffixes, d_offsets, d_lengths, task_size, d_flat_output, d_output_offsets);
        cudaDeviceSynchronize(); // 等待内核执行完成

        // 6. 将结果拷回CPU
        cudaMemcpy(h_flat_output, d_flat_output, total_output_chars * sizeof(char), cudaMemcpyDeviceToHost);

        // 7. 将扁平化的结果转换回 vector<string> 并存入全局guesses
        Unflatten(h_flat_output, h_output_offsets, h_lengths, prefix_len, task_size, this->guesses);
        this->total_guesses += task_size;
        
        // 8. 释放GPU和CPU内存
        cudaFree(d_prefix); cudaFree(d_flat_suffixes); cudaFree(d_offsets); cudaFree(d_lengths); cudaFree(d_flat_output); cudaFree(d_output_offsets);
        delete[] h_flat_suffixes; delete[] h_offsets; delete[] h_lengths;
        delete[] h_flat_output; delete[] h_output_offsets;
    } else {
        // --- CPU 多线程路径 (OpenMP 实现) ---
        vector<vector<string>> local_guesses(omp_get_max_threads());
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < task_size; i += 1)
        {
            int tid = omp_get_thread_num();
            string temp = prefix + last_seg_model_ptr->ordered_values[i];
            local_guesses[tid].emplace_back(temp);
        }

        // 串行合并所有线程的局部结果到全局guesses
        size_t total_new_guesses = 0;
        for(int i=0; i<omp_get_max_threads(); ++i) total_new_guesses += local_guesses[i].size();
        guesses.reserve(guesses.size() + total_new_guesses); 
        for(int i = 0; i < omp_get_max_threads(); ++i) {
            if (!local_guesses[i].empty()) {
                guesses.insert(guesses.end(), make_move_iterator(local_guesses[i].begin()), make_move_iterator(local_guesses[i].end()));
            }
        }
        total_guesses += total_new_guesses;
    }
}