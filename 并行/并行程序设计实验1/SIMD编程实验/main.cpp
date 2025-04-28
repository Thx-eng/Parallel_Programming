#include "PCFG.h" 
#include "md5.h"  

#include <chrono>
#include <fstream>
#include <iomanip> 
#include <vector>
#include <string>
#include <iostream> 
#include <algorithm> 
#include <cstdio>    

using namespace std;
using namespace chrono;

// +++ 优化：快速将 4 个字节（一个 bit32）转换为 8 个十六进制字符 +++
inline void word_to_hex_string(char* dest, bit32 val) {
    snprintf(dest, 9, "%08x", val); // 使用 snprintf 保证格式和原始 cout << hex << setw(8) << setfill('0') 一致
}

// +++ 优化：快速将 16 字节 MD5 (4x bit32) 转换为 32 个十六进制字符 +++
inline void bytes_to_hex_string(char* dest, const bit32* hash_state) {
    word_to_hex_string(dest + 0,  hash_state[0]);
    word_to_hex_string(dest + 8,  hash_state[1]);
    word_to_hex_string(dest + 16, hash_state[2]);
    word_to_hex_string(dest + 24, hash_state[3]);
    dest[32] = '\0'; 
}

// 编译指令如下
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O1
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2

int main()
{
    // --- 时间和队列初始化 ---
    double time_hash = 0;
    double time_guess_total = 0; // 用于最后计算总时间
    double time_train = 0;
    PriorityQueue q;

    // --- 模型训练  ---
    auto start_train = system_clock::now();
    try {
        q.m.train("/guessdata/Rockyou-singleLined-full.txt");
        q.m.order();
    } catch (const std::exception& e) {
        cerr << "Error during training: " << e.what() << endl;
        return 1;
    }
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    // --- 初始化猜测队列  ---
    try {
         q.init();
    } catch (const std::exception& e) {
         cerr << "Error during queue initialization: " << e.what() << endl;
         return 1;
    }
    cout << "here" << endl; 

    // --- [优化] 文件输出设置 ---
    std::ofstream a;
    const char* output_filename = "./files/simd_results.txt";
    const size_t file_buffer_size = 1024 * 1024;
    std::vector<char> file_output_buffer(file_buffer_size);
    a.rdbuf()->pubsetbuf(file_output_buffer.data(), file_buffer_size);
    a.open(output_filename);
    if (!a.is_open()) {
        cerr << "Error: Could not open output file " << output_filename << endl;
        return 1;
    }

    // --- [优化] MD5 计算所需的可重用缓冲区 ---
    std::vector<Byte> neon_interleaved_buffer;
    std::vector<Byte> serial_padded_buffer;

    // --- 主循环变量  ---
    int curr_num = 0;           // 关键状态变量，int 类型，意义见原始逻辑
    int history = 0;          // 已处理历史总数，int 类型
    size_t current_total_guesses = 0; // 当前 q.guesses.size()
    const int report_interval = 100000;   
    const int process_threshold = 1000000; 
    const int generate_n = 10000000; 

    auto start_loop_timer = system_clock::now(); // 记录主循环开始时间

    // --- 主循环  ---
    while (!q.priority.empty())
    {
        q.PopNext(); // 生成密码
        current_total_guesses = q.guesses.size(); // 更新当前缓冲区总大小

        // --- 进度报告逻辑  ---
        if (current_total_guesses - curr_num >= report_interval) // 使用 size_t 和 int 比较，安全
        {
            cout << "Guesses generated: " << history + current_total_guesses << endl; // 报告使用 history + 当前总数
            // --- 关键: 更新 curr_num ---
            if (current_total_guesses <= static_cast<size_t>(std::numeric_limits<int>::max())) {
                 curr_num = static_cast<int>(current_total_guesses); // 更新为当前总数 
            } else {
                 cerr << "Warning: current_total_guesses exceeds INT_MAX, curr_num logic might break." << endl;
                 curr_num = std::numeric_limits<int>::max(); 
            }
        }

        // --- 退出条件检查  ---
        // 使用 history (int) + current_total_guesses (size_t) > generate_n (int)
        if (static_cast<long long>(history) + static_cast<long long>(current_total_guesses) > generate_n)
        {
             // 检测到退出条件，跳出循环前不进行处理，直接在循环外计算时间
             break;
        }


        // --- 处理触发条件检查  ---
        if (curr_num > process_threshold)
        {
            // 记录下实际要处理的数量 
            size_t actual_to_process = current_total_guesses;

            // --- 哈希计算部分 (使用内部优化) ---
            auto start_hash_timer = system_clock::now();

            const size_t BATCH_SIZE = NEON_LANES;
            std::vector<uint8_t> batch_hash_results(BATCH_SIZE * 16);
            std::vector<std::string> batch_passwords_temp;
            batch_passwords_temp.reserve(BATCH_SIZE);
            bit32 state_buffer[4];
            char hex_buffer[33];

            size_t processed_in_this_round = 0;

            // 处理 actual_to_process 数量的密码
            while (processed_in_this_round < actual_to_process)
            {
                size_t remaining_in_buffer = actual_to_process - processed_in_this_round;
                size_t current_batch_size = std::min((size_t)BATCH_SIZE, remaining_in_buffer);

                if (current_batch_size == BATCH_SIZE) {
                    // NEON 批处理
                    batch_passwords_temp.clear();
                    auto start_iter = q.guesses.begin() + processed_in_this_round;
                    auto end_iter = start_iter + current_batch_size;
                    batch_passwords_temp.assign(start_iter, end_iter);
                    MD5Hash_NEON_Batch(batch_passwords_temp, current_batch_size, batch_hash_results.data(), neon_interleaved_buffer);

                    // 输出结果
                    for (size_t i = 0; i < current_batch_size; ++i) {
                        const string& pw = batch_passwords_temp[i];
                        const uint8_t* hash_ptr = batch_hash_results.data() + i * 16;
                        memcpy(state_buffer, hash_ptr, 16);
                        bytes_to_hex_string(hex_buffer, state_buffer);
                        a << pw << "\t" << hex_buffer << "\n";
                    }
                    processed_in_this_round += current_batch_size;

                } else if (current_batch_size > 0) {
                    // 串行处理剩余
                    for (size_t i = 0; i < current_batch_size; ++i) {
                        const string& pw = q.guesses[processed_in_this_round + i];
                        MD5Hash(pw, state_buffer, serial_padded_buffer);
                        bytes_to_hex_string(hex_buffer, state_buffer);
                        a << pw << "\t" << hex_buffer << "\n";
                    }
                    processed_in_this_round += current_batch_size;
                } else {
                    break;
                }
            } 

            // --- 哈希计时 ---
            auto end_hash_timer = system_clock::now();
            auto duration_hash = duration_cast<microseconds>(end_hash_timer - start_hash_timer);
            time_hash += double(duration_hash.count()) * microseconds::period::num / microseconds::period::den;

            // --- 关键: 更新状态  ---
            history += curr_num; // history 增加的是触发处理时的 curr_num 值
            curr_num = 0;       // 重置 curr_num 为 0
            q.guesses.clear();  // 清空密码缓冲区
            current_total_guesses = 0; // 清空后，当前大小也为0

        } 

    } 

    // --- 循环结束后 ---
    a.flush();
    a.close();

    // --- 计算最终时间 ---
    auto end_loop_timer = system_clock::now();
    // 总时间应该是从循环开始到循环结束
    auto duration_total = duration_cast<microseconds>(end_loop_timer - start_loop_timer);
    time_guess_total = double(duration_total.count()) * microseconds::period::num / microseconds::period::den;

    // --- 输出最终统计  ---
    cout << "Guess time:" << time_guess_total - time_hash << "seconds"<< endl; // 总时间 - hash时间
    cout << "Hash time:" << time_hash << "seconds"<<endl;
    cout << "Train time:" << time_train <<"seconds"<<endl;

    return 0;
}