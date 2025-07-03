#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
// #include <mpi.h> // 如果使用MPI，请取消此行注释

using namespace std;
using namespace chrono;

int main(int argc, char* argv[])
{
    int rank = 0;
    int size = 1;

    // --- MPI 初始化代码框架 ---
    // MPI_Init(&argc, &argv);
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Comm_size(MPI_COMM_WORLD, &size);

    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    
    PriorityQueue q;

    // 只有Rank 0执行训练和初始化
    if (rank == 0) {
        cout << "Start training model..." << endl;
        auto start_train = system_clock::now();
        q.m.train("./input/Rockyou-singleLined-full.txt");
        q.m.order();
        auto end_train = system_clock::now();
        auto duration_train = duration_cast<microseconds>(end_train - start_train);
        time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;
        cout << "Training finished. Time: " << time_train << "s" << endl;

        q.init();
        cout << "Priority queue initialized." << endl;
    }
    
    // --- 生产者-消费者模型框架 ---
    // if (rank == 0) {
    //   // Rank 0 (Producer) 逻辑:
    //   // - 维护优先队列
    //   // - 监听消费者请求
    //   // - 分发任务 (prefix, suffix_id)
    // } else {
    //   // Ranks > 0 (Consumer) 逻辑:
    //   // - 请求任务
    //   // - 接收任务
    //   // - 调用 Generate() 生成密码
    //   // - 调用 MD5Hash_NEON_Batch() 哈希
    //   // - (可选)返回结果
    // }

    // 当前为单机融合模式 (模拟 Rank 0 的完整工作)
    if (rank == 0) {
        int curr_num_since_last_hash = 0;
        auto start = system_clock::now();
        // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
        long long history_total = 0;
        std::ofstream a("./output/results.txt");
        
        while (!q.priority.empty())
        {
            q.guesses.clear(); // 清空上一轮的猜测
            
            q.PopNext(); // 包含并行的Generate

            // 使用 q.guesses.size() 作为本轮新生成的数量
            curr_num_since_last_hash = q.guesses.size();

            if (curr_num_since_last_hash > 0) {
                history_total += curr_num_since_last_hash;
                cout << "Guesses generated: " << history_total << " (+" << curr_num_since_last_hash << ")" << endl;
            }

            // 在此处更改实验生成的猜测上限
            int generate_n=10000000;
            if (history_total > generate_n)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Total guesses limit reached." << endl;
                cout << "Guess generation and management time:" << time_guess - time_hash << " seconds"<< endl;
                cout << "Hash time:" << time_hash << "seconds"<<endl;
                cout << "Train time:" << time_train <<" seconds"<<endl;
                cout << "Total Throughput: " << history_total / time_guess << " guesses/sec" << endl;
                break;
            }

            // 为了避免内存超限，我们在q.guesses中口令达到一定数目时，将其中的所有口令取出并且进行哈希
            // 由于Generate的并行化，现在PopNext一次可能生成大量guesses，所以每次都处理
            if (!q.guesses.empty())
            {
                auto start_hash = system_clock::now();
                
                // --- SIMD 批量哈希 ---
                // TODO：对于SIMD实验，将这里替换成你的SIMD MD5函数
                vector<array<bit32, 4>> all_hashes;
                MD5Hash_NEON_Batch(q.guesses, all_hashes);

                // 在这里对哈希所需的总时长进行计算
                auto end_hash = system_clock::now();
                auto duration = duration_cast<microseconds>(end_hash - start_hash);
                time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
                
                // 以下注释部分用于输出猜测和哈希，但是由于自动测试系统不太能写文件，所以这里你可以改成cout
                /*
                for (size_t i = 0; i < q.guesses.size(); ++i) {
                    a << q.guesses[i] << "\t";
                    for (int i1 = 0; i1 < 4; i1 += 1)
                    {
                        // 注意：SIMD版本不需要做字节序转换，如果需要和标准MD5输出一致，需在此处转换
                        a << std::setw(8) << std::setfill('0') << hex << all_hashes[i][i1];
                    }
                    a << endl;
                }
                */
            }
        }
    }

    // --- MPI 终结 ---
    // MPI_Finalize();
    return 0;
}