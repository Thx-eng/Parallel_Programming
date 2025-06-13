#include "PCFG.h"
#include <mpi.h>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <vector>
#include <queue>

using namespace std;

const int TAG_PASSWORD = 1;
const int TAG_TERMINATE = 2;
const int BATCH_SIZE = 1000;  // 每批发送的密码数量

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size < 2) {
        if (rank == 0) {
            cerr << "需要至少2个进程：1个生产者和至少1个消费者" << endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;
    
    if (rank == 0) {
        // 进程0作为生产者（口令猜测）
        PriorityQueue q;
        
        double start_train = MPI_Wtime();
        q.m.train("/guessdata/Rockyou-singleLined-full.txt");
        q.m.order();
        double end_train = MPI_Wtime();
        time_train = end_train - start_train;
        
        q.init();
        cout << "here" << endl;
        
        double start = MPI_Wtime();
        int total_generated = 0;
        int curr_num = 0;
        int history = 0;
        int generate_n = 10000000;
        vector<string> batch;
        batch.reserve(BATCH_SIZE);
        
        // 用于轮询分配任务给消费者
        int next_worker = 1;
        
        while (!q.priority.empty() && history + total_generated < generate_n) {
            q.PopNext();
            
            // 将新生成的密码加入批次
            for (const string& pw : q.guesses) {
                batch.push_back(pw);
                total_generated++;
                
                // 当批次满了时，发送给消费者
                if (batch.size() >= BATCH_SIZE) {
                    // 发送批次大小
                    int batch_size = batch.size();
                    MPI_Send(&batch_size, 1, MPI_INT, next_worker, TAG_PASSWORD, MPI_COMM_WORLD);
                    
                    // 发送密码数据
                    for (const string& password : batch) {
                        int len = password.length();
                        MPI_Send(&len, 1, MPI_INT, next_worker, TAG_PASSWORD, MPI_COMM_WORLD);
                        MPI_Send(password.c_str(), len, MPI_CHAR, next_worker, TAG_PASSWORD, MPI_COMM_WORLD);
                    }
                    
                    batch.clear();
                    
                    // 轮询下一个工作进程
                    next_worker = (next_worker % (size - 1)) + 1;
                }
            }
            
            // 检查是否需要输出进度
            if (total_generated - curr_num >= 100000) {
                cout << "Guesses generated: " << history + total_generated << endl;
                curr_num = total_generated;
            }
            
            // 检查是否需要重置（模拟原版的清理逻辑）
            if (total_generated > 1000000) {
                history += total_generated;
                total_generated = 0;
                curr_num = 0;
            }
            
            q.guesses.clear();
        }
        
        // 发送剩余的批次
        if (!batch.empty()) {
            int batch_size = batch.size();
            MPI_Send(&batch_size, 1, MPI_INT, next_worker, TAG_PASSWORD, MPI_COMM_WORLD);
            for (const string& password : batch) {
                int len = password.length();
                MPI_Send(&len, 1, MPI_INT, next_worker, TAG_PASSWORD, MPI_COMM_WORLD);
                MPI_Send(password.c_str(), len, MPI_CHAR, next_worker, TAG_PASSWORD, MPI_COMM_WORLD);
            }
        }
        
        // 向所有工作进程发送终止信号
        for (int i = 1; i < size; i++) {
            int terminate = -1;
            MPI_Send(&terminate, 1, MPI_INT, i, TAG_PASSWORD, MPI_COMM_WORLD);
        }
        
        double end = MPI_Wtime();
        time_guess = end - start;
        
        // 收集所有工作进程的哈希时间
        double global_time_hash = 0;
        for (int i = 1; i < size; i++) {
            double worker_hash_time;
            MPI_Recv(&worker_hash_time, 1, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            global_time_hash = max(global_time_hash, worker_hash_time);
        }
        
        // 输出与原版一致的格式
        cout << "Guess time:" << time_guess - global_time_hash << " seconds" << endl;
        cout << "Hash time:" << global_time_hash << " seconds" << endl;
        cout << "Train time:" << time_train << " seconds" << endl;
        
    } else {
        // 其他进程作为消费者（口令哈希）
        int total_hashed = 0;
        double hash_time = 0;
        
        while (true) {
            int batch_size;
            MPI_Status status;
            MPI_Recv(&batch_size, 1, MPI_INT, 0, TAG_PASSWORD, MPI_COMM_WORLD, &status);
            
            // 检查是否收到终止信号
            if (batch_size == -1) {
                break;
            }
            
            // 接收密码批次
            vector<string> passwords;
            for (int i = 0; i < batch_size; i++) {
                int len;
                MPI_Recv(&len, 1, MPI_INT, 0, TAG_PASSWORD, MPI_COMM_WORLD, &status);
                
                char* buffer = new char[len + 1];
                MPI_Recv(buffer, len, MPI_CHAR, 0, TAG_PASSWORD, MPI_COMM_WORLD, &status);
                buffer[len] = '\0';
                passwords.push_back(string(buffer));
                delete[] buffer;
            }
            
            // 执行哈希
            double start_hash = MPI_Wtime();
            bit32 state[4];
            for (const string& pw : passwords) {
                MD5Hash(pw, state);
                total_hashed++;
            }
            double end_hash = MPI_Wtime();
            hash_time += end_hash - start_hash;
        }
        
        // 发送哈希时间给主进程
        MPI_Send(&hash_time, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    return 0;
}