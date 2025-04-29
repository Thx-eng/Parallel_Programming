#include "md5.h"
#include <vector>
#include <algorithm>
#include <stdexcept> 
#include <iostream>  
#include <cstring>   

using namespace std;

// --- 串行 MD5 实现 (优化内存管理) ---

/**
 * @brief [优化] 计算单个输入字符串的 MD5 哈希值 (串行)。
 *        使用传入的缓冲区进行填充，避免内部动态分配。
 *        这个函数现在包含了原 StringProcess 的逻辑。
 * @param input 输入字符串。
 * @param state 输出 MD5 结果 (4 个 bit32)。
 * @param padded_buffer 用于填充消息的可重用缓冲区 
 */
void MD5Hash(const std::string& input, bit32 *state, std::vector<Byte>& padded_buffer)
{
    size_t length = input.length();

    // 1. 计算填充后的长度
    bit64 bitLength = (bit64)length * 8;
    size_t paddedLength_bits = (length * 8);
    paddedLength_bits += (512 - (paddedLength_bits % 512)); // 先补到 512 的倍数
    if (((paddedLength_bits - 448) % 512) != 0) { // 如果末尾没有至少 64 位留给长度
         paddedLength_bits += 512; // 再加一个块
    }
     // 将末尾 64 位替换为长度，确保总长是 512 位 (64 字节) 的倍数
     paddedLength_bits = (paddedLength_bits - 64 + 512) % 512 == 448 ? paddedLength_bits : paddedLength_bits + 512 - 64;

    // 修正：计算正确的填充后字节长度
    int paddingBits = (length * 8) % 512;
    if (paddingBits >= 448) {
        paddingBits = 512 - (paddingBits - 448);
    } else {
        paddingBits = 448 - paddingBits;
    }
    int paddingBytes = paddingBits / 8;
    size_t paddedLength = length + paddingBytes + 8; // 最终填充后的字节长度

    // 2. 准备缓冲区
    padded_buffer.resize(paddedLength); // 调整缓冲区大小
    Byte* paddedMessage = padded_buffer.data(); // 获取原始指针

    // 3. 填充数据
    // 复制原始消息
    memcpy(paddedMessage, input.c_str(), length);

    // 添加填充字节 (0x80 后跟 0x00)
    paddedMessage[length] = 0x80;
    if (paddingBytes > 0) { // 只有在需要填充0时才调用 memset
         memset(paddedMessage + length + 1, 0, paddingBytes - 1);
    }


    // 添加消息长度
    for (int i = 0; i < 8; ++i) {
        paddedMessage[length + paddingBytes + i] = (Byte)((bitLength >> (i * 8)) & 0xFF);
    }

    // 验证最终长度是否为 64 字节的倍数 
    if (paddedLength % 64 != 0) {
         cerr << "Error: Padded length (" << paddedLength << ") is not a multiple of 64 bytes for input: " << input << endl;
         // 设置错误状态或抛出异常
         state[0] = state[1] = state[2] = state[3] = 0xFFFFFFFF;
         return;
    }


    // --- MD5 计算核心  ---
    int n_blocks = paddedLength / 64;

    state[0] = 0x67452301;
    state[1] = 0xefcdab89;
    state[2] = 0x98badcfe;
    state[3] = 0x10325476;

    for (int i = 0; i < n_blocks; ++i)
    {
        bit32 x[16];
        const Byte* block_start = paddedMessage + i * 64; // 当前块的起始地址

        // [优化] 使用 memcpy 加载小端数据 
        for (int i1 = 0; i1 < 16; ++i1) {
            memcpy(&x[i1], block_start + (i1 * 4), sizeof(bit32));
        }
        
        bit32 a = state[0], b = state[1], c = state[2], d = state[3];

        /* Round 1 */
        FF_serial(a, b, c, d, x[ 0], s11, 0xd76aa478); FF_serial(d, a, b, c, x[ 1], s12, 0xe8c7b756);
		FF_serial(c, d, a, b, x[ 2], s13, 0x242070db); FF_serial(b, c, d, a, x[ 3], s14, 0xc1bdceee);
		FF_serial(a, b, c, d, x[ 4], s11, 0xf57c0faf); FF_serial(d, a, b, c, x[ 5], s12, 0x4787c62a);
		FF_serial(c, d, a, b, x[ 6], s13, 0xa8304613); FF_serial(b, c, d, a, x[ 7], s14, 0xfd469501);
		FF_serial(a, b, c, d, x[ 8], s11, 0x698098d8); FF_serial(d, a, b, c, x[ 9], s12, 0x8b44f7af);
		FF_serial(c, d, a, b, x[10], s13, 0xffff5bb1); FF_serial(b, c, d, a, x[11], s14, 0x895cd7be);
		FF_serial(a, b, c, d, x[12], s11, 0x6b901122); FF_serial(d, a, b, c, x[13], s12, 0xfd987193);
		FF_serial(c, d, a, b, x[14], s13, 0xa679438e); FF_serial(b, c, d, a, x[15], s14, 0x49b40821);
        /* Round 2 */
		GG_serial(a, b, c, d, x[ 1], s21, 0xf61e2562); GG_serial(d, a, b, c, x[ 6], s22, 0xc040b340);
		GG_serial(c, d, a, b, x[11], s23, 0x265e5a51); GG_serial(b, c, d, a, x[ 0], s24, 0xe9b6c7aa);
		GG_serial(a, b, c, d, x[ 5], s21, 0xd62f105d); GG_serial(d, a, b, c, x[10], s22, 0x02441453);
		GG_serial(c, d, a, b, x[15], s23, 0xd8a1e681); GG_serial(b, c, d, a, x[ 4], s24, 0xe7d3fbc8);
		GG_serial(a, b, c, d, x[ 9], s21, 0x21e1cde6); GG_serial(d, a, b, c, x[14], s22, 0xc33707d6);
		GG_serial(c, d, a, b, x[ 3], s23, 0xf4d50d87); GG_serial(b, c, d, a, x[ 8], s24, 0x455a14ed);
		GG_serial(a, b, c, d, x[13], s21, 0xa9e3e905); GG_serial(d, a, b, c, x[ 2], s22, 0xfcefa3f8);
		GG_serial(c, d, a, b, x[ 7], s23, 0x676f02d9); GG_serial(b, c, d, a, x[12], s24, 0x8d2a4c8a);
        /* Round 3 */
		HH_serial(a, b, c, d, x[ 5], s31, 0xfffa3942); HH_serial(d, a, b, c, x[ 8], s32, 0x8771f681);
		HH_serial(c, d, a, b, x[11], s33, 0x6d9d6122); HH_serial(b, c, d, a, x[14], s34, 0xfde5380c);
		HH_serial(a, b, c, d, x[ 1], s31, 0xa4beea44); HH_serial(d, a, b, c, x[ 4], s32, 0x4bdecfa9);
		HH_serial(c, d, a, b, x[ 7], s33, 0xf6bb4b60); HH_serial(b, c, d, a, x[10], s34, 0xbebfbc70);
		HH_serial(a, b, c, d, x[13], s31, 0x289b7ec6); HH_serial(d, a, b, c, x[ 0], s32, 0xeaa127fa);
		HH_serial(c, d, a, b, x[ 3], s33, 0xd4ef3085); HH_serial(b, c, d, a, x[ 6], s34, 0x04881d05);
		HH_serial(a, b, c, d, x[ 9], s31, 0xd9d4d039); HH_serial(d, a, b, c, x[12], s32, 0xe6db99e5);
		HH_serial(c, d, a, b, x[15], s33, 0x1fa27cf8); HH_serial(b, c, d, a, x[ 2], s34, 0xc4ac5665);
        /* Round 4 */
		II_serial(a, b, c, d, x[ 0], s41, 0xf4292244); II_serial(d, a, b, c, x[ 7], s42, 0x432aff97);
		II_serial(c, d, a, b, x[14], s43, 0xab9423a7); II_serial(b, c, d, a, x[ 5], s44, 0xfc93a039);
		II_serial(a, b, c, d, x[12], s41, 0x655b59c3); II_serial(d, a, b, c, x[ 3], s42, 0x8f0ccc92);
		II_serial(c, d, a, b, x[10], s43, 0xffeff47d); II_serial(b, c, d, a, x[ 1], s44, 0x85845dd1);
		II_serial(a, b, c, d, x[ 8], s41, 0x6fa87e4f); II_serial(d, a, b, c, x[15], s42, 0xfe2ce6e0);
		II_serial(c, d, a, b, x[ 6], s43, 0xa3014314); II_serial(b, c, d, a, x[13], s44, 0x4e0811a1);
		II_serial(a, b, c, d, x[ 4], s41, 0xf7537e82); II_serial(d, a, b, c, x[11], s42, 0xbd3af235);
		II_serial(c, d, a, b, x[ 2], s43, 0x2ad7d2bb); II_serial(b, c, d, a, x[ 9], s44, 0xeb86d391);

        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
    }
}


// --- NEON MD5 批处理实现 (优化内存管理) ---

// +++ NEON: 辅助函数 - 计算填充并将数据直接写入交错缓冲区 +++
inline void PadAndInterleave_NEON(const string& input, size_t lane_index, Byte* interleaved_buffer, size_t max_n_blocks) {
    size_t length = input.length();
    bit64 bitLength = (bit64)length * 8;

    // --- 重新计算填充逻辑以确保正确性 ---
    int paddingBits = (length * 8) % 512;
    if (paddingBits >= 448) {
        paddingBits = 512 - (paddingBits - 448);
    } else {
        paddingBits = 448 - paddingBits;
    }
    int paddingBytes = paddingBits / 8;
    int paddedLength = length + paddingBytes + 8; // 最终填充后的字节长度
    size_t current_n_blocks = paddedLength / 64; // 当前字符串实际需要的块数
    // --- 结束重新计算 ---


    uint8_t length_bytes[8]; // 存储 64 位长度
    for (int i = 0; i < 8; ++i) {
        length_bytes[i] = (bitLength >> (i * 8)) & 0xFF;
    }

    for (size_t block_idx = 0; block_idx < max_n_blocks; ++block_idx) {
        Byte* block_base_interleaved = interleaved_buffer + block_idx * NEON_LANES * 64; // 当前块在交错缓冲区的基址

        if (block_idx < current_n_blocks) {
            // 处理实际需要的块
            for (int word_idx = 0; word_idx < 16; ++word_idx) {
                // 目标写入地址: (块基址) + (字偏移*总通道数*字大小) + (通道偏移*字大小)
                Byte* dest_ptr = block_base_interleaved + word_idx * NEON_LANES * 4 + lane_index * 4;
                size_t current_byte_offset = block_idx * 64 + word_idx * 4; // 当前字在填充后消息中的字节偏移
                Byte source_bytes[4] = {0}; // 准备要写入的4字节 

                // 根据偏移量确定源数据
                for(int k=0; k<4; ++k) {
                    size_t byte_pos = current_byte_offset + k;
                    if (byte_pos < length) {
                        source_bytes[k] = input[byte_pos]; // 来自原始输入
                    } else if (byte_pos == length) {
                        source_bytes[k] = 0x80; // 填充的第一个字节
                    } else if (byte_pos < (size_t)(paddedLength - 8)) { // 显式转换以避免警告
                        source_bytes[k] = 0x00; // 填充的零字节
                    } else if (byte_pos < (size_t)paddedLength) {
                         source_bytes[k] = length_bytes[byte_pos - (paddedLength - 8)]; 
                    } else {
                        // 超出部分理论上不应访问，如果访问到说明 max_n_blocks 计算或逻辑有误
                        source_bytes[k] = 0xAA; // 使用特殊值标记错误，而不是0
                    }
                }
                memcpy(dest_ptr, source_bytes, 4); // 将准备好的4字节写入目标 
            }
        } else {
            // 如果当前块超出实际需要，用 0 填充该通道对应的所有字
            for (int word_idx = 0; word_idx < 16; ++word_idx) {
                Byte* dest_ptr = block_base_interleaved + word_idx * NEON_LANES * 4 + lane_index * 4;
                memset(dest_ptr, 0, 4); // 写入 4 个零字节
            }
        }
    }
}


/**
 * @brief [优化] 使用 NEON 指令并行计算一批密码的 MD5 哈希值。
 *        使用传入的缓冲区进行数据交错，避免内部动态分配。
 */
void MD5Hash_NEON_Batch(
    const std::vector<std::string>& passwords,
    size_t count,
    uint8_t* output_hashes,
    std::vector<Byte>& interleaved_buffer // 使用传入的缓冲区
)
{
    if (count == 0) return;
    if (count > passwords.size()) {
        cerr << "Error: count > passwords.size() in MD5Hash_NEON_Batch" << endl;
        return; // 或者抛出异常
    }
    if (count > NEON_LANES) {
        cerr << "Error: count > NEON_LANES in MD5Hash_NEON_Batch. Max supported: " << NEON_LANES << endl;
        return; // 当前实现仅支持最多 NEON_LANES 个密码
    }

    // 1. 确定批次中所有密码填充后的最大块数 
    size_t max_n_blocks = 0;
    for (size_t i = 0; i < count; ++i) {
        const string& input = passwords[i];
        size_t length = input.length();
        // --- 重新计算填充逻辑以确保正确性 ---
        int paddingBits = (length * 8) % 512;
         if (paddingBits >= 448) {
             paddingBits = 512 - (paddingBits - 448);
         } else {
             paddingBits = 448 - paddingBits;
         }
        int paddingBytes = paddingBits / 8;
        int paddedLength = length + paddingBytes + 8;
        // --- 结束重新计算 ---
        max_n_blocks = std::max(max_n_blocks, (size_t)paddedLength / 64);
    }
    // 如果 max_n_blocks 为 0 (例如输入都是空字符串)，至少需要一个块
    if (max_n_blocks == 0) max_n_blocks = 1;


    // 2. [优化] 检查并调整传入的交错缓冲区大小
    size_t required_size = NEON_LANES * max_n_blocks * 64;
    if (interleaved_buffer.size() < required_size) {
        interleaved_buffer.resize(required_size); // 调整大小
    }

    // 3. 并行填充并将数据直接写入交错缓冲区
    // 循环仍然是 NEON_LANES，以填充无效通道
    for (size_t i = 0; i < NEON_LANES; ++i) {
        if (i < count) {
            // 对有效的密码调用填充和交错函数
            PadAndInterleave_NEON(passwords[i], i, interleaved_buffer.data(), max_n_blocks);
        } else {
            // 对无效的通道 (如果 count < NEON_LANES)，用空字符串逻辑填充
            // PadAndInterleave_NEON 内部会处理空字符串的情况，并用0填充多余块
            PadAndInterleave_NEON("", i, interleaved_buffer.data(), max_n_blocks);
        }
    }


    // 4. 初始化 NEON 状态向量
    VEC_bit32 a_vec = vdupq_n_u32(0x67452301);
    VEC_bit32 b_vec = vdupq_n_u32(0xefcdab89);
    VEC_bit32 c_vec = vdupq_n_u32(0x98badcfe);
    VEC_bit32 d_vec = vdupq_n_u32(0x10325476);

    // 5. NEON 计算核心 - 逐块处理 
    for (size_t block_idx = 0; block_idx < max_n_blocks; ++block_idx) {
        VEC_bit32 x_vec[16];
        // 直接从交错缓冲区加载数据
        // 计算当前块在交错缓冲区的起始地址
        uint32_t* current_block_interleaved = (uint32_t*)(interleaved_buffer.data() + block_idx * NEON_LANES * 64);

        // 加载 16 个交错的 32 位字向量
        for (int i = 0; i < 16; ++i) {
            // vld1q_u32 从内存加载 4 个连续的 uint32_t 到一个向量寄存器
            // 地址计算: block_start + word_index * num_lanes
            x_vec[i] = vld1q_u32(current_block_interleaved + i * NEON_LANES);
        }

        VEC_bit32 aa = a_vec, bb = b_vec, cc = c_vec, dd = d_vec;

         /* Round 1 */
        a_vec = FF_neon(a_vec, b_vec, c_vec, d_vec, x_vec[ 0], s11, 0xd76aa478); d_vec = FF_neon(d_vec, a_vec, b_vec, c_vec, x_vec[ 1], s12, 0xe8c7b756);
        c_vec = FF_neon(c_vec, d_vec, a_vec, b_vec, x_vec[ 2], s13, 0x242070db); b_vec = FF_neon(b_vec, c_vec, d_vec, a_vec, x_vec[ 3], s14, 0xc1bdceee);
        a_vec = FF_neon(a_vec, b_vec, c_vec, d_vec, x_vec[ 4], s11, 0xf57c0faf); d_vec = FF_neon(d_vec, a_vec, b_vec, c_vec, x_vec[ 5], s12, 0x4787c62a);
        c_vec = FF_neon(c_vec, d_vec, a_vec, b_vec, x_vec[ 6], s13, 0xa8304613); b_vec = FF_neon(b_vec, c_vec, d_vec, a_vec, x_vec[ 7], s14, 0xfd469501);
        a_vec = FF_neon(a_vec, b_vec, c_vec, d_vec, x_vec[ 8], s11, 0x698098d8); d_vec = FF_neon(d_vec, a_vec, b_vec, c_vec, x_vec[ 9], s12, 0x8b44f7af);
        c_vec = FF_neon(c_vec, d_vec, a_vec, b_vec, x_vec[10], s13, 0xffff5bb1); b_vec = FF_neon(b_vec, c_vec, d_vec, a_vec, x_vec[11], s14, 0x895cd7be);
        a_vec = FF_neon(a_vec, b_vec, c_vec, d_vec, x_vec[12], s11, 0x6b901122); d_vec = FF_neon(d_vec, a_vec, b_vec, c_vec, x_vec[13], s12, 0xfd987193);
        c_vec = FF_neon(c_vec, d_vec, a_vec, b_vec, x_vec[14], s13, 0xa679438e); b_vec = FF_neon(b_vec, c_vec, d_vec, a_vec, x_vec[15], s14, 0x49b40821);
        /* Round 2 */
        a_vec = GG_neon(a_vec, b_vec, c_vec, d_vec, x_vec[ 1], s21, 0xf61e2562); d_vec = GG_neon(d_vec, a_vec, b_vec, c_vec, x_vec[ 6], s22, 0xc040b340);
        c_vec = GG_neon(c_vec, d_vec, a_vec, b_vec, x_vec[11], s23, 0x265e5a51); b_vec = GG_neon(b_vec, c_vec, d_vec, a_vec, x_vec[ 0], s24, 0xe9b6c7aa);
        a_vec = GG_neon(a_vec, b_vec, c_vec, d_vec, x_vec[ 5], s21, 0xd62f105d); d_vec = GG_neon(d_vec, a_vec, b_vec, c_vec, x_vec[10], s22, 0x02441453);
        c_vec = GG_neon(c_vec, d_vec, a_vec, b_vec, x_vec[15], s23, 0xd8a1e681); b_vec = GG_neon(b_vec, c_vec, d_vec, a_vec, x_vec[ 4], s24, 0xe7d3fbc8);
        a_vec = GG_neon(a_vec, b_vec, c_vec, d_vec, x_vec[ 9], s21, 0x21e1cde6); d_vec = GG_neon(d_vec, a_vec, b_vec, c_vec, x_vec[14], s22, 0xc33707d6);
        c_vec = GG_neon(c_vec, d_vec, a_vec, b_vec, x_vec[ 3], s23, 0xf4d50d87); b_vec = GG_neon(b_vec, c_vec, d_vec, a_vec, x_vec[ 8], s24, 0x455a14ed);
        a_vec = GG_neon(a_vec, b_vec, c_vec, d_vec, x_vec[13], s21, 0xa9e3e905); d_vec = GG_neon(d_vec, a_vec, b_vec, c_vec, x_vec[ 2], s22, 0xfcefa3f8);
        c_vec = GG_neon(c_vec, d_vec, a_vec, b_vec, x_vec[ 7], s23, 0x676f02d9); b_vec = GG_neon(b_vec, c_vec, d_vec, a_vec, x_vec[12], s24, 0x8d2a4c8a);
        /* Round 3 */
        a_vec = HH_neon(a_vec, b_vec, c_vec, d_vec, x_vec[ 5], s31, 0xfffa3942); d_vec = HH_neon(d_vec, a_vec, b_vec, c_vec, x_vec[ 8], s32, 0x8771f681);
        c_vec = HH_neon(c_vec, d_vec, a_vec, b_vec, x_vec[11], s33, 0x6d9d6122); b_vec = HH_neon(b_vec, c_vec, d_vec, a_vec, x_vec[14], s34, 0xfde5380c);
        a_vec = HH_neon(a_vec, b_vec, c_vec, d_vec, x_vec[ 1], s31, 0xa4beea44); d_vec = HH_neon(d_vec, a_vec, b_vec, c_vec, x_vec[ 4], s32, 0x4bdecfa9);
        c_vec = HH_neon(c_vec, d_vec, a_vec, b_vec, x_vec[ 7], s33, 0xf6bb4b60); b_vec = HH_neon(b_vec, c_vec, d_vec, a_vec, x_vec[10], s34, 0xbebfbc70);
        a_vec = HH_neon(a_vec, b_vec, c_vec, d_vec, x_vec[13], s31, 0x289b7ec6); d_vec = HH_neon(d_vec, a_vec, b_vec, c_vec, x_vec[ 0], s32, 0xeaa127fa);
        c_vec = HH_neon(c_vec, d_vec, a_vec, b_vec, x_vec[ 3], s33, 0xd4ef3085); b_vec = HH_neon(b_vec, c_vec, d_vec, a_vec, x_vec[ 6], s34, 0x04881d05);
        a_vec = HH_neon(a_vec, b_vec, c_vec, d_vec, x_vec[ 9], s31, 0xd9d4d039); d_vec = HH_neon(d_vec, a_vec, b_vec, c_vec, x_vec[12], s32, 0xe6db99e5);
        c_vec = HH_neon(c_vec, d_vec, a_vec, b_vec, x_vec[15], s33, 0x1fa27cf8); b_vec = HH_neon(b_vec, c_vec, d_vec, a_vec, x_vec[ 2], s34, 0xc4ac5665);
        /* Round 4 */
        a_vec = II_neon(a_vec, b_vec, c_vec, d_vec, x_vec[ 0], s41, 0xf4292244); d_vec = II_neon(d_vec, a_vec, b_vec, c_vec, x_vec[ 7], s42, 0x432aff97);
        c_vec = II_neon(c_vec, d_vec, a_vec, b_vec, x_vec[14], s43, 0xab9423a7); b_vec = II_neon(b_vec, c_vec, d_vec, a_vec, x_vec[ 5], s44, 0xfc93a039);
        a_vec = II_neon(a_vec, b_vec, c_vec, d_vec, x_vec[12], s41, 0x655b59c3); d_vec = II_neon(d_vec, a_vec, b_vec, c_vec, x_vec[ 3], s42, 0x8f0ccc92);
        c_vec = II_neon(c_vec, d_vec, a_vec, b_vec, x_vec[10], s43, 0xffeff47d); b_vec = II_neon(b_vec, c_vec, d_vec, a_vec, x_vec[ 1], s44, 0x85845dd1);
        a_vec = II_neon(a_vec, b_vec, c_vec, d_vec, x_vec[ 8], s41, 0x6fa87e4f); d_vec = II_neon(d_vec, a_vec, b_vec, c_vec, x_vec[15], s42, 0xfe2ce6e0);
        c_vec = II_neon(c_vec, d_vec, a_vec, b_vec, x_vec[ 6], s43, 0xa3014314); b_vec = II_neon(b_vec, c_vec, d_vec, a_vec, x_vec[13], s44, 0x4e0811a1);
        a_vec = II_neon(a_vec, b_vec, c_vec, d_vec, x_vec[ 4], s41, 0xf7537e82); d_vec = II_neon(d_vec, a_vec, b_vec, c_vec, x_vec[11], s42, 0xbd3af235);
        c_vec = II_neon(c_vec, d_vec, a_vec, b_vec, x_vec[ 2], s43, 0x2ad7d2bb); b_vec = II_neon(b_vec, c_vec, d_vec, a_vec, x_vec[ 9], s44, 0xeb86d391);


        // 累加结果
        a_vec = vaddq_u32(a_vec, aa);
        b_vec = vaddq_u32(b_vec, bb);
        c_vec = vaddq_u32(c_vec, cc);
        d_vec = vaddq_u32(d_vec, dd);
    }

    // 6. 结果提取、字节序转换（如有必要）与存储
    // --- 使用 vrev32q_u8 进行字节翻转 ---
    typedef uint8x16_t VEC_Byte16; // 需要这个类型定义
    a_vec = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(a_vec)));
    b_vec = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(b_vec)));
    c_vec = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(c_vec)));
    d_vec = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(d_vec)));
    // --- 字节翻转结束 ---


    // 存储结果到临时数组 (结构：states[word_index][lane_index])
    // 这样 vst1q 可以直接写入
    alignas(16) uint32_t final_states_packed[4][NEON_LANES]; // 保证对齐
    vst1q_u32(final_states_packed[0], a_vec); // a0, a1, a2, a3
    vst1q_u32(final_states_packed[1], b_vec); // b0, b1, b2, b3
    vst1q_u32(final_states_packed[2], c_vec); // c0, c1, c2, c3
    vst1q_u32(final_states_packed[3], d_vec); // d0, d1, d2, d3

    // 复制需要的结果 (前 count 个) 到输出缓冲区
    // 输出格式: hash0_a, hash0_b, hash0_c, hash0_d, hash1_a, ...
    // 每个哈希是 16 字节
    for (size_t i = 0; i < count; ++i) {
        uint8_t* dest_hash = output_hashes + i * 16;
        // 从 final_states_packed 中提取第 i 个通道的数据
        memcpy(dest_hash + 0,  &final_states_packed[0][i], 4); // a_i (byte-swapped)
        memcpy(dest_hash + 4,  &final_states_packed[1][i], 4); // b_i (byte-swapped)
        memcpy(dest_hash + 8,  &final_states_packed[2][i], 4); // c_i (byte-swapped)
        memcpy(dest_hash + 12, &final_states_packed[3][i], 4); // d_i (byte-swapped)
    }
}
// +++ NEON: 实现结束 +++