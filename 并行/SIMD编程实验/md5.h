#ifndef MD5_H
#define MD5_H

#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <cstdint> 

// +++ NEON: 包含 ARM NEON 头文件 +++
#include <arm_neon.h>
// +++ NEON: ------------------------ +++

// +++ 将 NEON_LANES 定义移到头文件 +++
#define NEON_LANES 4
// +++ ----------------------------- +++


// 定义了Byte，便于使用 (使用标准类型)
typedef uint8_t Byte;
// 定义了32比特 (使用标准类型)
typedef uint32_t bit32;
typedef uint64_t bit64; // 用于长度

// +++ NEON: 定义 NEON 使用的向量类型 +++
typedef uint32x4_t VEC_bit32; // 代表4个并行处理的32位无符号整数
// +++ NEON: -------------------------- +++

// MD5的一系列参数。参数是固定的
constexpr int s11 = 7;
constexpr int s12 = 12;
constexpr int s13 = 17;
constexpr int s14 = 22;
constexpr int s21 = 5;
constexpr int s22 = 9;
constexpr int s23 = 14;
constexpr int s24 = 20;
constexpr int s31 = 4;
constexpr int s32 = 11;
constexpr int s33 = 16;
constexpr int s34 = 23;
constexpr int s41 = 6;
constexpr int s42 = 10;
constexpr int s43 = 15;
constexpr int s44 = 21;

// --- 串行版本: 使用 static inline 函数代替宏 ---
// 基础 MD5 函数
static inline bit32 F_serial(bit32 x, bit32 y, bit32 z) {
    return (((x) & (y)) | ((~x) & (z)));
}

static inline bit32 G_serial(bit32 x, bit32 y, bit32 z) {
    return (((x) & (z)) | ((y) & (~z)));
}

static inline bit32 H_serial(bit32 x, bit32 y, bit32 z) {
    return ((x) ^ (y) ^ (z));
}

static inline bit32 I_serial(bit32 x, bit32 y, bit32 z) {
    return ((y) ^ ((x) | (~z)));
}

// 循环左移
static inline bit32 ROTATELEFT_serial(bit32 num, int n) {
     return (((num) << (n)) | ((num) >> (32-(n))));
}

// 轮函数
static inline void FF_serial(bit32 &a, bit32 b, bit32 c, bit32 d, bit32 x, int s, bit32 ac) {
    a += F_serial(b, c, d) + x + ac;
    a = ROTATELEFT_serial(a, s);
    a += b;
}

static inline void GG_serial(bit32 &a, bit32 b, bit32 c, bit32 d, bit32 x, int s, bit32 ac) {
    a += G_serial(b, c, d) + x + ac;
    a = ROTATELEFT_serial(a, s);
    a += b;
}

static inline void HH_serial(bit32 &a, bit32 b, bit32 c, bit32 d, bit32 x, int s, bit32 ac) {
    a += H_serial(b, c, d) + x + ac;
    a = ROTATELEFT_serial(a, s);
    a += b;
}

static inline void II_serial(bit32 &a, bit32 b, bit32 c, bit32 d, bit32 x, int s, bit32 ac) {
    a += I_serial(b, c, d) + x + ac;
    a = ROTATELEFT_serial(a, s);
    a += b;
}
// --- 串行函数结束 ---


// --- NEON 并行计算函数定义 ---
// +++ NEON: 并行的 F, G, H, I 函数 +++
inline VEC_bit32 F_neon(VEC_bit32 x, VEC_bit32 y, VEC_bit32 z) {
    // return vbslq_u32(x, y, z); // Bit Select: x ? y : z (equivalent to (x&y) | (~x&z))
     return vorrq_u32(vandq_u32(x, y), vbicq_u32(z, x)); // (x&y) | (z & ~x) - Original is correct
}

inline VEC_bit32 G_neon(VEC_bit32 x, VEC_bit32 y, VEC_bit32 z) {
    // return vbslq_u32(z, x, y); // Bit Select: z ? x : y (equivalent to (z&x) | (~z&y))
     return vorrq_u32(vandq_u32(x, z), vbicq_u32(y, z)); // (x&z) | (y & ~z) - Original is correct
}

inline VEC_bit32 H_neon(VEC_bit32 x, VEC_bit32 y, VEC_bit32 z) {
    return veorq_u32(veorq_u32(x, y), z); // x ^ y ^ z
}

inline VEC_bit32 I_neon(VEC_bit32 x, VEC_bit32 y, VEC_bit32 z) {
    return veorq_u32(y, vorrq_u32(x, vmvnq_u32(z))); // y ^ (x | ~z)
}
// +++ NEON: ------------------------ +++

// +++ NEON: 并行的 ROTATELEFT 函数 +++
template<int N>
inline VEC_bit32 ROTATELEFT_neon(VEC_bit32 num) {
    static_assert(N > 0 && N < 32, "Rotate amount must be between 1 and 31");
    return vorrq_u32(vshlq_n_u32(num, N), vshrq_n_u32(num, 32 - N));
}
// +++ NEON: -------------------------- +++

// +++ NEON: 并行的 FF, GG, HH, II 轮函数 +++
inline VEC_bit32 FF_neon(VEC_bit32 a, VEC_bit32 b, VEC_bit32 c, VEC_bit32 d, VEC_bit32 x, int s, uint32_t ac) {
    a = vaddq_u32(a, F_neon(b, c, d));
    a = vaddq_u32(a, x);
    a = vaddq_u32(a, vdupq_n_u32(ac)); 

    if (s == s11) a = ROTATELEFT_neon<s11>(a);
    else if (s == s12) a = ROTATELEFT_neon<s12>(a);
    else if (s == s13) a = ROTATELEFT_neon<s13>(a);
    else if (s == s14) a = ROTATELEFT_neon<s14>(a);
    a = vaddq_u32(a, b);
    return a;
}

inline VEC_bit32 GG_neon(VEC_bit32 a, VEC_bit32 b, VEC_bit32 c, VEC_bit32 d, VEC_bit32 x, int s, uint32_t ac) {
    a = vaddq_u32(a, G_neon(b, c, d));
    a = vaddq_u32(a, x);
    a = vaddq_u32(a, vdupq_n_u32(ac));
    if (s == s21) a = ROTATELEFT_neon<s21>(a);
    else if (s == s22) a = ROTATELEFT_neon<s22>(a);
    else if (s == s23) a = ROTATELEFT_neon<s23>(a);
    else if (s == s24) a = ROTATELEFT_neon<s24>(a);
    a = vaddq_u32(a, b);
    return a;
}

inline VEC_bit32 HH_neon(VEC_bit32 a, VEC_bit32 b, VEC_bit32 c, VEC_bit32 d, VEC_bit32 x, int s, uint32_t ac) {
    a = vaddq_u32(a, H_neon(b, c, d));
    a = vaddq_u32(a, x);
    a = vaddq_u32(a, vdupq_n_u32(ac));
    if (s == s31) a = ROTATELEFT_neon<s31>(a);
    else if (s == s32) a = ROTATELEFT_neon<s32>(a);
    else if (s == s33) a = ROTATELEFT_neon<s33>(a);
    else if (s == s34) a = ROTATELEFT_neon<s34>(a);
    a = vaddq_u32(a, b);
    return a;
}

inline VEC_bit32 II_neon(VEC_bit32 a, VEC_bit32 b, VEC_bit32 c, VEC_bit32 d, VEC_bit32 x, int s, uint32_t ac) {
    a = vaddq_u32(a, I_neon(b, c, d));
    a = vaddq_u32(a, x);
    a = vaddq_u32(a, vdupq_n_u32(ac));
    if (s == s41) a = ROTATELEFT_neon<s41>(a);
    else if (s == s42) a = ROTATELEFT_neon<s42>(a);
    else if (s == s43) a = ROTATELEFT_neon<s43>(a);
    else if (s == s44) a = ROTATELEFT_neon<s44>(a);
    a = vaddq_u32(a, b);
    return a;
}
// +++ NEON: ------------------------------ +++


// --- 函数声明 ---

/**
 * @brief [优化] 计算单个输入字符串的 MD5 哈希值 (串行)。
 *        使用传入的缓冲区进行填充，避免内部动态分配。
 * @param input 输入字符串。
 * @param state 输出 MD5 结果 (4 个 bit32)。
 * @param padded_buffer 用于填充消息的可重用缓冲区 
 */
void MD5Hash(const std::string& input, bit32 *state, std::vector<Byte>& padded_buffer);

/**
 * @brief [优化] 使用 NEON 指令并行计算一批密码的 MD5 哈希值。
 *        使用传入的缓冲区进行数据交错，避免内部动态分配。
 *
 * @param passwords 输入的密码字符串向量。
 * @param count 要处理的密码数量 (必须 <= passwords.size() 且 <= NEON_LANES)。
 * @param output_hashes 指向输出缓冲区的指针，用于存储计算出的哈希值。
 *                      缓冲区大小必须至少为 count * 16 字节。
 * @param interleaved_buffer 用于存储交错数据的可重用缓冲区 
 */
void MD5Hash_NEON_Batch(
    const std::vector<std::string>& passwords,
    size_t count,
    uint8_t* output_hashes,
    std::vector<Byte>& interleaved_buffer // 添加缓冲区参数
);
// +++ NEON: ------------------------- +++


#endif // MD5_H