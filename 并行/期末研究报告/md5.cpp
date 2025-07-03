#include "md5.h"
#include <iomanip>
#include <assert.h>
#include <chrono>

using namespace std;
using namespace chrono;

/**
 * StringProcess: 将单个输入字符串转换成MD5计算所需的消息数组
 * @param input 输入
 * @param[out] n_byte 用于给调用者传递额外的返回值，即最终Byte数组的长度
 * @return Byte消息数组
 */
Byte *StringProcess(string input, int *n_byte)
{
	// 将输入的字符串转换为Byte为单位的数组
	const Byte *blocks = (const Byte *)input.c_str(); // Use const Byte*
	int length = input.length();

	// 计算原始消息长度（以比特为单位）
	uint64_t bitLength = (uint64_t)length * 8; // Use uint64_t for bit length

	// paddingBits: 原始消息需要的padding长度（以bit为单位）
	// 对于给定的消息，将其补齐至length%512==448为止
	// 需要注意的是，即便给定的消息满足length%512==448，也需要再pad 512bits
	int paddingBits = 512 - (bitLength % 512);
    if (paddingBits <= 64) {
        paddingBits += 512;
    }
    paddingBits -= 64;


	// 原始消息需要的padding长度（以Byte为单位）
	int paddingBytes = paddingBits / 8;
	// 创建最终的字节数组
	// length + paddingBytes + 8:
	// 1. length为原始消息的长度（bits）
	// 2. paddingBytes为原始消息需要的padding长度（Bytes）
	// 3. 在pad到length%512==448之后，需要额外附加64bits的原始消息长度，即8个bytes
	int paddedLength = length + paddingBytes + 8;
	Byte *paddedMessage = new Byte[paddedLength];

	// 复制原始消息
	memcpy(paddedMessage, blocks, length);

	// 添加填充字节。填充时，第一位为1，后面的所有位均为0。
	// 所以第一个byte是0x80
	paddedMessage[length] = 0x80;							 // 添加一个0x80字节
	memset(paddedMessage + length + 1, 0, paddingBytes - 1); // 填充0字节

	// 添加消息长度（64比特，小端格式）
	memcpy(paddedMessage + length + paddingBytes, &bitLength, 8);


	// 验证长度是否满足要求。此时长度应当是512bit的倍数
	int residual = 8 * paddedLength % 512;
	// assert(residual == 0);

	// 在填充+添加长度之后，消息被分为n_blocks个512bit的部分
	*n_byte = paddedLength;
	return paddedMessage;
}


/**
 * MD5Hash: 将单个输入字符串转换成MD5
 * @param input 输入
 * @param[out] state 用于给调用者传递额外的返回值，即最终的缓冲区，也就是MD5的结果
 * @return Byte消息数组
 */
void MD5Hash(string input, bit32 *state)
{

	Byte *paddedMessage;
	int messageLength; // No need for an array
	paddedMessage = StringProcess(input, &messageLength);
	
	int n_blocks = messageLength / 64;

	// bit32* state= new bit32[4];
	state[0] = 0x67452301;
	state[1] = 0xefcdab89;
	state[2] = 0x98badcfe;
	state[3] = 0x10325476;

	// 逐block地更新state
	for (int i = 0; i < n_blocks; i += 1)
	{
		bit32 x[16];
        bit32* block_start = (bit32*)(paddedMessage + i * 64);
		for (int i1 = 0; i1 < 16; ++i1)
		{
			x[i1] = block_start[i1];
		}

		bit32 a = state[0], b = state[1], c = state[2], d = state[3];

		/* Round 1 */
		FF(a, b, c, d, x[0], s11, 0xd76aa478);
		FF(d, a, b, c, x[1], s12, 0xe8c7b756);
		FF(c, d, a, b, x[2], s13, 0x242070db);
		FF(b, c, d, a, x[3], s14, 0xc1bdceee);
		FF(a, b, c, d, x[4], s11, 0xf57c0faf);
		FF(d, a, b, c, x[5], s12, 0x4787c62a);
		FF(c, d, a, b, x[6], s13, 0xa8304613);
		FF(b, c, d, a, x[7], s14, 0xfd469501);
		FF(a, b, c, d, x[8], s11, 0x698098d8);
		FF(d, a, b, c, x[9], s12, 0x8b44f7af);
		FF(c, d, a, b, x[10], s13, 0xffff5bb1);
		FF(b, c, d, a, x[11], s14, 0x895cd7be);
		FF(a, b, c, d, x[12], s11, 0x6b901122);
		FF(d, a, b, c, x[13], s12, 0xfd987193);
		FF(c, d, a, b, x[14], s13, 0xa679438e);
		FF(b, c, d, a, x[15], s14, 0x49b40821);

		/* Round 2 */
		GG(a, b, c, d, x[1], s21, 0xf61e2562);
		GG(d, a, b, c, x[6], s22, 0xc040b340);
		GG(c, d, a, b, x[11], s23, 0x265e5a51);
		GG(b, c, d, a, x[0], s24, 0xe9b6c7aa);
		GG(a, b, c, d, x[5], s21, 0xd62f105d);
		GG(d, a, b, c, x[10], s22, 0x02441453);
		GG(c, d, a, b, x[15], s23, 0xd8a1e681);
		GG(b, c, d, a, x[4], s24, 0xe7d3fbc8);
		GG(a, b, c, d, x[9], s21, 0x21e1cde6);
		GG(d, a, b, c, x[14], s22, 0xc33707d6);
		GG(c, d, a, b, x[3], s23, 0xf4d50d87);
		GG(b, c, d, a, x[8], s24, 0x455a14ed);
		GG(a, b, c, d, x[13], s21, 0xa9e3e905);
		GG(d, a, b, c, x[2], s22, 0xfcefa3f8);
		GG(c, d, a, b, x[7], s23, 0x676f02d9);
		GG(b, c, d, a, x[12], s24, 0x8d2a4c8a);

		/* Round 3 */
		HH(a, b, c, d, x[5], s31, 0xfffa3942);
		HH(d, a, b, c, x[8], s32, 0x8771f681);
		HH(c, d, a, b, x[11], s33, 0x6d9d6122);
		HH(b, c, d, a, x[14], s34, 0xfde5380c);
		HH(a, b, c, d, x[1], s31, 0xa4beea44);
		HH(d, a, b, c, x[4], s32, 0x4bdecfa9);
		HH(c, d, a, b, x[7], s33, 0xf6bb4b60);
		HH(b, c, d, a, x[10], s34, 0xbebfbc70);
		HH(a, b, c, d, x[13], s31, 0x289b7ec6);
		HH(d, a, b, c, x[0], s32, 0xeaa127fa);
		HH(c, d, a, b, x[3], s33, 0xd4ef3085);
		HH(b, c, d, a, x[6], s34, 0x04881d05);
		HH(a, b, c, d, x[9], s31, 0xd9d4d039);
		HH(d, a, b, c, x[12], s32, 0xe6db99e5);
		HH(c, d, a, b, x[15], s33, 0x1fa27cf8);
		HH(b, c, d, a, x[2], s34, 0xc4ac5665);

		/* Round 4 */
		II(a, b, c, d, x[0], s41, 0xf4292244);
		II(d, a, b, c, x[7], s42, 0x432aff97);
		II(c, d, a, b, x[14], s43, 0xab9423a7);
		II(b, c, d, a, x[5], s44, 0xfc93a039);
		II(a, b, c, d, x[12], s41, 0x655b59c3);
		II(d, a, b, c, x[3], s42, 0x8f0ccc92);
		II(c, d, a, b, x[10], s43, 0xffeff47d);
		II(b, c, d, a, x[1], s44, 0x85845dd1);
		II(a, b, c, d, x[8], s41, 0x6fa87e4f);
		II(d, a, b, c, x[15], s42, 0xfe2ce6e0);
		II(c, d, a, b, x[6], s43, 0xa3014314);
		II(b, c, d, a, x[13], s44, 0x4e0811a1);
		II(a, b, c, d, x[4], s41, 0xf7537e82);
		II(d, a, b, c, x[11], s42, 0xbd3af235);
		II(c, d, a, b, x[2], s43, 0x2ad7d2bb);
		II(b, c, d, a, x[9], s44, 0xeb86d391);

		state[0] += a;
		state[1] += b;
		state[2] += c;
		state[3] += d;
	}

	// 释放动态分配的内存
	delete[] paddedMessage;
}

// ** 新增：基于 NEON SIMD 的批量 MD5 哈希函数 **
void MD5Hash_NEON_Batch(const vector<string>& inputs, vector<array<bit32, 4>>& hashes) {
    size_t n = inputs.size();
    hashes.resize(n);

    const size_t batch_size = 4;
    size_t num_batches = n / batch_size;

    for (size_t i = 0; i < num_batches; ++i) {
        Byte* padded_messages[batch_size];
        int message_lengths[batch_size];
        int max_len = 0;

        // 1. 对批次内的4个字符串进行填充
        for (size_t j = 0; j < batch_size; ++j) {
            padded_messages[j] = StringProcess(inputs[i * batch_size + j], &message_lengths[j]);
            if (message_lengths[j] > max_len) {
                max_len = message_lengths[j];
            }
        }
        
        // 确保批次内所有填充后消息长度一致 (对于MD5标准算法来说，长度可能不同)
        int n_blocks = message_lengths[0] / 64;

        // 初始化状态向量
        VEC_bit32 A = vdupq_n_u32(0x67452301);
        VEC_bit32 B = vdupq_n_u32(0xefcdab89);
        VEC_bit32 C = vdupq_n_u32(0x98badcfe);
        VEC_bit32 D = vdupq_n_u32(0x10325476);

        for (int blk = 0; blk < n_blocks; ++blk) {
            // 2. 数据交错 (Data Interleaving)
            // 将4个消息的同一个数据块(512bit)交错加载
            VEC_bit32 X[16];
            for (int k = 0; k < 16; ++k) {
                uint32_t temp[4];
                for(int m=0; m<4; ++m) {
                    temp[m] = ((uint32_t*)(padded_messages[m] + blk * 64))[k];
                }
                X[k] = vld1q_u32(temp);
            }

            VEC_bit32 AA = A, BB = B, CC = C, DD = D;

            // 3. SIMD 并行计算
            /* Round 1 */
            AA = FF_neon(AA, BB, CC, DD, X[0],  s11, 0xd76aa478);
            DD = FF_neon(DD, AA, BB, CC, X[1],  s12, 0xe8c7b756);
            CC = FF_neon(CC, DD, AA, BB, X[2],  s13, 0x242070db);
            BB = FF_neon(BB, CC, DD, AA, X[3],  s14, 0xc1bdceee);
            AA = FF_neon(AA, BB, CC, DD, X[4],  s11, 0xf57c0faf);
            DD = FF_neon(DD, AA, BB, CC, X[5],  s12, 0x4787c62a);
            CC = FF_neon(CC, DD, AA, BB, X[6],  s13, 0xa8304613);
            BB = FF_neon(BB, CC, DD, AA, X[7],  s14, 0xfd469501);
            AA = FF_neon(AA, BB, CC, DD, X[8],  s11, 0x698098d8);
            DD = FF_neon(DD, AA, BB, CC, X[9],  s12, 0x8b44f7af);
            CC = FF_neon(CC, DD, AA, BB, X[10], s13, 0xffff5bb1);
            BB = FF_neon(BB, CC, DD, AA, X[11], s14, 0x895cd7be);
            AA = FF_neon(AA, BB, CC, DD, X[12], s11, 0x6b901122);
            DD = FF_neon(DD, AA, BB, CC, X[13], s12, 0xfd987193);
            CC = FF_neon(CC, DD, AA, BB, X[14], s13, 0xa679438e);
            BB = FF_neon(BB, CC, DD, AA, X[15], s14, 0x49b40821);

            /* Round 2 */
            AA = GG_neon(AA, BB, CC, DD, X[1],  s21, 0xf61e2562);
            DD = GG_neon(DD, AA, BB, CC, X[6],  s22, 0xc040b340);
            CC = GG_neon(CC, DD, AA, BB, X[11], s23, 0x265e5a51);
            BB = GG_neon(BB, CC, DD, AA, X[0],  s24, 0xe9b6c7aa);
            AA = GG_neon(AA, BB, CC, DD, X[5],  s21, 0xd62f105d);
            DD = GG_neon(DD, AA, BB, CC, X[10], s22, 0x02441453);
            CC = GG_neon(CC, DD, AA, BB, X[15], s23, 0xd8a1e681);
            BB = GG_neon(BB, CC, DD, AA, X[4],  s24, 0xe7d3fbc8);
            AA = GG_neon(AA, BB, CC, DD, X[9],  s21, 0x21e1cde6);
            DD = GG_neon(DD, AA, BB, CC, X[14], s22, 0xc33707d6);
            CC = GG_neon(CC, DD, AA, BB, X[3],  s23, 0xf4d50d87);
            BB = GG_neon(BB, CC, DD, AA, X[8],  s24, 0x455a14ed);
            AA = GG_neon(AA, BB, CC, DD, X[13], s21, 0xa9e3e905);
            DD = GG_neon(DD, AA, BB, CC, X[2],  s22, 0xfcefa3f8);
            CC = GG_neon(CC, DD, AA, BB, X[7],  s23, 0x676f02d9);
            BB = GG_neon(BB, CC, DD, AA, X[12], s24, 0x8d2a4c8a);
            
            /* Round 3 */
            AA = HH_neon(AA, BB, CC, DD, X[5],  s31, 0xfffa3942);
            DD = HH_neon(DD, AA, BB, CC, X[8],  s32, 0x8771f681);
            CC = HH_neon(CC, DD, AA, BB, X[11], s33, 0x6d9d6122);
            BB = HH_neon(BB, CC, DD, AA, X[14], s34, 0xfde5380c);
            AA = HH_neon(AA, BB, CC, DD, X[1],  s31, 0xa4beea44);
            DD = HH_neon(DD, AA, BB, CC, X[4],  s32, 0x4bdecfa9);
            CC = HH_neon(CC, DD, AA, BB, X[7],  s33, 0xf6bb4b60);
            BB = HH_neon(BB, CC, DD, AA, X[10], s34, 0xbebfbc70);
            AA = HH_neon(AA, BB, CC, DD, X[13], s31, 0x289b7ec6);
            DD = HH_neon(DD, AA, BB, CC, X[0],  s32, 0xeaa127fa);
            CC = HH_neon(CC, DD, AA, BB, X[3],  s33, 0xd4ef3085);
            BB = HH_neon(BB, CC, DD, AA, X[6],  s34, 0x04881d05);
            AA = HH_neon(AA, BB, CC, DD, X[9],  s31, 0xd9d4d039);
            DD = HH_neon(DD, AA, BB, CC, X[12], s32, 0xe6db99e5);
            CC = HH_neon(CC, DD, AA, BB, X[15], s33, 0x1fa27cf8);
            BB = HH_neon(BB, CC, DD, AA, X[2],  s34, 0xc4ac5665);

            /* Round 4 */
            AA = II_neon(AA, BB, CC, DD, X[0],  s41, 0xf4292244);
            DD = II_neon(DD, AA, BB, CC, X[7],  s42, 0x432aff97);
            CC = II_neon(CC, DD, AA, BB, X[14], s43, 0xab9423a7);
            BB = II_neon(BB, CC, DD, AA, X[5],  s44, 0xfc93a039);
            AA = II_neon(AA, BB, CC, DD, X[12], s41, 0x655b59c3);
            DD = II_neon(DD, AA, BB, CC, X[3],  s42, 0x8f0ccc92);
            CC = II_neon(CC, DD, AA, BB, X[10], s43, 0xffeff47d);
            BB = II_neon(BB, CC, DD, AA, X[1],  s44, 0x85845dd1);
            AA = II_neon(AA, BB, CC, DD, X[8],  s41, 0x6fa87e4f);
            DD = II_neon(DD, AA, BB, CC, X[15], s42, 0xfe2ce6e0);
            CC = II_neon(CC, DD, AA, BB, X[6],  s43, 0xa3014314);
            BB = II_neon(BB, CC, DD, AA, X[13], s44, 0x4e0811a1);
            AA = II_neon(AA, BB, CC, DD, X[4],  s41, 0xf7537e82);
            DD = II_neon(DD, AA, BB, CC, X[11], s42, 0xbd3af235);
            CC = II_neon(CC, DD, AA, BB, X[2],  s43, 0x2ad7d2bb);
            BB = II_neon(BB, CC, DD, AA, X[9],  s44, 0xeb86d391);

            A = vaddq_u32(A, AA);
            B = vaddq_u32(B, BB);
            C = vaddq_u32(C, CC);
            D = vaddq_u32(D, DD);
        }
        
        // 4. 提取结果
        uint32_t res_A[4], res_B[4], res_C[4], res_D[4];
        vst1q_u32(res_A, A);
        vst1q_u32(res_B, B);
        vst1q_u32(res_C, C);
        vst1q_u32(res_D, D);

        for (size_t j = 0; j < batch_size; ++j) {
            hashes[i * batch_size + j] = {res_A[j], res_B[j], res_C[j], res_D[j]};
        }
        
        // 释放为批次内消息分配的内存
        for(size_t j = 0; j < batch_size; ++j) {
            delete[] padded_messages[j];
        }
    }
    
    // 5. 处理剩余的 (少于4个) 密码
    for (size_t i = num_batches * batch_size; i < n; ++i) {
        MD5Hash(inputs[i], hashes[i].data());
    }
}