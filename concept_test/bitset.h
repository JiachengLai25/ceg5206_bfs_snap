#ifndef BITSET_H
#define BITSET_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

// =========================================================
// 宏定义和类型定义
// =========================================================

// 定义底层存储单元的类型和位数
typedef unsigned int bitset_word_t;
#define WORD_SIZE (sizeof(bitset_word_t) * CHAR_BIT)

// 假设我们希望实现一个固定大小 N 的 Bitset，例如 256 位
#define BITSET_SIZE 5120

// 计算所需的存储单元（word）数量
#define NUM_WORDS ((BITSET_SIZE + WORD_SIZE - 1) / WORD_SIZE)

// Bitset 结构体定义
typedef struct {
    bitset_word_t words[NUM_WORDS];
} bitset_t;

// =========================================================
// 核心位操作宏
// =========================================================

/**
 * @brief 设置 Bitset 中指定位置的位为 1
 * @param bs Bitset 结构体指针
 * @param bit_pos 要设置的位位置 (0 到 BITSET_SIZE - 1)
 */
#define BITSET_SET(bs, bit_pos) \
    do { \
        /* 1. 计算位位置所在的存储单元索引 */ \
        size_t word_idx = (bit_pos) / WORD_SIZE; \
        /* 2. 计算位在该存储单元内的偏移量 */ \
        size_t bit_offset = (bit_pos) % WORD_SIZE; \
        /* 3. 使用按位或 (|=) 和左移 (1 << offset) 来设置位 */ \
        if (word_idx < NUM_WORDS) { \
            (bs)->words[word_idx] |= ((bitset_word_t)1 << bit_offset); \
        } \
    } while (0)

/**
 * @brief 清除 Bitset 中指定位置的位为 0
 * @param bs Bitset 结构体指针
 * @param bit_pos 要清除的位位置
 */
#define BITSET_CLEAR(bs, bit_pos) \
    do { \
        size_t word_idx = (bit_pos) / WORD_SIZE; \
        size_t bit_offset = (bit_pos) % WORD_SIZE; \
        /* 1. 创建一个只有该位为 1 的掩码 */ \
        /* 2. 对掩码取反 (~) 使其只有该位为 0 */ \
        /* 3. 使用按位与 (&=) 来清除位 */ \
        if (word_idx < NUM_WORDS) { \
            (bs)->words[word_idx] &= ~((bitset_word_t)1 << bit_offset); \
        } \
    } while (0)

/**
 * @brief 测试 Bitset 中指定位置的位值
 * @param bs Bitset 结构体指针
 * @param bit_pos 要测试的位位置
 * @return 1 (位为 1) 或 0 (位为 0)
 */
#define BITSET_TEST(bs, bit_pos) \
    ( \
        ((bit_pos) < BITSET_SIZE) ? \
        ( \
            /* 1. 计算索引和偏移量 */ \
            ((bs)->words[(bit_pos) / WORD_SIZE] & ((bitset_word_t)1 << ((bit_pos) % WORD_SIZE))) \
            ? 1 \
            : 0 \
        ) \
        : 0 /* 越界返回 0 */ \
    )

#define BITSET_COPY(dest, src) \
    memcpy((dest)->words, (src)->words, NUM_WORDS * sizeof((dest)->words[0]))
// =========================================================
// 辅助函数声明
// =========================================================

void bitset_init(bitset_t *bs);
void bitset_set_all(bitset_t *bs);
void bitset_clear_all(bitset_t *bs);
size_t bitset_count(const bitset_t *bs);
void bitset_print(const bitset_t *bs);

#endif // BITSET_H