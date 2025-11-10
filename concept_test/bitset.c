#include "bitset.h"

// Initialize bit set
void bitset_init(bitset_t *bs) {
    memset(bs->words, 0, NUM_WORDS * sizeof(bitset_word_t));
}

// Set all bits in the bit set
void bitset_set_all(bitset_t *bs) {
    memset(bs->words, 0xFF, NUM_WORDS * sizeof(bitset_word_t));
}

// Clear all bits in the bit set
void bitset_clear_all(bitset_t *bs) {
    memset(bs->words, 0, NUM_WORDS * sizeof(bitset_word_t));
}

// Count the number of bits set to 1 in the bit set
size_t bitset_count(const bitset_t *bs) {
    size_t count = 0;
    
    // Process each word
    for (int i = 0; i < NUM_WORDS; i++) {
        bitset_word_t word = bs->words[i];
        
        // Use Brian Kernighan's algorithm to count set bits
        while (word) {
            word &= word - 1;
            count++;
        }
    }
    
    return count;
}

// Print bit set content (for debugging)
void bitset_print(const bitset_t *bs) {
    printf("Bitset content: ");
    
    // Print from highest to lowest bit
    for (int i = BITSET_SIZE - 1; i >= 0; i--) {
        printf("%d", BITSET_TEST(bs, i));
        
        // Add a space every 8 bits for readability
        if (i % 8 == 0) {
            printf(" ");
        }
    }
    
    printf("\n");
}