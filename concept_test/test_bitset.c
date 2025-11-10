#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bitset.h"

// Test bit set and clear operations
void test_bit_operations() {
    bitset_t bitset;
    bitset_init(&bitset);
    
    printf("Testing bit set and clear operations...\n");
    
    // Test single bit set
    BITSET_SET(&bitset, 42);
    if (BITSET_TEST(&bitset, 42)) {
        printf("✓ Bit 42 set successfully\n");
    } else {
        printf("✗ Failed to set bit 42\n");
    }
    
    // Test single bit clear
    BITSET_CLEAR(&bitset, 42);
    if (!BITSET_TEST(&bitset, 42)) {
        printf("✓ Bit 42 cleared successfully\n");
    } else {
        printf("✗ Failed to clear bit 42\n");
    }
    
    // Test multiple bits set and clear
    BITSET_SET(&bitset, 0);
    BITSET_SET(&bitset, 31);
    BITSET_SET(&bitset, 32);
    BITSET_SET(&bitset, 63);
    
    int pass = 1;
    if (!BITSET_TEST(&bitset, 0) || !BITSET_TEST(&bitset, 31) || 
        !BITSET_TEST(&bitset, 32) || !BITSET_TEST(&bitset, 63)) {
        pass = 0;
    }
    
    if (pass) {
        printf("✓ Multiple bits set successfully\n");
    } else {
        printf("✗ Failed to set multiple bits\n");
    }
    
    // Clear all bits
    bitset_clear_all(&bitset);
    if (!BITSET_TEST(&bitset, 0) && !BITSET_TEST(&bitset, 31) && 
        !BITSET_TEST(&bitset, 32) && !BITSET_TEST(&bitset, 63)) {
        printf("✓ All bits cleared successfully\n");
    } else {
        printf("✗ Failed to clear all bits\n");
    }
}

// Test bit count function
void test_bit_count() {
    bitset_t bitset;
    bitset_init(&bitset);
    
    printf("\nTesting bit count functionality...\n");
    
    // Set a specific number of bits
    for (int i = 0; i < 100; i++) {
        BITSET_SET(&bitset, i);
    }
    
    int count = bitset_count(&bitset);
    if (count == 100) {
        printf("✓ Bit count correct: expected 100, actual %d\n", count);
    } else {
        printf("✗ Bit count incorrect: expected 100, actual %d\n", count);
    }
}

// Test bitset empty check
void test_bitset_empty() {
    bitset_t bitset;
    bitset_init(&bitset);
    
    printf("\nTesting bitset empty check...\n");
    
    // Use bitset_count to check if empty
    if (bitset_count(&bitset) == 0) {
        printf("✓ New initialized bitset is empty\n");
    } else {
        printf("✗ New initialized bitset is not empty\n");
    }
    
    BITSET_SET(&bitset, 10);
    if (bitset_count(&bitset) > 0) {
        printf("✓ Bitset is not empty after setting bit\n");
    } else {
        printf("✗ Bitset is still empty after setting bit\n");
    }
}

// Test bitset copy functionality
void test_bitset_copy() {
    bitset_t bitset1, bitset2;
    bitset_init(&bitset1);
    bitset_init(&bitset2);
    
    printf("\nTesting bitset copy functionality...\n");
    
    // Set some bits in the first bitset
    for (int i = 0; i < 100; i += 2) {
        BITSET_SET(&bitset1, i);
    }
    
    // Copy to the second bitset
    BITSET_COPY(&bitset2, &bitset1);
    
    // Verify copy is correct
    int pass = 1;
    for (int i = 0; i < 100; i++) {
        int expected = (i % 2 == 0) ? 1 : 0;
        int actual = BITSET_TEST(&bitset2, i);
        if (expected != actual) {
            pass = 0;
            break;
        }
    }
    
    if (pass) {
        printf("✓ Bitset copied successfully\n");
    } else {
        printf("✗ Failed to copy bitset\n");
    }
}

int main() {
    printf("Starting bitset unit tests...\n\n");
    
    test_bit_operations();
    test_bit_count();
    test_bitset_empty();
    test_bitset_copy();
    
    printf("\nbitset unit tests completed!\n");
    return 0;
}