CC = gcc
CFLAGS = -Wall -O2 -I.
LDFLAGS = -lm

# All test programs
TESTS = test_bitset test_csr test_bfs

# All targets
all: $(TESTS)

# Compile test_bitset
test_bitset: test_bitset.c bitset.c
	$(CC) $(CFLAGS) -o $@ $^

# Compile test_bfs
test_bfs: test_bfs.c bitset.c
	$(CC) $(CFLAGS) -o $@ $^

# Compile test_csr
test_csr: test_csr.c
	$(CC) $(CFLAGS) -o $@ $^



# Run all tests
run_tests: $(TESTS)
	@echo "\n=== Running all unit tests ===\n"
	@for test in $(TESTS); do \
	    echo "Running $$test..."; \
	    ./$$test; \
	    echo "\n-----------------------------\n"; \
	done

# Clean up compilation products
clean:
	rm -f $(TESTS) *.o

# Show help information
help:
	@echo "Available commands:"
	@echo "  make            - Compile all test programs"
	@echo "  make run_tests  - Compile and run all test programs"
	@echo "  make clean      - Clean up compilation products"
	@echo "  make help       - Show this help message"