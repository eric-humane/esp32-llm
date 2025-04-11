#ifndef MEMORY_UTILS_H
#define MEMORY_UTILS_H

#include <stdlib.h>

#define BYTE_ALIGNMENT 16 // 16-byte alignment for SIMD

/**
 * @brief Allocate aligned memory
 *
 * @param alignment Memory alignment in bytes (typically 16 for SIMD)
 * @param size Size of memory to allocate in bytes
 * @return void* Pointer to aligned memory or NULL on failure
 */
void *esp32_aligned_malloc(size_t alignment, size_t size);

/**
 * @brief Allocate aligned memory and initialize to zero
 *
 * @param alignment Memory alignment in bytes (typically 16 for SIMD)
 * @param num Number of elements
 * @param size Size of each element in bytes
 * @return void* Pointer to aligned zeroed memory or NULL on failure
 */
void *esp32_aligned_calloc(size_t alignment, size_t num, size_t size);

/**
 * @brief Free memory allocated with esp32_aligned_malloc
 *
 * @param ptr Pointer to memory to free
 */
void esp32_aligned_free(void *ptr);

/**
 * @brief Log current memory usage statistics
 *
 * Displays free heap, free PSRAM (if available), and minimum free heap
 *
 * @param tag Tag to identify the module in logs
 * @param message Message to describe when the stats are being printed
 * @param token_number Optional token number for LLM operations (-1 to ignore)
 */
void log_memory_usage(const char *tag, const char *message, int token_number);

#endif /* MEMORY_UTILS_H */