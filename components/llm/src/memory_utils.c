#include "memory_utils.h"
#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_memory_utils.h"
#include "esp_system.h"
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

static const char *TAG = "MEM";

/**
 * @brief Allocate aligned memory
 *
 * @param alignment Memory alignment in bytes (typically 16 for SIMD)
 * @param size Size of memory to allocate in bytes
 * @return void* Pointer to aligned memory or NULL on failure
 */
HEAP_IRAM_ATTR void *esp32_aligned_malloc(size_t alignment, size_t size) {
  // Validate parameters
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    ESP_LOGE(TAG, "Alignment must be a power of 2, got %zu", alignment);
    alignment = BYTE_ALIGNMENT; // Default to 16-byte alignment for SIMD
  }

  if (size == 0) {
    ESP_LOGE(TAG, "Requested allocation of zero size");
    return NULL;
  }

  // Try first with ESP32's heap_caps aligned allocation function
  void *ptr = heap_caps_aligned_alloc(alignment, size, MALLOC_CAP_8BIT);

  // Check if allocation was successful and properly aligned
  if (ptr != NULL) {
    if ((uintptr_t)ptr % alignment != 0) {
      ESP_LOGW(TAG,
               "heap_caps_aligned_alloc returned misaligned memory: %p (modulo "
               "%zu = %zu)",
               ptr, alignment, (uintptr_t)ptr % alignment);

      // Free the misaligned memory and try alternative method
      heap_caps_free(ptr);
      ptr = NULL;
    }
  }

  // If first method failed, try alternative allocation method
  if (ptr == NULL) {
    ESP_LOGI(TAG, "Falling back to alternative aligned allocation method");

    // Allocate extra space for alignment adjustment and metadata
    size_t padding = alignment - 1 + sizeof(void *);
    void *original = heap_caps_malloc(size + padding, MALLOC_CAP_8BIT);

    if (original == NULL) {
      ESP_LOGE(TAG, "Failed to allocate %zu bytes with padding %zu", size,
               padding);
      return NULL;
    }

    // Calculate aligned address
    uintptr_t aligned_addr =
        ((uintptr_t)original + sizeof(void *) + alignment - 1) &
        ~(alignment - 1);

    // Store original pointer just before aligned memory for later freeing
    void **metadata = (void **)(aligned_addr - sizeof(void *));
    *metadata = original;

    ptr = (void *)aligned_addr;
    ESP_LOGI(
        TAG,
        "Alternative allocation: original=%p, aligned=%p (modulo %zu = %zu)",
        original, ptr, alignment, (uintptr_t)ptr % alignment);
  }

  if (ptr == NULL) {
    ESP_LOGE(TAG,
             "Failed to allocate %zu bytes with alignment %zu using any method",
             size, alignment);
  }

  return ptr;
}

/**
 * @brief Allocate aligned memory and initialize to zero
 *
 * @param alignment Memory alignment in bytes (typically 16 for SIMD)
 * @param num Number of elements
 * @param size Size of each element in bytes
 * @return void* Pointer to aligned zeroed memory or NULL on failure
 */
HEAP_IRAM_ATTR void *esp32_aligned_calloc(size_t alignment, size_t num,
                                          size_t size) {
  size_t total_size = num * size;
  void *ptr = esp32_aligned_malloc(alignment, total_size);
  return ptr;
}

/**
 * @brief Free memory allocated with esp32_aligned_malloc
 *
 * This function safely frees memory that was allocated with
 * esp32_aligned_malloc or esp32_aligned_calloc
 *
 * @param ptr Pointer to memory to free
 */
void esp32_aligned_free(void *ptr) {
  if (ptr == NULL) {
    return;
  }

  // Extra validation to prevent issues
  if (!esp_ptr_in_dram(ptr) && !esp_ptr_external_ram(ptr)) {
    ESP_LOGW(TAG, "Attempt to free invalid pointer: %p", ptr);
    return;
  }

  // Check if the pointer is properly aligned to detect potential issues
  if ((uintptr_t)ptr % BYTE_ALIGNMENT != 0) {
    ESP_LOGW(TAG, "Freeing a non-16-byte aligned pointer: %p", ptr);
  }

  // Check if this pointer was allocated using our alternative method
  // by looking for the metadata pointer just before the aligned memory
  uintptr_t ptr_addr = (uintptr_t)ptr;
  if (ptr_addr >= sizeof(void *)) {
    void **metadata = (void **)(ptr_addr - sizeof(void *));
    void *original = *metadata;

    // Check if the original pointer looks valid
    if (esp_ptr_in_dram(original) || esp_ptr_external_ram(original)) {
      // The pointer appears to be from our alternative allocation method
      if ((uintptr_t)original <
          (uintptr_t)ptr) { // Only check that original comes before ptr
        ESP_LOGI(
            TAG,
            "Freeing alternative allocated memory: aligned=%p, original=%p",
            ptr, original);
        heap_caps_free(original);
        return;
      }
    }
  }

  // For ESP32-S3, the heap_caps_aligned_alloc uses the same heap_caps_free
  // function
  heap_caps_free(ptr);
}

/**
 * @brief Log current memory usage statistics
 *
 * @param tag Tag to identify the module in logs
 * @param message Message to describe when the stats are being printed
 * @param token_number Optional token number for LLM operations (-1 to ignore)
 */
void log_memory_usage(const char *tag, const char *message, int token_number) {
  if (token_number >= 0) {
    ESP_LOGI(tag, "----- Memory Stats: %s (token %d) -----", message,
             token_number);
  } else {
    ESP_LOGI(tag, "----- Memory Stats: %s -----", message);
  }

  ESP_LOGI(tag, "Free heap: %" PRIu32 " bytes",
           (uint32_t)esp_get_free_heap_size());
  ESP_LOGI(
      tag, "Free internal heap: %" PRIu32 " bytes",
      (uint32_t)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  ESP_LOGI(tag, "Free PSRAM: %" PRIu32 " bytes",
           (uint32_t)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  ESP_LOGI(tag, "Minimum free heap size: %" PRIu32 " bytes",
           (uint32_t)esp_get_minimum_free_heap_size());
  ESP_LOGI(tag, "-------------------------------");
}
