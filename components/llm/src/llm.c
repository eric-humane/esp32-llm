/* Inference for Llama-2 Transformer model in pure C */
/**
 * Original author of this:
 * https://github.com/karpathy/llama2.c
 *
 * Slight modifications added to make it ESP32 friendly
 */

#include "llm.h"
#include "esp_attr.h"
#include "esp_dsp.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "memory_utils.h"
#include <ctype.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// Declare embedded binaries
extern const uint8_t tokenizer_bin_start[] asm("_binary_tokenizer_bin_start");
extern const uint8_t tokenizer_bin_end[] asm("_binary_tokenizer_bin_end");
extern const uint8_t model_bin_start[] asm("_binary_model_bin_start");
extern const uint8_t model_bin_end[] asm("_binary_model_bin_end");

static const char *TAG = "LLM";
static const int sizeof_v4sf = sizeof(v4sf);

#define MAP_FAILED NULL
#define munmap(ptr, length) custom_munmap(ptr)
#define close(fd) custom_close(fd)

// Global optimization configuration
OptimizationConfig g_opt_config = {
    .total_attention_time = 0,
    .total_matmul_time = 0,
    .total_ffn_time = 0,
    .total_forward_time = 0,
    .total_rmsnorm_time = 0,
    .total_residual_time = 0,
    .total_rope_time = 0,
    .total_misc_time = 0,
    .num_forward_calls = 0,
    .num_tokens_processed = 0,
    .perform_validation = true,
};

// Function prototypes
void custom_munmap(void *ptr);
int custom_close(int fd);
void softmax(v4sf *x, int size);
void rmsnorm(v4sf *o, v4sf *x, v4sf *weight, int size);
v4sf *forward(Transformer *transformer, int token, int pos);
void rope_llama(Config *p, RunState *s, int head_size, int pos,
                Transformer *transformer);

// Optimized fast_exp implementation that maintains good accuracy
static inline float fast_exp(float x) {
  // Early exit for common case
  if (x > 88.0f)
    return INFINITY;
  if (x < -88.0f)
    return 0.0f;

  // Constants for the fast approximation
  const float ln2 = 0.6931471805f;
  const float a = 8388608.0f / ln2;
  const float b = 1064866805.0f;

  // Use a union to avoid strict aliasing violations
  union {
    int i;
    float f;
  } u;

  // Scaling for better accuracy
  x *= ln2;
  u.i = (int)(a * x + b);

  // Return the float interpretation of the bit pattern
  return u.f;
}

// Implementation of new metrics utility functions
void llm_metrics_init(OptimizationConfig *config, int n_layers) {
  if (config == NULL)
    return;

  // Initialize the basic configuration
  config->tracing_enabled = true;
  config->tracing_level = LLM_TRACING_LEVEL;

  // Reset all existing counters
  config->total_attention_time = 0;
  config->total_matmul_time = 0;
  config->total_ffn_time = 0;
  config->total_forward_time = 0;
  config->total_rmsnorm_time = 0;
  config->total_residual_time = 0;
  config->total_rope_time = 0;
  config->total_misc_time = 0;
  config->num_forward_calls = 0;
  config->num_tokens_processed = 0;

  // Initialize statistical metrics
  config->min_token_time = LONG_MAX;
  config->max_token_time = 0;
  config->last_token_time = 0;

  // Allocate per-layer metrics arrays
  if (n_layers > 0) {
    config->layer_times = (long *)calloc(n_layers, sizeof(long));
    config->layer_attention_times = (long *)calloc(n_layers, sizeof(long));
    config->layer_ffn_times = (long *)calloc(n_layers, sizeof(long));
  } else {
    config->layer_times = NULL;
    config->layer_attention_times = NULL;
    config->layer_ffn_times = NULL;
  }

  // Allocate detailed operation metrics if needed
  if (config->tracing_level >= LLM_TRACING_DETAILED) {
    // Use a circular buffer for detailed operation metrics
    // This limits memory usage while still providing insights
    const int METRICS_BUFFER_SIZE = 1000;
    config->matmul_metrics_size = METRICS_BUFFER_SIZE;
    config->matmul_metrics_index = 0;
    config->matmul_count = (long *)calloc(METRICS_BUFFER_SIZE, sizeof(long));
    config->matmul_sizes = (long *)calloc(METRICS_BUFFER_SIZE, sizeof(long));
    config->matmul_times = (long *)calloc(METRICS_BUFFER_SIZE, sizeof(long));
  } else {
    config->matmul_count = NULL;
    config->matmul_sizes = NULL;
    config->matmul_times = NULL;
    config->matmul_metrics_size = 0;
  }
}

void llm_metrics_free(OptimizationConfig *config) {
  if (config == NULL)
    return;

  // Free all allocated arrays
  if (config->layer_times)
    free(config->layer_times);
  if (config->layer_attention_times)
    free(config->layer_attention_times);
  if (config->layer_ffn_times)
    free(config->layer_ffn_times);

  if (config->matmul_count)
    free(config->matmul_count);
  if (config->matmul_sizes)
    free(config->matmul_sizes);
  if (config->matmul_times)
    free(config->matmul_times);

  // Reset pointers
  config->layer_times = NULL;
  config->layer_attention_times = NULL;
  config->layer_ffn_times = NULL;
  config->matmul_count = NULL;
  config->matmul_sizes = NULL;
  config->matmul_times = NULL;
}

void llm_metrics_reset(OptimizationConfig *config) {
  if (config == NULL)
    return;

  // Reset all counters and timers
  config->total_attention_time = 0;
  config->total_matmul_time = 0;
  config->total_ffn_time = 0;
  config->total_forward_time = 0;
  config->total_rmsnorm_time = 0;
  config->total_residual_time = 0;
  config->total_rope_time = 0;
  config->total_misc_time = 0;
  config->num_forward_calls = 0;
  config->num_tokens_processed = 0;

  // Reset statistical metrics
  config->min_token_time = LONG_MAX;
  config->max_token_time = 0;
  config->last_token_time = 0;

  // Reset per-layer metrics if they exist
  int n_layers = 0;
  if (config->layer_times != NULL) {
    // Figure out how many layers we have by finding the first layer with zero
    // time
    while (config->layer_times[n_layers] > 0)
      n_layers++;
    if (n_layers == 0)
      n_layers = 32; // Reasonable default if we can't determine

    dsps_memset_aes3(config->layer_times, 0, n_layers * sizeof(long));
    dsps_memset_aes3(config->layer_attention_times, 0, n_layers * sizeof(long));
    dsps_memset_aes3(config->layer_ffn_times, 0, n_layers * sizeof(long));
  }

  // Reset detailed operation metrics
  if (config->matmul_count != NULL) {
    dsps_memset_aes3(config->matmul_count, 0,
                     config->matmul_metrics_size * sizeof(long));
    dsps_memset_aes3(config->matmul_sizes, 0,
                     config->matmul_metrics_size * sizeof(long));
    dsps_memset_aes3(config->matmul_times, 0,
                     config->matmul_metrics_size * sizeof(long));
    config->matmul_metrics_index = 0;
  }
}

void llm_metrics_set_tracing_level(OptimizationConfig *config, int level) {
  if (config == NULL)
    return;

  // If changing from a lower level to DETAILED, allocate resources
  if (level >= LLM_TRACING_DETAILED &&
      config->tracing_level < LLM_TRACING_DETAILED) {
    const int METRICS_BUFFER_SIZE = 1000;
    config->matmul_metrics_size = METRICS_BUFFER_SIZE;
    config->matmul_metrics_index = 0;
    config->matmul_count = (long *)calloc(METRICS_BUFFER_SIZE, sizeof(long));
    config->matmul_sizes = (long *)calloc(METRICS_BUFFER_SIZE, sizeof(long));
    config->matmul_times = (long *)calloc(METRICS_BUFFER_SIZE, sizeof(long));
  }
  // If changing from DETAILED to a lower level, free resources
  else if (level < LLM_TRACING_DETAILED &&
           config->tracing_level >= LLM_TRACING_DETAILED) {
    if (config->matmul_count)
      free(config->matmul_count);
    if (config->matmul_sizes)
      free(config->matmul_sizes);
    if (config->matmul_times)
      free(config->matmul_times);
    config->matmul_count = NULL;
    config->matmul_sizes = NULL;
    config->matmul_times = NULL;
    config->matmul_metrics_size = 0;
  }

  config->tracing_level = level;
}

// Print a summary of metrics (similar to current print_performance_metrics)
void llm_metrics_print_summary(OptimizationConfig *config) {
  if (config == NULL)
    return;

  ESP_LOGI(TAG, "=== Performance Metrics Summary ===");
  ESP_LOGI(TAG, "Forward calls: %d", config->num_forward_calls);
  ESP_LOGI(TAG, "Tokens processed: %d", config->num_tokens_processed);

  if (config->num_forward_calls > 0) {
    v4sf avg_forward_time = (v4sf)config->total_forward_time /
                            config->num_forward_calls /
                            1000.0f; // Convert to ms

    // Prevent division by zero
    if (avg_forward_time <= 0.001f) {
      avg_forward_time = 0.001f; // Prevent division by zero
    }

    v4sf avg_attention_time = (v4sf)config->total_attention_time /
                              config->num_forward_calls /
                              1000.0f; // Convert to ms
    v4sf avg_matmul_time = (v4sf)config->total_matmul_time /
                           config->num_forward_calls / 1000.0f; // Convert to ms
    v4sf avg_ffn_time = (v4sf)config->total_ffn_time /
                        config->num_forward_calls / 1000.0f; // Convert to ms
    v4sf avg_rmsnorm_time = (v4sf)config->total_rmsnorm_time /
                            config->num_forward_calls /
                            1000.0f; // Convert to ms
    v4sf avg_residual_time = (v4sf)config->total_residual_time /
                             config->num_forward_calls /
                             1000.0f; // Convert to ms
    v4sf avg_rope_time = (v4sf)config->total_rope_time /
                         config->num_forward_calls / 1000.0f; // Convert to ms
    v4sf avg_misc_time = (v4sf)config->total_misc_time /
                         config->num_forward_calls / 1000.0f; // Convert to ms

    ESP_LOGI(TAG, "Avg forward time: %.2f ms", avg_forward_time);
    ESP_LOGI(TAG, "  Attention: %.2f ms (%.1f%%)", avg_attention_time,
             100.0f * avg_attention_time / avg_forward_time);
    ESP_LOGI(TAG, "  MatMul: %.2f ms (%.1f%%)", avg_matmul_time,
             100.0f * avg_matmul_time / avg_forward_time);
    ESP_LOGI(TAG, "  FFN: %.2f ms (%.1f%%)", avg_ffn_time,
             100.0f * avg_ffn_time / avg_forward_time);
    ESP_LOGI(TAG, "  RMSNorm: %.2f ms (%.1f%%)", avg_rmsnorm_time,
             100.0f * avg_rmsnorm_time / avg_forward_time);
    ESP_LOGI(TAG, "  Residual: %.2f ms (%.1f%%)", avg_residual_time,
             100.0f * avg_residual_time / avg_forward_time);
    ESP_LOGI(TAG, "  RoPE: %.2f ms (%.1f%%)", avg_rope_time,
             100.0f * avg_rope_time / avg_forward_time);
    ESP_LOGI(TAG, "  Misc: %.2f ms (%.1f%%)", avg_misc_time,
             100.0f * avg_misc_time / avg_forward_time);
  }

  // Print token time statistics
  if (config->num_tokens_processed > 0) {
    ESP_LOGI(TAG, "Token time statistics:");
    ESP_LOGI(TAG, "  Min: %.2f ms", config->min_token_time / 1000.0f);
    ESP_LOGI(TAG, "  Max: %.2f ms", config->max_token_time / 1000.0f);
    ESP_LOGI(TAG, "  Last: %.2f ms", config->last_token_time / 1000.0f);
    ESP_LOGI(TAG, "  Avg: %.2f ms",
             config->total_forward_time / (float)config->num_tokens_processed /
                 1000.0f);
  }
}

// Print detailed metrics including per-layer analysis
void llm_metrics_print_detailed(OptimizationConfig *config) {
  if (config == NULL)
    return;

  // First print the summary
  llm_metrics_print_summary(config);

  // If we have per-layer metrics, print them
  if (config->layer_times != NULL) {
    ESP_LOGI(TAG, "\n=== Per-Layer Performance ===");

    // Count number of layers with non-zero time
    int n_layers = 0;
    while (config->layer_times[n_layers] > 0)
      n_layers++;

    // Calculate totals for percentage computation
    long total_layer_time = 0;
    for (int i = 0; i < n_layers; i++) {
      total_layer_time += config->layer_times[i];
    }

    // Print per-layer stats
    for (int i = 0; i < n_layers; i++) {
      float layer_ms = config->layer_times[i] / 1000.0f;
      float pct =
          (total_layer_time > 0)
              ? (100.0f * config->layer_times[i] / (float)total_layer_time)
              : 0.0f;

      ESP_LOGI(TAG, "Layer %2d: %.2f ms (%.1f%%)", i, layer_ms, pct);

      // Only print component breakdown if detailed tracing is enabled
      if (config->tracing_level >= LLM_TRACING_DETAILED) {
        float attn_ms = config->layer_attention_times[i] / 1000.0f;
        float ffn_ms = config->layer_ffn_times[i] / 1000.0f;
        float attn_pct = (config->layer_times[i] > 0)
                             ? (100.0f * config->layer_attention_times[i] /
                                (float)config->layer_times[i])
                             : 0.0f;
        float ffn_pct = (config->layer_times[i] > 0)
                            ? (100.0f * config->layer_ffn_times[i] /
                               (float)config->layer_times[i])
                            : 0.0f;

        ESP_LOGI(TAG, "  Attention: %.2f ms (%.1f%%)", attn_ms, attn_pct);
        ESP_LOGI(TAG, "  FFN: %.2f ms (%.1f%%)", ffn_ms, ffn_pct);
      }
    }
  }

  // Print matmul statistics if available and detailed tracing is enabled
  if (config->matmul_count != NULL &&
      config->tracing_level >= LLM_TRACING_DETAILED) {
    ESP_LOGI(TAG, "\n=== MatMul Operations ===");

    // Analyze matmul sizes and times
    long total_matmuls = 0;
    long total_size = 0;
    long min_time = LONG_MAX;
    long max_time = 0;

    for (int i = 0; i < config->matmul_metrics_size; i++) {
      if (config->matmul_count[i] > 0) {
        total_matmuls += config->matmul_count[i];
        total_size += config->matmul_sizes[i];
        if (config->matmul_times[i] < min_time)
          min_time = config->matmul_times[i];
        if (config->matmul_times[i] > max_time)
          max_time = config->matmul_times[i];
      }
    }

    ESP_LOGI(TAG, "Total MatMul operations: %ld", total_matmuls);
    ESP_LOGI(TAG, "Average size: %ld elements",
             total_matmuls > 0 ? total_size / total_matmuls : 0);
    ESP_LOGI(TAG, "Time range: %.3f - %.3f ms", min_time / 1000.0f,
             max_time / 1000.0f);

    // Show top 5 most expensive matmul operations
    ESP_LOGI(TAG, "Top 5 most expensive MatMul operations:");

    // Create temporary array for sorting
    typedef struct {
      long size;
      long time;
    } MatMulInfo;
    MatMulInfo infos[config->matmul_metrics_size];

    int valid_count = 0;
    for (int i = 0; i < config->matmul_metrics_size; i++) {
      if (config->matmul_count[i] > 0) {
        infos[valid_count].size = config->matmul_sizes[i];
        infos[valid_count].time = config->matmul_times[i];
        valid_count++;
      }
    }

    // Simple insertion sort for top 5
    for (int i = 0; i < valid_count; i++) {
      for (int j = i + 1; j < valid_count; j++) {
        if (infos[j].time > infos[i].time) {
          MatMulInfo temp = infos[i];
          infos[i] = infos[j];
          infos[j] = temp;
        }
      }
    }

    // Print top 5 or as many as we have
    int to_print = (valid_count < 5) ? valid_count : 5;
    for (int i = 0; i < to_print; i++) {
      ESP_LOGI(TAG, "  #%d: Size %ld, Time %.3f ms", i + 1, infos[i].size,
               infos[i].time / 1000.0f);
    }
  }
}

// Replace current reset_performance_metrics with new one
void reset_performance_metrics() { llm_metrics_reset(&g_opt_config); }

// Replace current print_performance_metrics with new one
void print_performance_metrics() {
  if (g_opt_config.tracing_level >= LLM_TRACING_DETAILED) {
    llm_metrics_print_detailed(&g_opt_config);
  } else {
    llm_metrics_print_summary(&g_opt_config);
  }
}

void swiglu_activation(v4sf *output, v4sf *input1, v4sf *input2, int size) {
  // Start timing for FFN using tracing macro
  LLM_TRACE_START(ffn);

  // Check alignment of inputs and outputs
  if ((uintptr_t)output % BYTE_ALIGNMENT != 0) {
    ESP_LOGW(TAG, "SwiGLU output not aligned: %p (modulo BYTE_ALIGNMENT = %u)",
             output, (unsigned int)((uintptr_t)output % BYTE_ALIGNMENT));
  }

  // Simple direct loop for all elements
  for (int i = 0; i < size; i++) {
    // Get current value
    float val = input1[i];

    // Calculate exp(-val)
    float exp_neg_val = fast_exp(-val);
    float sigmoid_val = 1.0f / (1.0f + exp_neg_val);

    // Final computation
    output[i] = val * sigmoid_val * input2[i];
  }

  // Update timing metrics using tracing macro
  LLM_TRACE_END(ffn, total_ffn_time);
}

bool malloc_run_state(RunState *s, Config *p) {
  // Use aligned memory allocation for all RunState buffers
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;

  // Calculate total memory needed
  size_t total_size = 0;
  size_t offsets[10]; // Store offsets for each buffer

// Calculate offsets and total size with alignment
#define ALIGN_UP(x, a) (((x) + ((a) - 1)) & ~((a) - 1))

  offsets[0] = 0;
  total_size += ALIGN_UP(p->dim * sizeof_v4sf, BYTE_ALIGNMENT); // x

  offsets[1] = total_size;
  total_size += ALIGN_UP(p->dim * sizeof_v4sf, BYTE_ALIGNMENT); // xb

  offsets[2] = total_size;
  total_size += ALIGN_UP(p->dim * sizeof_v4sf, BYTE_ALIGNMENT); // xb2

  offsets[3] = total_size;
  total_size += ALIGN_UP(p->hidden_dim * sizeof_v4sf, BYTE_ALIGNMENT); // hb

  offsets[4] = total_size;
  total_size += ALIGN_UP(p->hidden_dim * sizeof_v4sf, BYTE_ALIGNMENT); // hb2

  offsets[5] = total_size;
  total_size += ALIGN_UP(p->dim * sizeof_v4sf, BYTE_ALIGNMENT); // q

  offsets[6] = total_size;
  total_size += ALIGN_UP(p->n_layers * p->seq_len * kv_dim * sizeof_v4sf,
                         BYTE_ALIGNMENT); // key_cache

  offsets[7] = total_size;
  total_size += ALIGN_UP(p->n_layers * p->seq_len * kv_dim * sizeof_v4sf,
                         BYTE_ALIGNMENT); // value_cache

  offsets[8] = total_size;
  total_size +=
      ALIGN_UP(p->n_heads * p->seq_len * sizeof_v4sf, BYTE_ALIGNMENT); // att

  offsets[9] = total_size;
  total_size += ALIGN_UP(p->vocab_size * sizeof_v4sf, BYTE_ALIGNMENT); // logits

  // Single allocation
  uint8_t *memory_block =
      (uint8_t *)esp32_aligned_calloc(BYTE_ALIGNMENT, 1, total_size);
  if (!memory_block) {
    ESP_LOGE(TAG, "Failed to allocate memory block of size %zu", total_size);
    return false;
  }

  // Assign pointers using offsets
  s->x = (v4sf *)(memory_block + offsets[0]);
  s->xb = (v4sf *)(memory_block + offsets[1]);
  s->xb2 = (v4sf *)(memory_block + offsets[2]);
  s->hb = (v4sf *)(memory_block + offsets[3]);
  s->hb2 = (v4sf *)(memory_block + offsets[4]);
  s->q = (v4sf *)(memory_block + offsets[5]);
  s->key_cache = (v4sf *)(memory_block + offsets[6]);
  s->value_cache = (v4sf *)(memory_block + offsets[7]);
  s->att = (v4sf *)(memory_block + offsets[8]);
  s->logits = (v4sf *)(memory_block + offsets[9]);

  // Store the base pointer for cleanup
  s->memory_block = memory_block;

  // ensure all mallocs went fine
  if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q ||
      !s->key_cache || !s->value_cache || !s->att || !s->logits) {
    ESP_LOGE(TAG, "Malloc assignment failed");
    // Free the allocated memory block
    esp32_aligned_free(s->memory_block);
    s->memory_block = NULL;
    return false;
  }

  // Verify alignment of critical buffers
  if ((uintptr_t)s->logits % BYTE_ALIGNMENT != 0) {
    ESP_LOGW(TAG, "Logits buffer not properly aligned: %p", s->logits);
  } else {
    ESP_LOGI(TAG, "Logits buffer properly aligned: %p", s->logits);
  }

  return true;
}

void free_run_state(RunState *s) { esp32_aligned_free(s->memory_block); }

void memory_map_weights(TransformerWeights *w, Config *p, v4sf *ptr,
                        int shared_weights) {
  // Validate input parameters
  if (w == NULL || p == NULL || ptr == NULL) {
    ESP_LOGE(TAG, "NULL pointer passed to memory_map_weights");
    return;
  }

  int head_size = p->dim / p->n_heads;
  // make sure the calculations use appropriate sizing for ESP32 models
  uint64_t n_layers = p->n_layers;

  // Calculate expected memory locations for validation
  size_t expected_size = 0;
  expected_size += p->vocab_size * p->dim; // token_embedding_table
  expected_size += n_layers * p->dim;      // rms_att_weight
  expected_size += n_layers * p->dim * (p->n_heads * head_size);    // wq
  expected_size += n_layers * p->dim * (p->n_kv_heads * head_size); // wk
  expected_size += n_layers * p->dim * (p->n_kv_heads * head_size); // wv
  expected_size += n_layers * (p->n_heads * head_size) * p->dim;    // wo
  expected_size += n_layers * p->dim;                 // rms_ffn_weight
  expected_size += n_layers * p->dim * p->hidden_dim; // w1
  expected_size += n_layers * p->hidden_dim * p->dim; // w2
  expected_size += n_layers * p->dim * p->hidden_dim; // w3
  expected_size += p->dim;                            // rms_final_weight
  expected_size += p->seq_len * head_size; // freq_cis_real and freq_cis_imag
  if (!shared_weights) {
    expected_size += p->vocab_size * p->dim; // wcls (if not shared)
  }

  ESP_LOGI(TAG, "Expected model size: %zu floats", expected_size);

  w->token_embedding_table = ptr;
  ptr += p->vocab_size * p->dim;
  w->rms_att_weight = ptr;
  ptr += n_layers * p->dim;
  w->wq = ptr;
  ptr += n_layers * p->dim * (p->n_heads * head_size);
  w->wk = ptr;
  ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
  w->wv = ptr;
  ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
  w->wo = ptr;
  ptr += n_layers * (p->n_heads * head_size) * p->dim;
  w->rms_ffn_weight = ptr;
  ptr += n_layers * p->dim;
  w->w1 = ptr;
  ptr += n_layers * p->dim * p->hidden_dim;
  w->w2 = ptr;
  ptr += n_layers * p->hidden_dim * p->dim;
  w->w3 = ptr;
  ptr += n_layers * p->dim * p->hidden_dim;
  w->rms_final_weight = ptr;
  ptr += p->dim;
  ptr += p->seq_len * head_size /
         2; // skip what used to be freq_cis_real (for RoPE)
  ptr += p->seq_len * head_size /
         2; // skip what used to be freq_cis_imag (for RoPE)
  w->wcls = shared_weights ? w->token_embedding_table : ptr;

  // Check first few values of critical weights for sanity
  ESP_LOGI(TAG, "First token embedding value: %f", w->token_embedding_table[0]);
  ESP_LOGI(TAG, "First WQ value: %f", w->wq[0]);
  ESP_LOGI(TAG, "First WCLS value: %f", w->wcls[0]);
}

bool read_checkpoint(char *checkpoint, Config *config,
                     TransformerWeights *weights, int *fd, v4sf **data,
                     size_t *file_size) {
  // Check if we should use the embedded model binary
  if (checkpoint == NULL || strcmp(checkpoint, "__EMBEDDED__") == 0 ||
      strcmp(checkpoint, "/data/model.bin") == 0) {

    const uint8_t *embedded_data = model_bin_start;
    size_t model_size = model_bin_end - model_bin_start;

    if (embedded_data != NULL && model_size > 0) {
      ESP_LOGI(TAG, "Using embedded model data (%d bytes)", model_size);

      // Read in the config header from embedded data
      dsps_memcpy_aes3(config, embedded_data, sizeof(Config));

      // negative vocab size is hacky way of signaling unshared weights. bit
      // yikes.
      int shared_weights = config->vocab_size > 0 ? 1 : 0;
      config->vocab_size = abs(config->vocab_size);
      ESP_LOGI(TAG, "Vocab size is %ld", config->vocab_size);

      *file_size = model_size;
      ESP_LOGI(TAG, "Model size: %zu bytes", *file_size);

      // Calculate how much extra padding we need for alignment
      size_t config_size = sizeof(Config);
      size_t padding_needed =
          (BYTE_ALIGNMENT - (config_size % BYTE_ALIGNMENT)) %
          BYTE_ALIGNMENT; // Padding to make weight data 16-byte aligned
      size_t total_size = *file_size + padding_needed;

      // Use 16-byte aligned memory allocation for model data
      *data = (v4sf *)esp32_aligned_malloc(BYTE_ALIGNMENT, total_size);
      if (*data == NULL) {
        ESP_LOGE(TAG, "Malloc operation failed");
        return false;
      }

      // Verify alignment
      if ((uintptr_t)*data % BYTE_ALIGNMENT != 0) {
        ESP_LOGE(TAG,
                 "Model data not properly aligned: %p (modulo "
                 "BYTE_ALIGNMENT = %u)",
                 *data, (unsigned int)((uintptr_t)*data % BYTE_ALIGNMENT));
      } else {
        ESP_LOGI(TAG, "Model data properly aligned at: %p", *data);
      }

      // Copy the entire model data into memory
      dsps_memcpy_aes3(*data, embedded_data, *file_size);

      ESP_LOGI(TAG, "Successfully read embedded LLM into memory");

      // Declare weights_ptr variable
      v4sf *weights_ptr;

      // If we need padding, shift the weights data to ensure alignment
      if (padding_needed > 0) {
        // Calculate the start of weights data
        uint8_t *weight_start = (uint8_t *)*data + config_size;
        // Calculate the aligned destination for weights
        uint8_t *aligned_weight_start =
            (uint8_t *)*data + config_size + padding_needed;
        // Size of weight data
        size_t weight_size = *file_size - config_size;

        // Move weight data to the aligned position (must move backward to avoid
        // overlap)
        // memmove(aligned_weight_start, weight_start, weight_size);

        // Zero out the padding area
        dsps_memset_aes3(weight_start, 0, padding_needed);

        // Set the weights pointer to the aligned location
        weights_ptr = (v4sf *)aligned_weight_start;

        ESP_LOGI(TAG, "Applied %zu bytes of padding to align weights",
                 padding_needed);
      } else {
        // No padding needed, weights are already aligned
        weights_ptr = (v4sf *)((uint8_t *)*data + config_size);
      }

      // Check if weights_ptr is properly aligned
      if ((uintptr_t)weights_ptr % BYTE_ALIGNMENT != 0) {
        ESP_LOGW(TAG,
                 "Weight pointer not properly aligned: %p (modulo "
                 "BYTE_ALIGNMENT = %u)",
                 weights_ptr,
                 (unsigned int)((uintptr_t)weights_ptr % BYTE_ALIGNMENT));
      }

      memory_map_weights(weights, config, weights_ptr, shared_weights);
      return true;
    } else {
      ESP_LOGW(TAG, "Embedded model not found, falling back to file: %s",
               checkpoint);
    }
  }

  // Original file-based loading code
  FILE *file = fopen(checkpoint, "rb");
  if (!file) {
    ESP_LOGE(TAG, "Couldn't open file %s", checkpoint);
    return false;
  }

  // read in the config header
  if (fread(config, sizeof(Config), 1, file) != 1) {
    ESP_LOGE(TAG, "Failed to read config header");
    fclose(file);
    return false;
  }

  // negative vocab size is hacky way of signaling unshared weights. bit yikes.
  int shared_weights = config->vocab_size > 0 ? 1 : 0;
  config->vocab_size = abs(config->vocab_size);
  ESP_LOGI(TAG, "Vocab size is %ld", config->vocab_size);

  // figure out the file size
  fseek(file, 0, SEEK_END); // move file pointer to end of file
  *file_size = ftell(file); // get the file size, in bytes
  fseek(file, 0, SEEK_SET); // move back to beginning for reading
  ESP_LOGI(TAG, "File size: %zu bytes", *file_size);

  // Calculate how much extra padding we need for alignment
  size_t config_size = sizeof(Config);
  size_t padding_needed =
      (BYTE_ALIGNMENT - (config_size % BYTE_ALIGNMENT)) % BYTE_ALIGNMENT;
  size_t total_size = *file_size + padding_needed;

  // Use 16-byte aligned memory allocation for model data
  *data = (v4sf *)esp32_aligned_malloc(BYTE_ALIGNMENT, total_size);
  if (*data == NULL) {
    ESP_LOGE(TAG, "Malloc operation failed");
    fclose(file);
    return false;
  }

  // Verify alignment
  if ((uintptr_t)*data % BYTE_ALIGNMENT != 0) {
    ESP_LOGE(TAG,
             "Model data not properly aligned: %p (modulo "
             "BYTE_ALIGNMENT = %u)",
             *data, (unsigned int)((uintptr_t)*data % BYTE_ALIGNMENT));
  } else {
    ESP_LOGI(TAG, "Model data properly aligned at: %p", *data);
  }

  // Read the entire file into memory
  size_t bytes_read = fread(*data, 1, *file_size, file);
  if (bytes_read != *file_size) {
    ESP_LOGE(TAG, "Failed to read file into memory");
    ESP_LOGE(TAG, "Bytes read %zu bytes", bytes_read);
    fclose(file);
    esp32_aligned_free(*data);
    *data = NULL;
    return false;
  }
  fclose(file);

  ESP_LOGI(TAG, "Successfully read LLM into memory");

  // Declare weights_ptr variable
  v4sf *weights_ptr;

  // If we need padding, shift the weights data to ensure alignment
  if (padding_needed > 0) {
    // Calculate the start of weights data
    uint8_t *weight_start = (uint8_t *)*data + config_size;
    // Calculate the aligned destination for weights
    uint8_t *aligned_weight_start =
        (uint8_t *)*data + config_size + padding_needed;
    // Size of weight data
    size_t weight_size = *file_size - config_size;

    // Move weight data to the aligned position (must move backward to avoid
    // overlap)
    memmove(aligned_weight_start, weight_start, weight_size);

    // Zero out the padding area
    dsps_memset_aes3(weight_start, 0, padding_needed);

    // Set the weights pointer to the aligned location
    weights_ptr = (v4sf *)aligned_weight_start;

    ESP_LOGI(TAG, "Applied %zu bytes of padding to align weights",
             padding_needed);
  } else {
    // No padding needed, weights are already aligned
    weights_ptr = (v4sf *)((uint8_t *)*data + config_size);
  }

  // Check if weights_ptr is properly aligned
  if ((uintptr_t)weights_ptr % BYTE_ALIGNMENT != 0) {
    ESP_LOGW(
        TAG,
        "Weight pointer not properly aligned: %p (modulo BYTE_ALIGNMENT = %u)",
        weights_ptr, (unsigned int)((uintptr_t)weights_ptr % BYTE_ALIGNMENT));

    // Our new vector functions can handle unaligned data efficiently
    // No need to disable optimizations or perform realignment
    ESP_LOGI(TAG, "Using optimized vector functions with unaligned handling");
  }

  memory_map_weights(weights, config, weights_ptr, shared_weights);
  return true;
}

void rope_llama(Config *p, RunState *s, int head_size, int pos,
                Transformer *transformer) {
  // Start timing for RoPE
  long rope_start = esp_timer_get_time();

  // Bounds check for position
  if (pos < 0 || pos >= p->seq_len) {
    ESP_LOGW(TAG, "RoPE position %d out of bounds (0-%" PRId32 "), clamping",
             pos, p->seq_len - 1);
    pos = pos < 0 ? 0 : (pos >= p->seq_len ? p->seq_len - 1 : pos);
  }

  // Get the precomputed sine/cosine values for this position
  float *sin_cache = transformer->rope_sin_cache + pos * (head_size / 2);
  float *cos_cache = transformer->rope_cos_cache + pos * (head_size / 2);

  // Process heads sequentially
  for (int i = 0; i < p->n_heads; i++) {
    v4sf *q_ptr = &s->q[i * head_size];
    v4sf *k_ptr = NULL;
    if (i < p->n_kv_heads) {
      k_ptr = &s->k[i * head_size];
    }

    // Process all elements directly in pairs
    for (int j2 = 0; j2 < head_size; j2 += 2) {
      int j = j2 / 2;

      // Get query values
      float q0 = q_ptr[j2];
      float q1 = q_ptr[j2 + 1];

      // Apply rotation using pre-computed values
      float q0_cos = q0 * cos_cache[j];
      float q1_sin = q1 * sin_cache[j];
      float q0_new;
      dsps_sub_f32_ae32(&q0_cos, &q1_sin, &q0_new, 1, 1, 1, 1);

      float q0_sin = q0 * sin_cache[j];
      float q1_cos = q1 * cos_cache[j];
      float q1_new;
      dsps_add_f32_ae32(&q0_sin, &q1_cos, &q1_new, 1, 1, 1, 1);

      // Store results back
      q_ptr[j2] = q0_new;
      q_ptr[j2 + 1] = q1_new;

      // Apply rotation to key if this is a KV head
      if (k_ptr != NULL) {
        float k0 = k_ptr[j2];
        float k1 = k_ptr[j2 + 1];

        // Apply rotation using DSP functions
        float k0_cos = k0 * cos_cache[j];
        float k1_sin = k1 * sin_cache[j];
        float k0_new;
        dsps_sub_f32_ae32(&k0_cos, &k1_sin, &k0_new, 1, 1, 1, 1);

        float k0_sin = k0 * sin_cache[j];
        float k1_cos = k1 * cos_cache[j];
        float k1_new;
        dsps_add_f32_ae32(&k0_sin, &k1_cos, &k1_new, 1, 1, 1, 1);

        k_ptr[j2] = k0_new;
        k_ptr[j2 + 1] = k1_new;
      }
    }
  }

  // Update timing metrics
  g_opt_config.total_rope_time += (esp_timer_get_time() - rope_start);
}

// Replace the function signature
bool build_transformer(Transformer *t, char *checkpoint_path) {
  // read in the Config and the Weights from the checkpoint
  if (!read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd,
                       &t->data, &t->file_size)) {
    ESP_LOGE(TAG, "Failed to read checkpoint");
    return false;
  }

  // Initialize the metrics collection system with the number of layers
  llm_metrics_init(&g_opt_config, t->config.n_layers);

  // Validate transformer config
  if (t->config.vocab_size <= 0 || t->config.vocab_size > 65536) {
    ESP_LOGE(TAG, "Invalid vocab_size in config: %ld", t->config.vocab_size);
    t->config.vocab_size = 512; // Set to a reasonable default
  }

  if (t->config.dim <= 0 || t->config.dim > 8192) {
    ESP_LOGE(TAG, "Invalid dimension in config: %ld", t->config.dim);
    t->config.dim = 64; // Set to a reasonable default
  }

  if (t->config.n_layers <= 0 || t->config.n_layers > 128) {
    ESP_LOGE(TAG, "Invalid number of layers in config: %ld",
             t->config.n_layers);
    t->config.n_layers = 5; // Set to a reasonable default
  }

  // Memory layout debugging
  ESP_LOGI(TAG, "Token embedding table address: %p",
           t->weights.token_embedding_table);
  ESP_LOGI(TAG, "WQ address: %p", t->weights.wq);
  ESP_LOGI(TAG, "WK address: %p", t->weights.wk);
  ESP_LOGI(TAG, "WV address: %p", t->weights.wv);
  ESP_LOGI(TAG, "WO address: %p", t->weights.wo);
  ESP_LOGI(TAG, "WCLS address: %p", t->weights.wcls);

  // Check if any weights are NULL
  bool has_null_weights = false;
  if (t->weights.token_embedding_table == NULL) {
    ESP_LOGE(TAG, "Token embedding table is NULL");
    has_null_weights = true;
  }
  if (t->weights.wq == NULL || t->weights.wk == NULL || t->weights.wv == NULL ||
      t->weights.wo == NULL) {
    ESP_LOGE(TAG, "One or more attention weights are NULL");
    has_null_weights = true;
  }
  if (t->weights.wcls == NULL) {
    ESP_LOGE(TAG, "Classifier weights (wcls) is NULL");
    has_null_weights = true;
  }

  if (has_null_weights) {
    ESP_LOGE(TAG, "Critical model weights are NULL. Model loading failed.");
    return false;
  }

  // allocate the RunState buffers
  if (!malloc_run_state(&t->state, &t->config)) {
    ESP_LOGE(TAG, "Failed to allocate RunState memory");
    return false;
  }

  // Precompute RoPE sine and cosine tables
  int head_size = t->config.dim / t->config.n_heads;
  size_t rope_cache_size = t->config.seq_len * (head_size / 2);

  // Allocate memory for the RoPE cache with proper alignment
  t->rope_sin_cache = (float *)esp32_aligned_malloc(
      BYTE_ALIGNMENT, rope_cache_size * sizeof(float));
  t->rope_cos_cache = (float *)esp32_aligned_malloc(
      BYTE_ALIGNMENT, rope_cache_size * sizeof(float));

  if (!t->rope_sin_cache || !t->rope_cos_cache) {
    ESP_LOGE(TAG, "Failed to allocate memory for RoPE cache");
    // Clean up allocated resources
    if (t->rope_sin_cache) {
      esp32_aligned_free(t->rope_sin_cache);
      t->rope_sin_cache = NULL;
    }
    if (t->rope_cos_cache) {
      esp32_aligned_free(t->rope_cos_cache);
      t->rope_cos_cache = NULL;
    }
    free_run_state(&t->state);
    return false;
  }

  // Precompute the power values that don't depend on position
  float *inv_freq = (float *)malloc((head_size / 2) * sizeof(float));
  if (!inv_freq) {
    ESP_LOGE(TAG, "Failed to allocate memory for frequency table");
    // Clean up allocated resources
    esp32_aligned_free(t->rope_sin_cache);
    esp32_aligned_free(t->rope_cos_cache);
    t->rope_sin_cache = NULL;
    t->rope_cos_cache = NULL;
    free_run_state(&t->state);
    return false;
  }

#pragma GCC unroll 4
  for (int j = 0; j < head_size / 2; j++) {
    inv_freq[j] = 1.0f / powf(10000.0f, j * 2.0f / head_size);
  }

  // Initialize RoPE cache with precomputed values
  for (int pos = 0; pos < t->config.seq_len; pos++) {
    for (int j = 0; j < head_size / 2; j++) {
      float freq = pos * inv_freq[j]; // Multiply instead of divide
      int idx = pos * (head_size / 2) + j;
      sincosf(freq, &t->rope_sin_cache[idx], &t->rope_cos_cache[idx]);
    }
  }

  // Free temporary memory
  free(inv_freq);

  ESP_LOGI(TAG,
           "RoPE cache initialized for %" PRId32 " positions, head_size=%d",
           t->config.seq_len, head_size);

  ESP_LOGI(TAG, "Transformer successfully built");
  return true;
}

void free_transformer(Transformer *t) {
  // Free metrics collection resources
  llm_metrics_free(&g_opt_config);

  // close the memory mapping
  if (t->data != MAP_FAILED) {
    munmap(t->data, t->file_size);
  }
  if (t->fd != -1) {
    close(t->fd);
  }
  // free the RunState buffers
  free_run_state(&t->state);

  // Free RoPE cache memory
  if (t->rope_sin_cache) {
    esp32_aligned_free(t->rope_sin_cache);
    t->rope_sin_cache = NULL;
  }
  if (t->rope_cos_cache) {
    esp32_aligned_free(t->rope_cos_cache);
    t->rope_cos_cache = NULL;
  }
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void matmul_opt(v4sf *xout, const v4sf *x, const v4sf *w, int n, int d) {
  // Use the trace macro instead of direct timing
  LLM_TRACE_START(matmul);

  // Loop over rows in the weight matrix.
#pragma GCC unroll 2
  for (int ib = 0; ib < d; ib++) {
    v4sf dot = 0.0f;
    // The pointer arithmetic: Each row of w contains n floats.
    dsps_dotprod_f32_aes3(w + ib * n, x, &dot, n);
    (xout)[ib] = dot;
  }

  // End tracing and update metrics
  LLM_TRACE_END(matmul, total_matmul_time);

  // Record detailed matmul metrics if enabled
#if LLM_TRACING_LEVEL >= LLM_TRACING_DETAILED
  if (g_opt_config.tracing_enabled &&
      g_opt_config.matmul_metrics_index < g_opt_config.matmul_metrics_size) {
    int idx = g_opt_config.matmul_metrics_index++;
    if (idx < g_opt_config.matmul_metrics_size) {
      g_opt_config.matmul_count[idx]++;
      g_opt_config.matmul_sizes[idx] = (long)n * (long)d;
      g_opt_config.matmul_times[idx] = trace_end_matmul - trace_start_matmul;

      // Circular buffer implementation
      if (g_opt_config.matmul_metrics_index >=
          g_opt_config.matmul_metrics_size) {
        g_opt_config.matmul_metrics_index = 0;
      }
    }
  }
#endif
}

v4sf *forward(Transformer *transformer, int token, int pos) {
  // Start timing the forward pass
  LLM_TRACE_START(forward);

  // Validation
  if (transformer == NULL) {
    ESP_LOGE(TAG, "NULL transformer passed to forward");
    return NULL;
  }

  if (token < 0 || token >= transformer->config.vocab_size) {
    ESP_LOGW(TAG, "Invalid token %d (valid range: 0-%d), using token 0 instead",
             token, (int)(transformer->config.vocab_size) - 1);
    token = 0; // Use a safe default token
  }

  if (pos < 0 || pos >= transformer->config.seq_len) {
    ESP_LOGW(
        TAG,
        "Invalid position %d (valid range: 0-%d), using position 0 instead",
        pos, (int)(transformer->config.seq_len) - 1);
    pos = 0; // Use position 0 as a fallback
  }

  // Ensure RunState is valid
  if (transformer->state.x == NULL || transformer->state.logits == NULL) {
    ESP_LOGE(TAG, "Invalid RunState in transformer (x=%p, logits=%p)",
             transformer->state.x, transformer->state.logits);
    return NULL;
  }

  // Update metrics
  g_opt_config.num_forward_calls++;
  g_opt_config.num_tokens_processed++;

  // a few convenience variables
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;
  v4sf *x = s->x;
  int dim = p->dim;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int kv_mul =
      p->n_heads /
      p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
  int hidden_dim = p->hidden_dim;
  int head_size = dim / p->n_heads;

  // copy the token embedding into x
  if (w->token_embedding_table == NULL) {
    ESP_LOGE(TAG, "Token embedding table is NULL");
    return s->logits;
  }

  v4sf *content_row = w->token_embedding_table + token * dim;
  dsps_memcpy_aes3(x, content_row, dim * sizeof(*x));

  // forward all the layers
  for (uint64_t l = 0; l < p->n_layers; l++) {
    // Start timing for this layer
    LLM_TRACE_START(layer);

    // Use flash_rmsnorm instead of rmsnorm
    LLM_TRACE_START(rmsnorm1);
    rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);
    LLM_TRACE_END(rmsnorm1, total_rmsnorm_time);

    // key and value point to the kv cache
    int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
    s->k = s->key_cache + loff + pos * kv_dim;
    s->v = s->value_cache + loff + pos * kv_dim;

    // qkv matmuls for this position - could potentially fuse with rmsnorm in
    // future
    matmul_opt(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
    matmul_opt(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
    matmul_opt(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

    // RoPE relative positional encoding: complex-valued rotate q and k in each
    // head
    LLM_TRACE_START(rope);
    rope_llama(p, s, head_size, pos, transformer);
    LLM_TRACE_END(rope, total_rope_time);

    // Start timing for attention operations
    LLM_TRACE_START(attention);
    const v4sf inv_sqrt_head = 1.0f / dsps_sqrtf_f32_ansi(head_size);

    // multihead attention. iterate over all heads
#pragma GCC unroll 4
    for (int h = 0; h < p->n_heads; h++) {
      // get the query vector for this head
      v4sf *q = s->q + h * head_size;
      // attention scores for this head
      v4sf *att = s->att + h * p->seq_len;
      // iterate over all timesteps, including the current one
      for (int t = 0; t <= pos; t++) {
        // get the key vector for this head and at this timestep
        v4sf *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        // calculate the attention score as the dot product of q and k
        v4sf score = 0.0f;
        // Replace manual loop with optimized dot product
        dsps_dotprod_f32_aes3(q, k, &score, head_size);
        score = score * inv_sqrt_head;
        // save the score to the attention buffer
        att[t] = score;
      }

      // Use sparse softmax if we have the buffer and enough tokens
      softmax(att, pos + 1);

      // weighted sum of the values, store back into xb
      v4sf *xb = s->xb + h * head_size;
      dsps_memset_aes3(xb, 0, head_size * sizeof_v4sf);

// Standard attention - process all tokens
#pragma GCC unroll 4
      for (int t = 0; t <= pos; t++) {
        // get the value vector for this head and at this timestep
        v4sf *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        // get the attention weight for this timestep
        v4sf a = att[t];

        for (int i = 0; i < head_size; i++) {
          xb[i] += a * v[i];
        }
      }
    }

    // Record elapsed time for attention operations
    LLM_TRACE_END(attention, total_attention_time);
    // Manually get end time for layer tracking
    long trace_end_attention = esp_timer_get_time();
    // Also update the per-layer timing
    if (g_opt_config.tracing_enabled &&
        g_opt_config.layer_attention_times != NULL) {
      g_opt_config.layer_attention_times[l] +=
          (trace_end_attention - trace_start_attention);
    }

    // final matmul to get the output of the attention
    matmul_opt(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

    // residual connection back into x
    LLM_TRACE_START(residual1);
    dsps_add_f32_ansi(x, s->xb2, x, dim, 1, 1, 1);
    LLM_TRACE_END(residual1, total_residual_time);

    LLM_TRACE_START(rmsnorm2);
    rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);
    LLM_TRACE_END(rmsnorm2, total_rmsnorm_time);

    matmul_opt(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
    matmul_opt(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);

    // SwiGLU non-linearity - use optimized version
    LLM_TRACE_START(ffn_layer);
    swiglu_activation(s->hb, s->hb, s->hb2, hidden_dim);
    LLM_TRACE_END(ffn_layer, total_ffn_layer_time);
    // Manually get end time for layer tracking
    long trace_end_ffn_layer = esp_timer_get_time();
    // Update per-layer FFN time
    if (g_opt_config.tracing_enabled && g_opt_config.layer_ffn_times != NULL) {
      g_opt_config.layer_ffn_times[l] +=
          (trace_end_ffn_layer - trace_start_ffn_layer);
    }

    // final matmul to get the output of the ffn
    matmul_opt(s->xb, s->hb, w->w2 + l * hidden_dim * dim, hidden_dim, dim);

    // residual connection
    LLM_TRACE_START(residual2);
    dsps_add_f32_ansi(x, s->xb, x, dim, 1, 1, 1);
    LLM_TRACE_END(residual2, total_residual_time);

    // Record total layer time
    LLM_TRACE_END(layer, total_layer_time);
    // Manually get end time for layer tracking
    long trace_end_layer = esp_timer_get_time();
    // Update per-layer total time
    if (g_opt_config.tracing_enabled && g_opt_config.layer_times != NULL) {
      g_opt_config.layer_times[l] += (trace_end_layer - trace_start_layer);
    }
  }

  // Apply final rmsnorm and classifier separately
  LLM_TRACE_START(rmsnorm_final);
  rmsnorm(x, x, w->rms_final_weight, dim);
  LLM_TRACE_END(rmsnorm_final, total_rmsnorm_time);

  // Check if wcls pointer is valid
  if (w->wcls == NULL) {
    // Keep logits as zeros
    ESP_LOGW(TAG, "WCLS pointer is NULL, using default logits");
    s->logits[0] = 1.0f; // Set token 0 to have highest probability
    return s->logits;
  }

  // classifier into logits
  matmul_opt(s->logits, x, w->wcls, p->dim, p->vocab_size);

  // Replace NaN values with zeros
  for (int i = 0; i < p->vocab_size; i++) {
    if (isnan(s->logits[i])) {
      s->logits[i] = 0.0f;
    }
    if (isinf(s->logits[i])) {
      // Replace infinities with large but finite values
      s->logits[i] = s->logits[i] > 0 ? 1e30f : -1e30f;
    }
  }

  // Record elapsed time at the end
  long trace_start_time = trace_start_forward;
  long trace_end_time = 0;

  LLM_TRACE_END(forward, total_forward_time);

#if LLM_TRACING_LEVEL > LLM_TRACING_DISABLED
  // Get end time after the macro has executed
  trace_end_time = esp_timer_get_time();

  // Capture token timing statistics
  long token_time = trace_end_time - trace_start_time;
  if (token_time < g_opt_config.min_token_time) {
    g_opt_config.min_token_time = token_time;
  }
  if (token_time > g_opt_config.max_token_time) {
    g_opt_config.max_token_time = token_time;
  }
  g_opt_config.last_token_time = token_time;

  // Everything not explicitly timed goes into misc time
  long total_component_time =
      g_opt_config.total_attention_time + g_opt_config.total_matmul_time +
      g_opt_config.total_ffn_time + g_opt_config.total_rmsnorm_time +
      g_opt_config.total_residual_time + g_opt_config.total_rope_time;

  if (token_time > total_component_time) {
    g_opt_config.total_misc_time += (token_time - total_component_time);
  }
#endif

  return s->logits;
}

int compare_tokens(const void *a, const void *b) {
  return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

extern const uint8_t model_bin_end[] asm("_binary_model_bin_end");

void init_tokenizer(Tokenizer *t, int vocab_size) {
  // allocate memory based on the specified vocab_size
  t->vocab_size = vocab_size;
  ESP_LOGI(TAG, "Initializing tokenizer with vocab size: %d", vocab_size);

  // Print current memory status
  ESP_LOGI(
      TAG,
      "Free heap before vocab allocation: %d bytes, largest block: %d bytes",
      heap_caps_get_free_size(MALLOC_CAP_8BIT),
      heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));

  // Try to allocate vocabulary in SPIRAM first (external memory)
  t->vocab =
      heap_caps_malloc(t->vocab_size * sizeof(char *), MALLOC_CAP_SPIRAM);
  if (t->vocab == NULL) {
    // Fall back to internal memory
    ESP_LOGW(TAG,
             "Failed to allocate vocabulary in SPIRAM, trying internal memory");
    t->vocab =
        heap_caps_malloc(t->vocab_size * sizeof(char *), MALLOC_CAP_8BIT);
  }

  if (t->vocab == NULL) {
    ESP_LOGE(TAG, "Failed to allocate vocabulary array");
    t->vocab_size = 0; // Reset vocab_size to indicate initialization failure
    return;            // Exit function but don't crash
  }

  // Initialize the array to NULL to facilitate proper cleanup
  for (int i = 0; i < t->vocab_size; i++) {
    t->vocab[i] = NULL;
  }

  // Try to allocate scores in SPIRAM first
  t->vocab_scores =
      heap_caps_malloc(t->vocab_size * sizeof(v4sf), MALLOC_CAP_SPIRAM);
  if (t->vocab_scores == NULL) {
    // Fall back to internal memory
    ESP_LOGW(
        TAG,
        "Failed to allocate vocab scores in SPIRAM, trying internal memory");
    t->vocab_scores =
        heap_caps_malloc(t->vocab_size * sizeof(v4sf), MALLOC_CAP_8BIT);
  }

  if (t->vocab_scores == NULL) {
    ESP_LOGE(TAG, "Failed to allocate vocabulary scores array");
    // Clean up previously allocated memory
    if (t->vocab) {
      free(t->vocab);
      t->vocab = NULL;
    }
    t->vocab_size = 0; // Reset vocab_size to indicate initialization failure
    return;            // Exit function but don't crash
  }

  t->sorted_vocab = NULL;  // initialized lazily
  t->max_token_length = 0; // Will be set when reading tokenizer data

  ESP_LOGI(TAG, "Tokenizer initialization complete. Free heap: %d bytes",
           heap_caps_get_free_size(MALLOC_CAP_8BIT));
}

void build_tokenizer(Tokenizer *t, char *tokenizer_path) {
  // Initialize byte pieces
  for (int i = 0; i < 256; i++) {
    t->byte_pieces[i * 2] = (unsigned char)i;
    t->byte_pieces[i * 2 + 1] = '\0';
  }

  // Use embedded tokenizer data instead of loading from file
  const uint8_t *embedded_data = tokenizer_bin_start;
  size_t tokenizer_size = tokenizer_bin_end - tokenizer_bin_start;

  if (embedded_data == NULL || tokenizer_size == 0) {
    // Fall back to file if embedded data isn't available
    ESP_LOGW(TAG, "Embedded tokenizer not found, falling back to file: %s",
             tokenizer_path);
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) {
      ESP_LOGE(TAG, "couldn't load %s", tokenizer_path);
      return; // Return gracefully instead of exiting
    }
    ESP_LOGI(TAG, "Opened Tokenizer File");
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) {
      ESP_LOGE(TAG, "failed read");
      fclose(file);
      return; // Return gracefully
    }
    int len;
    for (int i = 0; i < t->vocab_size; i++) {
      if (fread(t->vocab_scores + i, sizeof_v4sf, 1, file) != 1) {
        ESP_LOGE(TAG, "failed read vocab scores");
        fclose(file);
        return; // Return gracefully
      }
      if (fread(&len, sizeof(int), 1, file) != 1) {
        ESP_LOGE(TAG, "failed read len");
        fclose(file);
        return; // Return gracefully
      }

      // Add security check for len value
      if (len <= 0 || len > 1024) {
        ESP_LOGE(TAG, "Invalid token length: %d, expected 1-1024", len);
        fclose(file);
        return; // Return gracefully
      }

      // Try SPIRAM first, then regular memory
      t->vocab[i] = heap_caps_malloc(len + 1, MALLOC_CAP_SPIRAM);
      if (t->vocab[i] == NULL) {
        t->vocab[i] = heap_caps_malloc(len + 1, MALLOC_CAP_8BIT);
      }

      if (t->vocab[i] == NULL) {
        ESP_LOGE(TAG, "Memory allocation failed for token %d (len: %d)", i,
                 len);
        // Free previously allocated tokens
        for (int j = 0; j < i; j++) {
          if (t->vocab[j])
            free(t->vocab[j]);
          t->vocab[j] = NULL;
        }
        fclose(file);
        return; // Return gracefully
      }

      if (fread(t->vocab[i], len, 1, file) != 1) {
        ESP_LOGE(TAG, "failed read vocab");
        fclose(file);
        return; // Return gracefully
      }
      t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
  } else {
    // Read token length from embedded data
    ESP_LOGI(TAG, "Using embedded tokenizer data (%zu bytes)", tokenizer_size);
    size_t data_offset = 0;
    dsps_memcpy_aes3(&t->max_token_length, embedded_data + data_offset,
                     sizeof(int));
    data_offset += sizeof(int);

    // Print memory status before reading vocabulary
    ESP_LOGI(TAG,
             "Free heap before vocabulary: %d bytes, largest block: %d bytes",
             heap_caps_get_free_size(MALLOC_CAP_8BIT),
             heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));

    // Read vocabulary data
    int len;
    for (int i = 0; i < t->vocab_size; i++) {
      // Read vocab score
      dsps_memcpy_aes3(t->vocab_scores + i, embedded_data + data_offset,
                       sizeof_v4sf);
      data_offset += sizeof_v4sf;

      // Read string length
      dsps_memcpy_aes3(&len, embedded_data + data_offset, sizeof(int));
      data_offset += sizeof(int);

      // Add security check for len value
      if (len <= 0 || len > 1024) {
        ESP_LOGE(TAG,
                 "Invalid token length in embedded data: %d, expected 1-1024",
                 len);
        return; // Return gracefully
      }

      // Ensure we don't exceed tokenizer size
      if (data_offset + len > tokenizer_size) {
        ESP_LOGE(TAG,
                 "Data offset (%zu) plus len (%d) exceeds tokenizer size (%zu)",
                 data_offset, len, tokenizer_size);
        return; // Return gracefully
      }

      // Allocate and read string data
      t->vocab[i] = (char *)malloc(len + 1);
      if (t->vocab[i] == NULL) {
        ESP_LOGE(TAG, "Memory allocation failed for token");
        // Clean up previously allocated tokens
        for (int j = 0; j < i; j++) {
          if (t->vocab[j]) {
            free(t->vocab[j]);
            t->vocab[j] = NULL;
          }
        }
        return; // Return gracefully
      }

      dsps_memcpy_aes3(t->vocab[i], embedded_data + data_offset, len);
      data_offset += len;
      t->vocab[i][len] = '\0'; // add the string terminating token
    }

    if (data_offset > tokenizer_size) {
      ESP_LOGE(TAG, "Data offset (%zu) exceeds tokenizer size (%zu)",
               data_offset, tokenizer_size);
    }

    // Print memory status after loading vocabulary
    ESP_LOGI(TAG,
             "Free heap after vocabulary: %d bytes, largest block: %d bytes",
             heap_caps_get_free_size(MALLOC_CAP_8BIT),
             heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
  }

  ESP_LOGI(TAG, "Tokenizer successfully built");
}

void free_tokenizer(Tokenizer *t) {
  for (int i = 0; i < t->vocab_size; i++) {
    free(t->vocab[i]);
  }
  free(t->vocab);
  free(t->vocab_scores);
  free(t->sorted_vocab);
}

char *decode(Tokenizer *t, int prev_token, int token) {
  // Add bounds checking to prevent crashes
  if (t == NULL || t->vocab == NULL || t->vocab_size <= 0) {
    ESP_LOGE(TAG, "Invalid tokenizer in decode: t=%p, vocab=%p, vocab_size=%d",
             t, t ? t->vocab : NULL, t ? t->vocab_size : -1);
    return ""; // Return empty string for invalid tokenizer
  }

  if (token < 0 || token >= t->vocab_size) {
    ESP_LOGE(TAG, "Token %d is out of bounds (vocab_size=%d)", token,
             t->vocab_size);
    return ""; // Return empty string for invalid tokens
  }

  char *piece = t->vocab[token];

  // Check for NULL pointer
  if (piece == NULL) {
    ESP_LOGE(TAG, "Vocabulary entry for token %d is NULL", token);
    return ""; // Return empty string for NULL vocabulary entries
  }

  // following BOS (1) token, sentencepiece decoder strips any leading
  // whitespace (see PR #89)
  if (prev_token == 1 && piece[0] == ' ') {
    piece++;
  }
  // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
  // parse this and convert and return the actual byte
  unsigned char byte_val;
  if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
    piece = (char *)t->byte_pieces + byte_val * 2;
  }
  return piece;
}

void build_sampler(Sampler *sampler, int vocab_size, float temperature,
                   float topp, unsigned long long rng_seed) {
  sampler->vocab_size = vocab_size;
  sampler->temperature = temperature;
  sampler->topp = topp;
  sampler->rng_state = rng_seed;
  sampler->probindex = NULL; // Initialize to NULL before allocation

  // buffer only used with nucleus sampling; may not need but it's ~small
  // Allocate with 16-byte alignment for SIMD operations
  size_t size_needed = sampler->vocab_size * sizeof(ProbIndex);

  // Try to allocate aligned memory using ESP-IDF's heap_caps functions
  sampler->probindex =
      (ProbIndex *)esp32_aligned_malloc(BYTE_ALIGNMENT, size_needed);

  if (sampler->probindex == NULL) {
    // Fallback to regular malloc if aligned allocation fails
    ESP_LOGW(TAG,
             "Aligned allocation failed for ProbIndex, using standard malloc");
    sampler->probindex = malloc(size_needed);
  }

  // Check alignment
  if ((uintptr_t)sampler->probindex % BYTE_ALIGNMENT != 0) {
    ESP_LOGW(TAG, "ProbIndex buffer not aligned: %p (modulo %u = %u)",
             sampler->probindex, (unsigned int)BYTE_ALIGNMENT,
             (unsigned int)((uintptr_t)sampler->probindex % BYTE_ALIGNMENT));
  } else {
    ESP_LOGI(TAG, "ProbIndex buffer properly aligned: %p", sampler->probindex);
  }

  ESP_LOGI(TAG, "Sampler Successfully built");
}

void free_sampler(Sampler *sampler) {
  // For ESP32, we can safely use free() for both aligned and regular
  // allocations
  if (sampler->probindex != NULL) {
    esp32_aligned_free(sampler->probindex);
    sampler->probindex = NULL;
  }
}

void reset_kv_cache(Transformer *transformer) {
  if (transformer == NULL) {
    ESP_LOGE(TAG, "NULL transformer passed to reset_kv_cache");
    return;
  }

  Config *p = &transformer->config;
  RunState *s = &transformer->state;

  // Validate that the state and caches are properly initialized
  if (s == NULL) {
    ESP_LOGE(TAG, "NULL RunState in reset_kv_cache");
    return;
  }

  // Calculate the size of the key-value cache
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  size_t cache_size = (p->n_layers * p->seq_len * kv_dim * sizeof_v4sf) / 2;

  // Validate the key cache
  if (s->key_cache == NULL) {
    ESP_LOGE(TAG, "Key cache is NULL, skipping reset");
  } else {
    // Validate memory address is in valid range
    if ((uintptr_t)s->key_cache < 0x10) {
      ESP_LOGE(TAG, "Invalid key_cache pointer: %p", s->key_cache);
    } else {
      // Reset key cache
      dsps_memset_aes3(s->key_cache, 0, cache_size);
      ESP_LOGD(TAG, "Key cache reset successfully");
    }
  }

  // Validate the value cache
  if (s->value_cache == NULL) {
    ESP_LOGE(TAG, "Value cache is NULL, skipping reset");
  } else {
    // Validate memory address is in valid range
    if ((uintptr_t)s->value_cache < 0x10) {
      ESP_LOGE(TAG, "Invalid value_cache pointer: %p", s->value_cache);
    } else {
      // Reset value cache
      dsps_memset_aes3(s->value_cache, 0, cache_size);
      ESP_LOGD(TAG, "Value cache reset successfully");
    }
  }
}

unsigned int random_u32(unsigned long long *state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

v4sf random_f32(unsigned long long *state) { // random v4sf32 in [0,1)
  return (random_u32(state) >> 8) / 16777216.0f;
}

int str_lookup(char *str, Tokenizer *t) {
  // efficiently find the perfect match for str in vocab, return its index or -1
  // if not found
  TokenIndex tok = {.str = str}; // acts as the key to search for
  TokenIndex *res = bsearch(&tok, t->sorted_vocab, t->vocab_size,
                            sizeof(TokenIndex), compare_tokens);
  return res != NULL ? res->id : -1;
}

void encode(Tokenizer *t, const char *text, int8_t bos, int8_t eos, int *tokens,
            int *n_tokens) {
  // encode the string text (input) into an upper-bound preallocated tokens[]
  // array bos != 0 means prepend the BOS token (=1), eos != 0 means append the
  // EOS token (=2)
  if (text == NULL) {
    ESP_LOGE(TAG, "cannot encode NULL text");
    *n_tokens = 0;
    return; // Return gracefully instead of exiting
  }

  // Check for valid tokenizer and vocabulary size
  if (t == NULL || t->vocab_size <= 0 || t->vocab == NULL) {
    ESP_LOGE(TAG, "Invalid tokenizer state: t=%p, vocab_size=%d, vocab=%p", t,
             t ? t->vocab_size : -1, t ? t->vocab : NULL);
    *n_tokens = 0;
    return;
  }

  if (t->sorted_vocab == NULL) {
    // lazily malloc and sort the vocabulary
    ESP_LOGI(TAG, "Allocating sorted vocabulary table for %d tokens",
             t->vocab_size);

    // First try with SPIRAM if available
    t->sorted_vocab =
        heap_caps_malloc(t->vocab_size * sizeof(TokenIndex), MALLOC_CAP_SPIRAM);

    // If SPIRAM allocation failed or not available, try with regular memory
    if (t->sorted_vocab == NULL) {
      t->sorted_vocab =
          heap_caps_malloc(t->vocab_size * sizeof(TokenIndex), MALLOC_CAP_8BIT);
    }

    // If still failed, report error and try to recover
    if (t->sorted_vocab == NULL) {
      ESP_LOGE(TAG, "Failed to allocate memory for sorted_vocab (%d bytes)",
               (int)(t->vocab_size * sizeof(TokenIndex)));
      ESP_LOGI(TAG, "Free heap: %d bytes, largest block: %d bytes",
               heap_caps_get_free_size(MALLOC_CAP_8BIT),
               heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));

      // Instead of exiting, use a fallback approach
      *n_tokens = 0;
      return; // Return with no tokens rather than crashing
    }

    // Initialize the sorted vocabulary - check each entry to avoid null
    // pointers
    for (int i = 0; i < t->vocab_size; i++) {
      if (t->vocab[i] != NULL) {
        t->sorted_vocab[i].str = t->vocab[i];
        t->sorted_vocab[i].id = i;
      } else {
        ESP_LOGW(TAG, "Null vocabulary entry at index %d", i);
        t->sorted_vocab[i].str = ""; // Use empty string as fallback
        t->sorted_vocab[i].id = i;
      }
    }

    // Sort the vocabulary
    qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    ESP_LOGI(TAG, "Vocabulary sorting complete");
  }

  // create a temporary buffer that will store merge candidates of always two
  // consecutive tokens *2 for concat, +1 for null terminator +2 for UTF8 (in
  // case max_token_length is 1)
  size_t str_buffer_size = t->max_token_length * 2 + 1 + 2;
  char *str_buffer = malloc(str_buffer_size * sizeof(char));
  if (str_buffer == NULL) {
    ESP_LOGE(TAG, "Failed to allocate string buffer");
    *n_tokens = 0;
    return; // Return gracefully instead of exiting
  }
  size_t str_len = 0;

  // start at 0 tokens
  *n_tokens = 0;

  // add optional BOS (=1) token, if desired
  if (bos)
    tokens[(*n_tokens)++] = 1;

  // add_dummy_prefix is true by default
  // so prepend a dummy prefix token to the input string, but only if text != ""
  // TODO: pretty sure this isn't correct in the general case but I don't have
  // the energy to read more of the sentencepiece code to figure out what it's
  // doing
  if (text[0] != '\0') {
    int dummy_prefix = str_lookup(" ", t);
    if (dummy_prefix != -1) { // Check if lookup succeeded
      tokens[(*n_tokens)++] = dummy_prefix;
    }
  }

  // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
  // Code point ↔ UTF-8 conversion
  // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
  // U+0000	U+007F	    0xxxxxxx
  // U+0080	U+07FF	    110xxxxx	10xxxxxx
  // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
  // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

  // process the raw (UTF-8) byte sequence of the input string
  for (const char *c = text; *c != '\0'; c++) {
    // reset buffer if the current byte is ASCII or a leading byte
    // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the
    // rest 0x80 is 10000000 in UTF-8, all continuation bytes start with "10" in
    // first two bits so in English this is: "if this byte is not a continuation
    // byte"
    if ((*c & 0xC0) != 0x80) {
      // this byte must be either a leading byte (11...) or an ASCII char
      // (0x...)
      // => reset our location, as we're starting a new UTF-8 codepoint
      str_len = 0;
    }

    // append the current byte to the buffer
    if (str_len <
        str_buffer_size -
            1) { // Ensure we have space for the byte and null terminator
      str_buffer[str_len++] =
          *c; // ++ is post-increment, incremented after this line
      str_buffer[str_len] = '\0';
    } else {
      // Buffer would overflow, log warning and continue with truncated token
      ESP_LOGW(TAG, "UTF-8 token too long (max %zu bytes), truncating",
               str_buffer_size - 1);
      break; // Exit the loop for this character sequence
    }

    // while the next character is a continuation byte, continue appending
    // but if there are too many of them, just stop to avoid overruning
    // str_buffer size.
    if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
      continue;
    }

    // ok c+1 is not a continuation byte, so we've read in a full codepoint
    int id = str_lookup(str_buffer, t);

    if (id != -1) {
      // we found this codepoint in vocab, add it as a token
      tokens[(*n_tokens)++] = id;
    } else {
      // byte_fallback encoding: just encode each byte as a token
      // +3 is here because the first 3 vocab elements are
      // so the individual bytes only start at index 3
      for (int i = 0; i < str_len; i++) {
        tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
      }
    }
    str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
  }

  // merge the best consecutive pair each iteration, according the scores in
  // vocab_scores
  while (1) {
    v4sf best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;

    for (int i = 0; i < (*n_tokens - 1); i++) {
      // Check the length of both vocabulary strings to prevent buffer overflow
      size_t len1 = strlen(t->vocab[tokens[i]]);
      size_t len2 = strlen(t->vocab[tokens[i + 1]]);

      // Ensure the buffer is large enough for concatenated string plus null
      // terminator
      if (len1 + len2 + 1 > str_buffer_size) {
        // Buffer too small, need to reallocate
        size_t new_size = len1 + len2 + 16; // Add some extra space
        char *new_buffer = realloc(str_buffer, new_size * sizeof(char));
        if (new_buffer == NULL) {
          ESP_LOGE(TAG, "Failed to reallocate merge buffer");
          continue; // Skip this pair if reallocation fails
        }
        str_buffer = new_buffer;
        str_buffer_size = new_size;
      }

      // Now safely concatenate the strings
      if (str_buffer != NULL) {
        strcpy(str_buffer, t->vocab[tokens[i]]);
        strcat(str_buffer, t->vocab[tokens[i + 1]]);

        int id = str_lookup(str_buffer, t);
        if (id != -1 && t->vocab_scores[id] > best_score) {
          // this merge pair exists in vocab! record its score and position
          best_score = t->vocab_scores[id];
          best_id = id;
          best_idx = i;
        }
      }
    }

    if (best_idx == -1) {
      break; // we couldn't find any more pairs to merge, so we're done
    }

    // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
    tokens[best_idx] = best_id;
    // delete token at position best_idx+1, shift the entire sequence back 1
    for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
      tokens[i] = tokens[i + 1];
    }
    (*n_tokens)--; // token length decreased
  }

  // add optional EOS (=2) token, if desired
  if (eos)
    tokens[(*n_tokens)++] = 2;

  if (str_buffer != NULL) {
    free(str_buffer);
  }
}

int sample_argmax(v4sf *probabilities, int n) {
  // return the index that has the highest probability
  if (n <= 0) {
    ESP_LOGE(TAG, "Invalid array size for sample_argmax: %d", n);
    return 0; // Return a safe default
  }

  int max_i = 0;
  v4sf max_p = probabilities[0];
  for (int i = 1; i < n; i++) {
    if (probabilities[i] > max_p) {
      max_i = i;
      max_p = probabilities[i];
    }
  }
  return max_i;
}

int sample_mult(v4sf *probabilities, int n, v4sf coin) {
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  v4sf cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1; // in case of rounding errors
}

// Function to get a fallback token with some basic rules
int get_safe_token(int prev_token) {
  // Simple heuristic to avoid repeating the same token
  if (prev_token >= 0 && prev_token < 10) {
    return prev_token + 10; // Shift to a different range
  }

  // Some common safe tokens that often produce reasonable text
  static const int safe_tokens[] = {29, 8, 13, 15, 1, 19, 30};

  // Get current time as a simple source of randomness
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  int idx = ts.tv_nsec % (sizeof(safe_tokens) / sizeof(safe_tokens[0]));

  return safe_tokens[idx];
}

int sample(Sampler *sampler, v4sf *logits) {
  // Validation
  if (!logits || !sampler || sampler->vocab_size <= 0) {
    ESP_LOGE(TAG,
             "Invalid parameters to sample function: sampler=%p, logits=%p, "
             "vocab_size=%d",
             sampler, logits, sampler ? sampler->vocab_size : -1);
    return get_safe_token(0);
  }

  // Validate ProbIndex buffer
  if (!sampler->probindex) {
    ESP_LOGE(TAG, "ProbIndex buffer is NULL in sampler");
    return get_safe_token(0);
  }

  // Find max logit and its index
  int max_idx = 0;
  v4sf max_val = logits[0];

  for (int i = 1; i < sampler->vocab_size; i++) {
    if (logits[i] > max_val) {
      max_val = logits[i];
      max_idx = i;
    }
  }

  // Check if logits look reasonable
  bool has_invalid = false;
  int valid_count = 0;

  for (int i = 0; i < sampler->vocab_size; i++) {
    if (isnan(logits[i]) || isinf(logits[i])) {
      has_invalid = true;
    } else if (fabsf(logits[i]) < 1000.0f) {
      valid_count++; // Count logits with reasonable magnitude
    }
  }

  // If logits look suspicious, return a safe token
  if (has_invalid || valid_count < 10) {
    ESP_LOGW(TAG, "Suspicious logits detected, using safe token");
    return get_safe_token(max_idx);
  }

  // If temperature is 0 or very close to 0, always use greedy sampling (argmax)
  if (sampler->temperature <= 0.01f) {
    ESP_LOGD(TAG, "Using greedy sampling (temperature near 0)");
    return max_idx;
  }

  // Proper temperature-based sampling
  // 1. Make a copy of logits to avoid modifying the original
  v4sf *temp_logits = NULL;

  // Try to allocate with alignment first
  temp_logits = (v4sf *)esp32_aligned_malloc(
      BYTE_ALIGNMENT, sampler->vocab_size * sizeof(v4sf));

  // If aligned allocation fails, try regular malloc
  if (!temp_logits) {
    ESP_LOGW(TAG,
             "Aligned allocation failed for sampling, trying regular malloc");
    temp_logits = (v4sf *)malloc(sampler->vocab_size * sizeof(v4sf));
  }

  // If all allocation attempts fail, fall back to argmax
  if (!temp_logits) {
    ESP_LOGW(TAG,
             "Memory allocation failed for temperature sampling, using argmax");
    return max_idx;
  }

  // 2. Apply temperature scaling to the logits (divide by temperature)
  for (int i = 0; i < sampler->vocab_size; i++) {
    temp_logits[i] = logits[i] / sampler->temperature;
  }

  // 3. Apply softmax to convert to probabilities
  softmax(temp_logits, sampler->vocab_size);

  // 4. Sample from the distribution
  v4sf random_val = random_f32(&sampler->rng_state);
  int selected_token =
      sample_mult(temp_logits, sampler->vocab_size, random_val);

  // 5. Clean up
  if ((uintptr_t)temp_logits % BYTE_ALIGNMENT == 0) {
    esp32_aligned_free(temp_logits); // Use aligned free if it was aligned
  } else {
    free(temp_logits); // Use regular free if it wasn't aligned
  }

  return selected_token;
}

// Directly use log_memory_usage instead of log_memory_status
#ifdef SHOW_LOGITS_DEBUG
#define LOG_MEMORY(phase, token) log_memory_usage(TAG, phase, token)
#else
#define LOG_MEMORY(phase, token)                                               \
  do {                                                                         \
  } while (0)
#endif

// Global token callback
static void (*g_token_callback)(const char *token, void *user_data) = NULL;
static void *g_token_callback_user_data = NULL;

// Global cancellation check callback
static bool (*g_cancellation_check)(void) = NULL;

/**
 * @brief Register a callback for checking if generation should be cancelled
 *
 * @param check_cancelled Function that returns true if generation should stop
 */
void register_cancellation_check(bool (*check_cancelled)(void)) {
  g_cancellation_check = check_cancelled;
  ESP_LOGI(TAG, "Cancellation check %s",
           check_cancelled ? "registered" : "cleared");
}

/**
 * @brief Register a callback for token output
 *
 * @param callback Function to call for each generated token
 * @param user_data User data to pass to the callback
 */
void register_token_callback(void (*callback)(const char *token,
                                              void *user_data),
                             void *user_data) {
  g_token_callback = callback;
  g_token_callback_user_data = user_data;
  // ESP_LOGI(TAG, "Token callback %s", callback ? "registered" : "cleared");
}

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
              const char *prompt, int steps, generated_complete_cb cb_done) {
  // Record accurate timing
  long overall_start = esp_timer_get_time();

  // Reset performance metrics before generation
  llm_metrics_reset(&g_opt_config);

  // Validate parameters
  if (transformer == NULL || tokenizer == NULL || sampler == NULL) {
    ESP_LOGE(TAG, "NULL parameter passed to generate function");
    if (cb_done) {
      cb_done(0.0f); // Call callback with zero tokens/sec to indicate failure
    }
    return;
  }

  // Check tokenizer validity
  if (tokenizer->vocab_size <= 0 || tokenizer->vocab == NULL) {
    ESP_LOGE(TAG, "Invalid tokenizer: vocab_size=%d, vocab=%p",
             tokenizer->vocab_size, tokenizer->vocab);
    if (cb_done) {
      cb_done(0.0f);
    }
    return;
  }

  if (steps <= 0 || steps > transformer->config.seq_len) {
    ESP_LOGI(TAG, "Adjusting steps from %d to max sequence length %ld", steps,
             transformer->config.seq_len);
    steps = transformer->config.seq_len;
  }

  char *empty_prompt = "";
  if (prompt == NULL) {
    prompt = empty_prompt;
  }

  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  // Allocate enough space for worst-case tokenization scenario:
  // - Each byte could potentially generate multiple tokens in complex
  // tokenization
  // - Plus tokens for BOS, EOS, and dummy prefix
  // - Using a conservative multiplier for safety
  int *prompt_tokens = (int *)malloc((strlen(prompt) * 4 + 10) * sizeof(int));

  if (prompt_tokens == NULL) {
    ESP_LOGE(TAG, "Failed to allocate memory for prompt tokens");
    if (cb_done) {
      cb_done(0.0f);
    }
    return;
  }

  encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
  if (num_prompt_tokens < 1) {
    ESP_LOGE(TAG, "something is wrong, expected at least 1 prompt token");
    free(prompt_tokens);
    if (cb_done) {
      cb_done(0.0f);
    }
    return;
  }

  ESP_LOGI(TAG, "Starting generation with %d prompt tokens", num_prompt_tokens);
  LOG_MEMORY("Before generation", 0);

  // Use a local flag to detect early termination
  bool terminated_early = false;

  // start the main loop
  long token_start = 0;
  long total_gen_time = 0;
  int next;                     // will store the next token in the sequence
  int token = prompt_tokens[0]; // kick off with the first token in the prompt
  int pos = 0;                  // position in the sequence
  int generated_tokens = 0; // count of newly generated tokens (not from prompt)

  while (pos < steps) {
    // Check if cancellation was requested
    if (g_cancellation_check != NULL && g_cancellation_check()) {
      ESP_LOGI(TAG, "Generation cancelled by user request");
      terminated_early = true;
      break;
    }

    // Start timing this token
    token_start = esp_timer_get_time();

    // forward the transformer to get logits for the next token
    v4sf *logits = forward(transformer, token, pos);

    // Check if logits are valid
    if (logits == NULL) {
      ESP_LOGE(TAG, "Forward pass returned NULL logits");
      terminated_early = true;
      break;
    }

    // advance the state machine
    if (pos < num_prompt_tokens - 1) {
      // if we are still processing the input prompt, force the next prompt
      // token
      next = prompt_tokens[pos + 1];
    } else {
      // otherwise sample the next token from the logits
      next = sample(sampler, logits);
      generated_tokens++;

      // Calculate token generation time
      long token_end = esp_timer_get_time();
      long token_time = token_end - token_start;
      total_gen_time += token_time;
    }
    pos++;

    // data-dependent terminating condition: the BOS (=1) token delimits
    // sequences
    if (next == 1) {
      ESP_LOGI(TAG, "Hit BOS token (1), ending generation");
      break;
    }

    // On ESP32-S3, we sometimes may need to skip specific tokens that end
    // generation
    if (next == 0) {
      ESP_LOGI(TAG, "Got end token (0), skipping...");
      next = 29; // Use a common token to keep generation going
      continue;
    }

    // Check token validity
    if (next < 0 || next >= tokenizer->vocab_size) {
      ESP_LOGW(TAG, "Invalid token generated: %d (vocab size: %d)", next,
               tokenizer->vocab_size);
      next = 29; // Use a common token as fallback
    }

    // decode the token as string
    char *piece = decode(tokenizer, token, next);
    if (piece == NULL || strlen(piece) == 0) {
      ESP_LOGW(TAG, "Empty or NULL token decoded, using fallback");
      piece = " "; // Use space as fallback
    } else {
      ESP_LOGD(TAG, "Generated piece: '%s' (token %d)", piece, next);
    }

    token = next;

    // Call the token callback if registered
    if (g_token_callback != NULL) {
      g_token_callback(piece, g_token_callback_user_data);
    }
  }

  // Print performance metrics with more detail
  if (g_opt_config.tracing_level >= LLM_TRACING_DETAILED) {
    llm_metrics_print_detailed(&g_opt_config);
  } else {
    llm_metrics_print_summary(&g_opt_config);
  }

  // Calculate elapsed time and token speed
  long end = esp_timer_get_time();
  long elapsed = end - overall_start;
  v4sf tks = 0.0f;

  // Prevent division by zero
  if (elapsed > 0) {
    tks = (v4sf)generated_tokens * 1000000.0f / (v4sf)elapsed;
  } else if (generated_tokens > 0) {
    // If somehow the timer shows no elapsed time but we did generate tokens
    tks = generated_tokens * 10.0f; // Just report some reasonable number
  }

  ESP_LOGI(TAG, "Total tokens generated: %d in %.2f seconds (%.2f tokens/sec)",
           generated_tokens, elapsed / 1000000.0f, tks);

  if (generated_tokens > 0) {
    ESP_LOGI(TAG, "Token timing: min=%.2f ms, max=%.2f ms, avg=%.2f ms",
             g_opt_config.min_token_time / 1000.0f,
             g_opt_config.max_token_time / 1000.0f,
             total_gen_time / generated_tokens / 1000.0f);
  }

  if (cb_done) {
    cb_done(terminated_early ? 0.0f : tks);
  }

  free(prompt_tokens);
}

void rmsnorm(v4sf *o, v4sf *x, v4sf *weight, int size) {
  // Add timing for rmsnorm
  long start_time = esp_timer_get_time();

  // Calculate sum of squares using dot product of vector with itself
  v4sf ss = 0.0f;
  dsps_dotprod_f32_aes3(x, x, &ss, size);

  ss = ss / size;
  ss += 1e-5f;

  // Use standard division for inverse square root
  v4sf scale = 1.0f / dsps_sqrtf_f32_ansi(ss);

// Apply normalization in one direct loop - safest approach
#pragma GCC unroll 4
  for (int i = 0; i < size; i++) {
    o[i] = weight[i] * (x[i] * scale);
  }

  // Update timing metrics
  long elapsed = esp_timer_get_time() - start_time;
  g_opt_config.total_rmsnorm_time += elapsed;
}

void softmax(v4sf *x, int size) {
  if (x == NULL || size <= 0)
    return;

  // Find the maximum value for numerical stability.
  v4sf max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val)
      max_val = x[i];
  }

  // Fuse the exponential computation and sum in one pass.
  v4sf sum = 0.0f;
  for (int i = 0; i < size; i++) {
    float t = fast_exp(x[i] - max_val);
    x[i] = t;
    sum += t;
  }

  // Normalize the results.
  if (sum > 0.0f) {
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; i++) {
      x[i] *= inv_sum;
    }
  } else {
    // Edge case: all values were extremely negative or expf overflowed.
    x[0] = 1.0f;
    for (int i = 1; i < size; i++) {
      x[i] = 0.0f;
    }
  }
}
