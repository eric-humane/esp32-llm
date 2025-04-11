#ifndef LLM_H
#define LLM_H

/**
 * Original author of this:
 * https://github.com/karpathy/llama2.c
 *
 * Slight modifications added to make it ESP32 friendly
 */

#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "memory_utils.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef float v4sf __attribute__((aligned(BYTE_ALIGNMENT)));
typedef float float4_t
    __attribute__((vector_size(BYTE_ALIGNMENT))); // GCC vector type for SIMD

extern void *dsps_memset_aes3(void *arr_dest, uint8_t set_val, size_t set_size);
extern void *dsps_memcpy_aes3(void *arr_dest, const void *arr_src,
                              size_t arr_len);

// Add configuration and benchmarking structure
typedef struct {
  // Performance metrics
  long total_attention_time;
  long total_matmul_time;
  long total_ffn_time;
  long total_forward_time;
  long total_rmsnorm_time;
  long total_residual_time;
  long total_rope_time;
  long total_misc_time;

  // Additional timing metrics for specialized operations
  long total_ffn_layer_time;
  long total_layer_time;

  int num_forward_calls;
  int num_tokens_processed;
  bool perform_validation;

  // New detailed metrics
  // Per-layer metrics
  long *layer_times;           // Time spent in each layer
  long *layer_attention_times; // Attention time per layer
  long *layer_ffn_times;       // FFN time per layer

  // Per-operation detailed metrics
  long *matmul_count; // Count of matrix multiplications
  long *matmul_sizes; // Sizes of matrix multiplications (flattened as n*d)
  long *matmul_times; // Times for individual matmuls
  int matmul_metrics_index; // Current index in circular buffer
  int matmul_metrics_size;  // Size of metrics circular buffer

  // Statistical metrics
  long min_token_time;  // Minimum time per token
  long max_token_time;  // Maximum time per token
  long last_token_time; // Last token generation time

  // Tracing configuration
  int tracing_level;    // Current tracing level
  bool tracing_enabled; // Master switch for tracing
} OptimizationConfig;

typedef struct {
  float prob;
  int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

// Expanded Sampler for additional sampling methods
typedef struct {
  int vocab_size;
  ProbIndex *probindex; // buffer used in top-p sampling
  float temperature;
  float topp;
  uint64_t rng_state;

  // Enhanced sampling parameters
  bool use_enhanced_sampling; // Whether to use enhanced sampling methods

  // Top-k sampling
  bool use_top_k;
  int top_k;
} Sampler;

typedef struct {
  char *str;
  int id;
} TokenIndex;

typedef struct {
  char **vocab;
  v4sf *vocab_scores;
  TokenIndex *sorted_vocab;
  int vocab_size;
  unsigned int max_token_length;
  unsigned char byte_pieces[1024]; // stores all single-byte strings
} Tokenizer;

#pragma pack(push, 1)
typedef struct {
  int32_t dim;        // transformer dimension
  int32_t hidden_dim; // for ffn layers
  int32_t n_layers;   // number of layers
  int32_t n_heads;    // number of query heads
  int32_t n_kv_heads; // number of key/value heads (can be < query heads because
                      // of multiquery)
  int32_t vocab_size; // vocabulary size, usually 256 (byte-level)
  int32_t seq_len;    // max sequence length
} Config;
#pragma pack(pop)

typedef struct {
  // token embedding table
  v4sf *token_embedding_table; // (vocab_size, dim)
  // weights for rmsnorms
  v4sf *rms_att_weight; // (layer, dim) rmsnorm weights
  v4sf *rms_ffn_weight; // (layer, dim)
  // weights for matmuls. note dim == n_heads * head_size
  v4sf *wq; // (layer, dim, n_heads * head_size)
  v4sf *wk; // (layer, dim, n_kv_heads * head_size)
  v4sf *wv; // (layer, dim, n_kv_heads * head_size)
  v4sf *wo; // (layer, n_heads * head_size, dim)
  // weights for ffn
  v4sf *w1; // (layer, hidden_dim, dim)
  v4sf *w2; // (layer, dim, hidden_dim)
  v4sf *w3; // (layer, hidden_dim, dim)
  // final rmsnorm
  v4sf *rms_final_weight; // (dim,)
  // (optional) classifier weights for the logits, on the last layer
  v4sf *wcls;
} TransformerWeights;

typedef struct {
  v4sf *x;      // activation at current time stamp (dim,)
  v4sf *xb;     // same, but inside a residual branch (dim,)
  v4sf *xb2;    // an additional buffer just for convenience (dim,)
  v4sf *hb;     // buffer for hidden dimension in the ffn (hidden_dim,)
  v4sf *hb2;    // buffer for hidden dimension in the ffn (hidden_dim,)
  v4sf *q;      // query (dim,)
  v4sf *k;      // key (dim,)
  v4sf *v;      // value (dim,)
  v4sf *att;    // buffer for scores/attention values (n_heads, seq_len)
  v4sf *logits; // output logits
  // Buffers that need to be retained across forward calls
  v4sf *key_cache;       // (layer, seq_len, dim)
  v4sf *value_cache;     // (layer, seq_len, dim)
  uint8_t *memory_block; // pointer to the single memory allocation
  int effective_seq_len; // the effective sequence length for KV cache
} RunState;

// Add after TransformerWeights definition and before RunState
typedef struct {
  // Original weight pointers (now will point to cached data)
  TransformerWeights weights;

  // Flash memory information
  const uint8_t *flash_data; // Pointer to model data in flash
  size_t flash_data_size;    // Size of model data
  size_t config_size;        // Size of config header

  // Layer offsets in flash (calculated once during initialization)
  size_t *layer_offsets; // Array of offsets for each layer

  // Weight cache
  v4sf *cache_buffer; // Buffer for cached weights
  int cached_layer;   // Currently cached layer (-1 if none)
  size_t cache_size;  // Size of cache buffer

  // Keep token embedding and final weights always in memory
  v4sf *token_embedding_table; // Permanent copy of token embeddings
  v4sf *rms_final_weight;      // Permanent copy of final layer norm weights
  v4sf *wcls; // Permanent copy of classifier weights (may share with
              // token_embedding)

  bool shared_weights;    // Whether token embedding and classifier weights are
                          // shared
  bool use_flash_weights; // Whether to load weights from flash on-demand
} WeightManager;

typedef struct {
  Config config; // the hyperparameters of the architecture (the blueprint)
  TransformerWeights weights; // the weights of the model
  RunState state; // buffers for the "wave" of activations in the forward pass
  // some more state needed to properly clean up the memory mapping (sigh)
  int fd;                       // file descriptor for memory mapping
  v4sf *data;                   // memory mapped data pointer
  size_t file_size;             // size of the checkpoint file in bytes
  WeightManager weight_manager; // Manages on-demand weight loading

  // Precomputed RoPE trigonometric values
  float *rope_sin_cache; // [seq_len][head_size/2] precomputed sine values
  float *rope_cos_cache; // [seq_len][head_size/2] precomputed cosine values
} Transformer;

typedef void (*generated_complete_cb)(float tokens_ps);

bool build_transformer(Transformer *t, char *checkpoint_path);
/**
 * @brief Build a transformer with the option to load weights from flash
 * on-demand
 *
 * @param t Pointer to transformer structure to initialize
 * @param checkpoint_path Path to the model checkpoint
 * @param use_flash_weights Whether to enable on-demand loading of weights from
 * flash
 */
bool build_transformer_with_flash(Transformer *t, char *checkpoint_path,
                                  bool use_flash_weights);
void init_tokenizer(Tokenizer *t, int vocab_size);
void build_tokenizer(Tokenizer *t, char *tokenizer_path);
void build_sampler(Sampler *sampler, int vocab_size, float temperature,
                   float topp, unsigned long long rng_seed);
void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
              const char *prompt, int steps, generated_complete_cb cb_done);
void free_sampler(Sampler *sampler);
void free_transformer(Transformer *t);
void free_tokenizer(Tokenizer *t);
void reset_kv_cache(Transformer *transformer);

void reset_performance_metrics();

void print_performance_metrics();

// Register callbacks
void register_token_callback(void (*callback)(const char *token,
                                              void *user_data),
                             void *user_data);
void register_cancellation_check(bool (*check_cancelled)(void));

// Add declaration for vector_rmsnorm function
void vector_rmsnorm(float *output, float *input, float *weight, int size);

/**
 * @brief Initialize the LLM core with configuration options
 * @return ESP_OK on success
 *
 * This function should be called before using any LLM functionality
 * to properly configure the memory and performance settings.
 */
esp_err_t llm_init();

// Add new function declarations
/**
 * @brief Initialize weight manager for on-demand loading from flash
 * @param transformer Pointer to transformer structure
 * @param use_flash Whether to use on-demand loading from flash
 * @return ESP_OK on success
 */
esp_err_t init_weight_manager(Transformer *transformer, bool use_flash);

/**
 * @brief Load weights for a specific layer from flash
 * @param transformer Pointer to transformer structure
 * @param layer Layer number to load
 * @return ESP_OK on success
 */
esp_err_t load_layer_weights(Transformer *transformer, int layer);

// Advanced metrics and tracing infrastructure
#define LLM_TRACING_DISABLED 0
#define LLM_TRACING_MINIMAL 1
#define LLM_TRACING_NORMAL 2
#define LLM_TRACING_DETAILED 3

// Set default tracing level - can be overridden in menuconfig
#ifndef LLM_TRACING_LEVEL
#define LLM_TRACING_LEVEL LLM_TRACING_NORMAL
#endif

// Tracing macros
#if LLM_TRACING_LEVEL > LLM_TRACING_DISABLED
#define LLM_TRACE_START(name)                                                  \
  long trace_start_##name =                                                    \
      (g_opt_config.tracing_enabled) ? esp_timer_get_time() : 0
#define LLM_TRACE_END(name, target)                                            \
  if (g_opt_config.tracing_enabled) {                                          \
    long trace_end_##name = esp_timer_get_time();                              \
    g_opt_config.target += (trace_end_##name - trace_start_##name);            \
  }
#define LLM_TRACE_END_LAYER(name, target, layer)                               \
  if (g_opt_config.tracing_enabled && g_opt_config.layer_times != NULL) {      \
    long trace_end_##name = esp_timer_get_time();                              \
    g_opt_config.total_##name##_time +=                                        \
        (trace_end_##name - trace_start_##name);                               \
    g_opt_config.target[layer] += (trace_end_##name - trace_start_##name);     \
  }
#else
#define LLM_TRACE_START(name)
#define LLM_TRACE_END(name, target)
#define LLM_TRACE_END_LAYER(name, target, layer)
#endif

// Additional detailed tracing macros
#if LLM_TRACING_LEVEL >= LLM_TRACING_DETAILED
#define LLM_TRACE_MATMUL(n, d, time)                                           \
  if (g_opt_config.tracing_enabled &&                                          \
      g_opt_config.matmul_metrics_index < g_opt_config.matmul_metrics_size) {  \
    int idx = g_opt_config.matmul_metrics_index++;                             \
    g_opt_config.matmul_count[idx]++;                                          \
    g_opt_config.matmul_sizes[idx] = (n) * (d);                                \
    g_opt_config.matmul_times[idx] = (time);                                   \
    if (g_opt_config.matmul_metrics_index >=                                   \
        g_opt_config.matmul_metrics_size) {                                    \
      g_opt_config.matmul_metrics_index = 0; /* Circular buffer */             \
    }                                                                          \
  }
#else
#define LLM_TRACE_MATMUL(n, d, time)
#endif

// Trace utility functions
void llm_metrics_init(OptimizationConfig *config, int n_layers);
void llm_metrics_free(OptimizationConfig *config);
void llm_metrics_reset(OptimizationConfig *config);
void llm_metrics_print_summary(OptimizationConfig *config);
void llm_metrics_print_detailed(OptimizationConfig *config);
void llm_metrics_set_tracing_level(OptimizationConfig *config, int level);

// Make global config variable accessible
extern OptimizationConfig g_opt_config;

bool malloc_run_state(RunState *s, Config *p);
bool read_checkpoint(char *checkpoint, Config *config,
                     TransformerWeights *weights, int *fd, v4sf **data,
                     size_t *file_size);

#endif