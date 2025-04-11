#include "web_server.h"
#include "cJSON.h"
#include "esp_http_server.h"
#include "esp_log.h"
// Removed watchdog-related includes:
// #include "esp_task_wdt.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "llm.h"
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static const char *TAG = "WEB";

// External function declarations from llm_core
extern void register_token_callback(void (*callback)(const char *, void *),
                                    void *user_data);
extern void reset_kv_cache(Transformer *transformer);
extern void register_cancellation_check(bool (*check_cancelled)(void));

// Global sampling parameters
static float g_temperature = 1.0f;
static float g_topp = 0.90f;

// Global variables to track LLM state and results
static Transformer *g_transformer = NULL;
static Tokenizer *g_tokenizer = NULL;
static Sampler *g_sampler = NULL;

// Atomic flags - use proper atomic types and operations
static atomic_bool g_token_generated = ATOMIC_VAR_INIT(false);
static atomic_bool g_inference_running = ATOMIC_VAR_INIT(false);
static atomic_bool g_cancel_requested = ATOMIC_VAR_INIT(false);

// Mutex for resource protection (not for atomic variables)
static SemaphoreHandle_t g_inference_semaphore = NULL;
static SemaphoreHandle_t g_ws_mutex = NULL; // Only for WebSocket ops
static httpd_handle_t g_server = NULL;      // Global server handle

// Global variables to manage the inference task
static TaskHandle_t g_inference_task_handle = NULL;
static QueueHandle_t g_inference_queue = NULL;
static volatile bool g_response_pending = false;
static httpd_req_t *g_current_req = NULL;

// WebSocket connection variables
static volatile int g_ws_fd = -1; // WebSocket file descriptor

// Define the inference task structure
typedef struct {
  char prompt[1024];
  int max_tokens;
  httpd_req_t *req;
} inference_request_t;

// Forward declarations
static void batch_output_capture_callback(const char *token, void *user_data);
static void llm_inference_task(void *pvParameters);
static void optimize_tokenizer_memory(Tokenizer *tokenizer);
static void generate_complete_cb(float tk_s);
static void flush_ws_buffer(void);

// -------------------
// Token Batching
// -------------------
#define TOKEN_BATCH_SIZE 16
#define TOKEN_BUF_SIZE 512

static char s_token_buffer[TOKEN_BUF_SIZE];
static size_t s_buffer_len = 0;
static size_t s_token_count = 0;

/**
 * @brief Flush batched tokens over WebSocket if connected
 */
static void flush_ws_buffer(void) {
  if (s_buffer_len == 0 || g_ws_fd < 0 || g_server == NULL) {
    // Reset batch counters regardless
    s_buffer_len = 0;
    s_token_count = 0;
    s_token_buffer[0] = '\0';
    return;
  }

  httpd_ws_frame_t ws_pkt = {.payload = (uint8_t *)s_token_buffer,
                             .len = s_buffer_len,
                             .type = HTTPD_WS_TYPE_TEXT};

  // Only lock for the duration of the send
  if (xSemaphoreTake(g_ws_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
    esp_err_t ret = httpd_ws_send_frame_async(g_server, g_ws_fd, &ws_pkt);
    if (ret != ESP_OK) {
      // Gracefully handle disconnection
      if (ret == ESP_ERR_INVALID_STATE || ret == ESP_ERR_INVALID_ARG) {
        ESP_LOGW(TAG, "WebSocket disconnected, marking as closed");
        g_ws_fd = -1;
      } else {
        ESP_LOGE(TAG, "httpd_ws_send_frame_async failed with %d", ret);
      }
    }
    xSemaphoreGive(g_ws_mutex);
  }

  // Reset batch
  s_buffer_len = 0;
  s_token_count = 0;
  s_token_buffer[0] = '\0';
}

/**
 * @brief Called after generation completes.
 * @param tk_s tokens/sec (unused in this example)
 */
static void generate_complete_cb(float tk_s) {
  // Flush any leftover tokens once the generation ends
  flush_ws_buffer();
}

/**
 * @brief Callback for batching tokens before sending to WebSocket
 */
static void batch_output_capture_callback(const char *token, void *user_data) {
  if (!token) {
    return;
  }
  // Check cancellation
  if (atomic_load(&g_cancel_requested)) {
    return; // Stop processing tokens
  }

  // Mark that we generated at least one token
  atomic_store(&g_token_generated, true);

  size_t token_len = strlen(token);
  if (token_len == 0) {
    return; // Skip empty
  }

  // If token won't fit, flush first
  if ((s_buffer_len + token_len) >= TOKEN_BUF_SIZE) {
    flush_ws_buffer();
  }

  // Append token to the buffer
  memcpy(&s_token_buffer[s_buffer_len], token, token_len);
  s_buffer_len += token_len;
  s_token_buffer[s_buffer_len] = '\0';

  s_token_count++;

  // Flush if we hit the batch size
  if (s_token_count >= TOKEN_BATCH_SIZE) {
    flush_ws_buffer();
  }
}

/**
 * @brief LLM inference task that runs on Core 1, event-driven via queue
 */
static void llm_inference_task(void *pvParameters) {
  inference_request_t request;

  while (1) {
    // Wait until we have a new request in the queue
    BaseType_t queue_result =
        xQueueReceive(g_inference_queue, &request, portMAX_DELAY);

    // We have a request
    if (queue_result == pdTRUE) {
      // Set global state
      g_response_pending = true;
      g_current_req = request.req;

      // Reset atomic flags
      atomic_store(&g_token_generated, false);
      atomic_store(&g_cancel_requested, false);

      // Ensure transformer is valid before resetting KV cache
      if (!g_transformer) {
        const char *error_msg =
            "{\"error\":\"Internal error: Transformer is NULL\"}";
        httpd_ws_frame_t ws_pkt = {0};
        ws_pkt.payload = (uint8_t *)error_msg;
        ws_pkt.len = strlen(error_msg);
        ws_pkt.type = HTTPD_WS_TYPE_TEXT;

        // Only lock during send
        if (xSemaphoreTake(g_ws_mutex, pdMS_TO_TICKS(500)) == pdTRUE) {
          if (g_ws_fd >= 0 && g_server != NULL) {
            httpd_ws_send_frame_async(g_server, g_ws_fd, &ws_pkt);
          }
          xSemaphoreGive(g_ws_mutex);
        }
        g_response_pending = false;
        atomic_store(&g_inference_running, false);
        xSemaphoreGive(g_inference_semaphore);
        continue;
      }

      // Reset the KV cache for a fresh generation
      reset_kv_cache(g_transformer);

      // Register our token callback
      register_token_callback(batch_output_capture_callback, NULL);

      // Generate the response (blocking call)
      generate(g_transformer, g_tokenizer, g_sampler, request.prompt,
               request.max_tokens, &generate_complete_cb);

      // Unregister callback
      register_token_callback(NULL, NULL);

      // Check if any tokens were generated
      bool token_generated = atomic_load(&g_token_generated);
      bool cancel_requested = atomic_load(&g_cancel_requested);

      if (!token_generated) {
        const char *error_msg = "{\"error\":\"No output was generated. "
                                "Possibly no recognized tokens.\"}";

        httpd_ws_frame_t ws_pkt = {0};
        ws_pkt.payload = (uint8_t *)error_msg;
        ws_pkt.len = strlen(error_msg);
        ws_pkt.type = HTTPD_WS_TYPE_TEXT;

        if (xSemaphoreTake(g_ws_mutex, pdMS_TO_TICKS(500)) == pdTRUE) {
          if (g_ws_fd >= 0 && g_server != NULL) {
            esp_err_t ret =
                httpd_ws_send_frame_async(g_server, g_ws_fd, &ws_pkt);
            if (ret != ESP_OK) {
              ESP_LOGW(TAG, "Failed to send error message; WebSocket may be "
                            "disconnected");
            }
          }
          xSemaphoreGive(g_ws_mutex);
        }
        ESP_LOGW(TAG, "No output captured during generation");
      } else if (cancel_requested) {
        // Inform client that generation was cancelled
        const char *cancel_msg = "{\"status\":\"cancelled\"}";
        httpd_ws_frame_t ws_pkt = {0};
        ws_pkt.payload = (uint8_t *)cancel_msg;
        ws_pkt.len = strlen(cancel_msg);
        ws_pkt.type = HTTPD_WS_TYPE_TEXT;

        if (xSemaphoreTake(g_ws_mutex, pdMS_TO_TICKS(500)) == pdTRUE) {
          if (g_ws_fd >= 0 && g_server != NULL) {
            esp_err_t ret =
                httpd_ws_send_frame_async(g_server, g_ws_fd, &ws_pkt);
            if (ret != ESP_OK) {
              ESP_LOGW(TAG, "Failed to send cancellation message; WebSocket "
                            "may be disconnected");
            }
          }
          xSemaphoreGive(g_ws_mutex);
        }
        ESP_LOGI(TAG, "Generation was cancelled");
      } else {
        // Send completion notification
        const char *completion_msg = "{\"status\":\"completed\"}";
        httpd_ws_frame_t ws_pkt = {0};
        ws_pkt.payload = (uint8_t *)completion_msg;
        ws_pkt.len = strlen(completion_msg);
        ws_pkt.type = HTTPD_WS_TYPE_TEXT;

        if (xSemaphoreTake(g_ws_mutex, pdMS_TO_TICKS(500)) == pdTRUE) {
          if (g_ws_fd >= 0 && g_server != NULL) {
            esp_err_t ret =
                httpd_ws_send_frame_async(g_server, g_ws_fd, &ws_pkt);
            if (ret != ESP_OK) {
              ESP_LOGW(TAG, "Failed to send completion message; WebSocket may "
                            "be disconnected");
            }
          }
          xSemaphoreGive(g_ws_mutex);
        }
      }

      // Inference is done
      g_response_pending = false;
      atomic_store(&g_inference_running, false);
      // Release the semaphore so other inferences can proceed
      xSemaphoreGive(g_inference_semaphore);
    }
  }
}

/**
 * @brief Optimize memory usage by freeing non-essential tokenizer data
 */
static void optimize_tokenizer_memory(Tokenizer *tokenizer) {
  if (!tokenizer || !tokenizer->vocab) {
    ESP_LOGW(TAG, "Cannot optimize NULL tokenizer or one with NULL vocab");
    return;
  }

  // The sorted_vocab is created lazily and only needed for encoding
  if (tokenizer->sorted_vocab) {
    free(tokenizer->sorted_vocab);
    tokenizer->sorted_vocab = NULL;
  }
}

/**
 * @brief WebSocket handler for token streaming
 */
static esp_err_t ws_handler(httpd_req_t *req) {
  if (req->method == HTTP_GET) {
    ESP_LOGI(TAG, "WebSocket connection established");

    // Store the WebSocket connection descriptor (mutex protected)
    if (xSemaphoreTake(g_ws_mutex, pdMS_TO_TICKS(500)) == pdTRUE) {
      // Close any existing connection first
      g_ws_fd = -1;
      g_ws_fd = httpd_req_to_sockfd(req);
      xSemaphoreGive(g_ws_mutex);
    } else {
      ESP_LOGE(TAG, "Failed to acquire WebSocket mutex");
      return ESP_FAIL;
    }

    return ESP_OK;
  }

  // We expect text frames
  httpd_ws_frame_t ws_pkt = {0};
  ws_pkt.type = HTTPD_WS_TYPE_TEXT;

  // First call with NULL to get size
  esp_err_t ret = httpd_ws_recv_frame(req, &ws_pkt, 0);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Failed to get WebSocket frame size: %d", ret);
    if (ret == ESP_ERR_INVALID_STATE || ret == ESP_FAIL) {
      if (xSemaphoreTake(g_ws_mutex, pdMS_TO_TICKS(500)) == pdTRUE) {
        g_ws_fd = -1;
        xSemaphoreGive(g_ws_mutex);
      }
    }
    return ESP_FAIL;
  }

  // Size limit
  if (ws_pkt.len > 16384) {
    ESP_LOGE(TAG, "WebSocket frame too large: %d bytes", ws_pkt.len);
    return ESP_FAIL;
  }

  // Allocate buffer for the incoming frame
  uint8_t *buf = malloc(ws_pkt.len + 1);
  if (!buf) {
    ESP_LOGE(TAG, "Failed to allocate memory for WebSocket message");
    return ESP_FAIL;
  }

  ws_pkt.payload = buf;
  ret = httpd_ws_recv_frame(req, &ws_pkt, ws_pkt.len);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "httpd_ws_recv_frame failed with %d", ret);
    free(buf);

    if (ret == ESP_ERR_INVALID_STATE || ret == ESP_FAIL) {
      if (xSemaphoreTake(g_ws_mutex, pdMS_TO_TICKS(500)) == pdTRUE) {
        g_ws_fd = -1;
        xSemaphoreGive(g_ws_mutex);
      }
    }
    return ESP_FAIL;
  }

  buf[ws_pkt.len] = 0;

  // Check for close message
  if (ws_pkt.type == HTTPD_WS_TYPE_CLOSE) {
    if (xSemaphoreTake(g_ws_mutex, pdMS_TO_TICKS(500)) == pdTRUE) {
      g_ws_fd = -1;
      xSemaphoreGive(g_ws_mutex);
    }
    free(buf);
    return ESP_OK;
  }

  // Process commands from client
  if (ws_pkt.type == HTTPD_WS_TYPE_TEXT) {
    cJSON *root = cJSON_Parse((char *)buf);
    if (root) {
      cJSON *cmd = cJSON_GetObjectItem(root, "command");
      if (cmd && cJSON_IsString(cmd)) {
        const char *command = cmd->valuestring;

        // PROMPT command
        if (strcmp(command, "prompt") == 0) {
          char prompt[1024] = {0};
          int max_tokens = 100;
          float temperature = g_temperature; // default

          // Read prompt
          cJSON *prompt_json = cJSON_GetObjectItem(root, "prompt");
          if (prompt_json && cJSON_IsString(prompt_json)) {
            strncpy(prompt, prompt_json->valuestring, sizeof(prompt) - 1);
          }

          // Read max_tokens
          cJSON *tokens_json = cJSON_GetObjectItem(root, "max_tokens");
          if (tokens_json && cJSON_IsNumber(tokens_json)) {
            max_tokens = tokens_json->valueint;
            if (max_tokens <= 0 || max_tokens > 256) {
              max_tokens = 100; // fallback
            }
          }

          // Read temperature
          cJSON *temp_json = cJSON_GetObjectItem(root, "temperature");
          if (temp_json && cJSON_IsNumber(temp_json)) {
            temperature = temp_json->valuedouble;
            if (temperature < 0.0f || temperature > 2.0f) {
              temperature = g_temperature; // out of range => revert
            } else {
              g_temperature = temperature;
              if (g_sampler) {
                g_sampler->temperature = temperature;
              }
            }
          }

          // Reset cancellation for new generation
          atomic_store(&g_cancel_requested, false);

          // Check if inference is already running
          if (atomic_load(&g_inference_running)) {
            const char *error_msg =
                "{\"error\":\"Another inference is already running\"}";
            httpd_ws_frame_t err_frame = {.final = true,
                                          .fragmented = false,
                                          .type = HTTPD_WS_TYPE_TEXT,
                                          .payload = (uint8_t *)error_msg,
                                          .len = strlen(error_msg)};
            httpd_ws_send_frame(req, &err_frame);
            cJSON_Delete(root);
            free(buf);
            return ESP_OK;
          }

          // Mark inference as running
          atomic_store(&g_inference_running, true);

          // Take inference semaphore
          if (xSemaphoreTake(g_inference_semaphore, pdMS_TO_TICKS(1000)) !=
              pdTRUE) {
            atomic_store(&g_inference_running, false);
            const char *error_msg =
                "{\"error\":\"Server busy (semaphore timeout)\"}";
            httpd_ws_frame_t err_frame = {.final = true,
                                          .fragmented = false,
                                          .type = HTTPD_WS_TYPE_TEXT,
                                          .payload = (uint8_t *)error_msg,
                                          .len = strlen(error_msg)};
            httpd_ws_send_frame(req, &err_frame);
            cJSON_Delete(root);
            free(buf);
            return ESP_OK;
          }

          // Create inference request
          inference_request_t request;
          memset(&request, 0, sizeof(request));
          strncpy(request.prompt, prompt, sizeof(request.prompt) - 1);
          request.max_tokens = max_tokens;
          request.req = req;

          // Send to inference queue
          if (xQueueSend(g_inference_queue, &request, pdMS_TO_TICKS(1000)) !=
              pdTRUE) {
            ESP_LOGE(TAG, "Failed to queue inference request");
            atomic_store(&g_inference_running, false);
            xSemaphoreGive(g_inference_semaphore);

            const char *error_msg = "{\"error\":\"Server busy (queue full)\"}";
            httpd_ws_frame_t err_frame = {.final = true,
                                          .fragmented = false,
                                          .type = HTTPD_WS_TYPE_TEXT,
                                          .payload = (uint8_t *)error_msg,
                                          .len = strlen(error_msg)};
            httpd_ws_send_frame(req, &err_frame);
            cJSON_Delete(root);
            free(buf);
            return ESP_OK;
          }

          // Immediately notify the inference task
          xTaskNotifyGive(g_inference_task_handle);
        }
        // CANCEL command
        else if (strcmp(command, "cancel") == 0) {
          bool is_inference_running = atomic_load(&g_inference_running);
          if (is_inference_running) {
            atomic_store(&g_cancel_requested, true);
            const char *msg = "{\"status\":\"cancellation_requested\"}";
            httpd_ws_frame_t resp = {.final = true,
                                     .fragmented = false,
                                     .type = HTTPD_WS_TYPE_TEXT,
                                     .payload = (uint8_t *)msg,
                                     .len = strlen(msg)};
            httpd_ws_send_frame(req, &resp);
            ESP_LOGI(TAG, "Cancellation requested");
          } else {
            const char *msg = "{\"status\":\"nothing_to_cancel\"}";
            httpd_ws_frame_t resp = {.final = true,
                                     .fragmented = false,
                                     .type = HTTPD_WS_TYPE_TEXT,
                                     .payload = (uint8_t *)msg,
                                     .len = strlen(msg)};
            httpd_ws_send_frame(req, &resp);
          }
        }
      }
      cJSON_Delete(root);
    } else {
      // JSON parsing failed
      ESP_LOGW(TAG, "Failed to parse JSON");
      const char *error_msg = "{\"error\":\"Invalid JSON format\"}";
      httpd_ws_frame_t err_frame = {.final = true,
                                    .fragmented = false,
                                    .type = HTTPD_WS_TYPE_TEXT,
                                    .payload = (uint8_t *)error_msg,
                                    .len = strlen(error_msg)};
      httpd_ws_send_frame(req, &err_frame);
    }
  }

  free(buf);
  return ESP_OK;
}

/**
 * @brief Minimal HTML page with WebSocket support (truncated for brevity).
 */
static const char *index_html =
    "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"UTF-8\"><meta "
    "name=\"viewport\" content=\"width=device-width, "
    "initial-scale=1.0\"><title>ESP32-S3 LLM</"
    "title><style>:root{--bg:#1e1e2a;--bg-input:#2a2a36;--text:#f9fafb;--text-"
    "secondary:#a0a0b0;--primary:#4d7cfe;--primary-hover:#3d6cee;--error:#"
    "dc2626;--success:#16a34a}*{box-sizing:border-box;margin:0;padding:0}body{"
    "font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe "
    "UI',Roboto,sans-serif;line-height:1.5;background-color:var(--bg);color:"
    "var(--text);padding:20px;max-width:800px;margin:0 "
    "auto}h1{margin-bottom:20px;font-size:28px;font-weight:500}textarea{width:"
    "100%;padding:16px;border:1px solid "
    "#444;border-radius:6px;background-color:var(--bg-input);color:var(--text);"
    "min-height:200px;font-family:inherit;margin-bottom:24px;resize:vertical;"
    "font-size:16px}textarea:focus{outline:none;border-color:var(--primary)}."
    "controls{display:flex;flex-direction:column;gap:24px;margin-bottom:24px}."
    "control{display:flex;align-items:center;justify-content:space-between}."
    "control "
    "label{font-size:16px;color:var(--text)}.slider-container{flex:1;padding:0 "
    "16px;position:relative}input[type=\"range\"]{-webkit-appearance:none;"
    "width:100%;height:4px;background:#444;border-radius:2px}input[type="
    "\"range\"]::-webkit-slider-thumb{-webkit-appearance:none;width:18px;"
    "height:18px;border-radius:50%;background:#fff;cursor:pointer}input[type="
    "\"range\"]::-moz-range-thumb{width:18px;height:18px;border-radius:50%;"
    "background:#fff;cursor:pointer;border:none}.value-display{min-width:50px;"
    "text-align:right;font-size:16px;color:var(--text)}.buttons{display:flex;"
    "gap:12px}button{padding:12px "
    "24px;border:none;border-radius:6px;font-size:16px;cursor:pointer;font-"
    "weight:500}button:disabled{opacity:.6;cursor:not-allowed}button.primary{"
    "background-color:var(--primary);color:#fff}button.primary:hover:not(:"
    "disabled){background-color:var(--primary-hover)}button.secondary{"
    "background-color:#333;color:var(--text)}button.secondary:hover{background-"
    "color:#444}button.danger{background-color:var(--error);color:#fff}#status{"
    "margin-top:16px;padding:8px;border-radius:6px}.status-info{background-"
    "color:rgb(79 124 254 / .2)}.status-error{background-color:rgb(220 38 38 / "
    ".2);color:var(--error)}.status-success{background-color:rgb(22 163 74 / "
    ".2);color:var(--success)}@media(max-width:600px){.buttons{flex-direction:"
    "column}button{width:100%}}</style></head><body><h1>ESP32-S3 LLM</h1><form "
    "id=\"promptForm\"><textarea id=\"prompt\" placeholder=\"Enter your prompt "
    "here...\"><|start_story|></textarea><div class=\"controls\"><div "
    "class=\"control\"><label for=\"temperature\">Temperature</label><div "
    "class=\"slider-container\"><input type=\"range\" id=\"temperature\" "
    "min=\"0\" max=\"2\" step=\"0.01\" value=\"1.00\"></div><span "
    "id=\"temperatureValue\" class=\"value-display\">1.00</span></div><div "
    "class=\"control\"><label for=\"max_tokens\">Max tokens</label><div "
    "class=\"slider-container\"><input type=\"range\" id=\"max_tokens\" "
    "min=\"1\" max=\"256\" value=\"256\"></div><span id=\"maxTokensValue\" "
    "class=\"value-display\">256</span></div></div><div "
    "class=\"buttons\"><button type=\"submit\" id=\"submitBtn\" "
    "class=\"primary\">Generate</button><button type=\"button\" "
    "id=\"resetBtn\" class=\"secondary\">Reset</button><button type=\"button\" "
    "id=\"cancelBtn\" class=\"danger\" style=\"display: "
    "none;\">Cancel</button></div></form><div id=\"status\" style=\"display: "
    "none;\"></div><script>const "
    "elements={form:document.getElementById('promptForm'),prompt:document."
    "getElementById('prompt'),maxTokens:document.getElementById('max_tokens'),"
    "maxTokensValue:document.getElementById('maxTokensValue'),temperature:"
    "document.getElementById('temperature'),temperatureValue:document."
    "getElementById('temperatureValue'),submitBtn:document.getElementById('"
    "submitBtn'),cancelBtn:document.getElementById('cancelBtn'),resetBtn:"
    "document.getElementById('resetBtn'),status:document.getElementById('"
    "status')};const "
    "state={wsConnected:!1,generating:!1,tokens:0,maxTokens:parseInt(elements."
    "maxTokens.value),temperature:parseFloat(elements.temperature.value)};let "
    "socket=null;let "
    "firstTokensReceived=!1;elements.temperature.oninput=function(){state."
    "temperature=parseFloat(this.value);elements.temperatureValue.textContent="
    "state.temperature.toFixed(2)};elements.maxTokens.oninput=function(){state."
    "maxTokens=parseInt(this.value);elements.maxTokensValue.textContent=state."
    "maxTokens};function connectWebSocket(){const "
    "wsUrl=`ws://${window.location.host}/ws`;socket=new "
    "WebSocket(wsUrl);socket.onopen=()=>{state.wsConnected=!0;console.log('"
    "WebSocket "
    "connected')};socket.onclose=()=>{state.wsConnected=!1;console.log('"
    "WebSocket disconnected, "
    "reconnecting...');setTimeout(connectWebSocket,2000)};socket.onerror=("
    "error)=>{showStatus('error','WebSocket error "
    "occurred');console.error('WebSocket "
    "error:',error)};socket.onmessage=(event)=>{try{const "
    "jsonMsg=JSON.parse(event.data);handleJsonMessage(jsonMsg)}catch(e){"
    "handleToken(event.data)}}}function "
    "handleJsonMessage(message){if(message.error){showStatus('error',`Error:${"
    "message.error}`);resetGenerationState()}else "
    "if(message.status){if(message.status==='completed'){showStatus('success','"
    "Generation completed');resetGenerationState()}else "
    "if(message.status==='cancelled'){showStatus('info','Generation "
    "cancelled');resetGenerationState()}}}function "
    "handleToken(token){if(!firstTokensReceived){elements.prompt.value='';"
    "firstTokensReceived=!0}elements.prompt.value+=token;elements.prompt."
    "scrollTop=elements.prompt.scrollHeight;state.tokens++}function "
    "showStatus(type,message){elements.status.textContent=message;elements."
    "status.className='';elements.status.classList.add(`status-${type}`);"
    "elements.status.style.display='block';if(type==='success'){setTimeout(()=>"
    "{elements.status.style.display='none'},3000)}}function "
    "resetGenerationState(){state.generating=!1;elements.submitBtn.disabled=!1;"
    "elements.cancelBtn.style.display='none'}elements.form.addEventListener('"
    "submit',(e)=>{e.preventDefault();if(!state.wsConnected){showStatus('error'"
    ",'WebSocket not connected. "
    "Reconnecting...');connectWebSocket();return}if(state.generating){"
    "showStatus('error','Already "
    "generating');return}state.generating=!0;state.tokens=0;"
    "firstTokensReceived=!1;elements.submitBtn.disabled=!0;elements.cancelBtn."
    "style.display='inline-block';elements.status.style.display='none';const "
    "message={command:'prompt',prompt:elements.prompt.value,max_tokens:state."
    "maxTokens,temperature:state.temperature};socket.send(JSON.stringify("
    "message))});elements.cancelBtn.addEventListener('click',()=>{if(!state."
    "generating)return;const "
    "message={command:'cancel'};socket.send(JSON.stringify(message));"
    "showStatus('info','Cancelling...')});elements.resetBtn.addEventListener('"
    "click',()=>{if(state.generating){socket.send(JSON.stringify({command:'"
    "cancel'}));resetGenerationState()}elements.prompt.value='<|start_story|>';"
    "elements.status.style.display='none'});window.addEventListener('load',"
    "connectWebSocket);</script></body></html>";

/**
 * @brief Handler for GET requests to the index page
 */
static esp_err_t index_get_handler(httpd_req_t *req) {
  httpd_resp_set_type(req, "text/html");
  httpd_resp_send(req, index_html, strlen(index_html));
  return ESP_OK;
}

/**
 * @brief Start the HTTP server with WebSocket support
 */
httpd_handle_t start_webserver(void) {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.stack_size = 8192; // Increase stack size if needed

  httpd_handle_t server = NULL;

  if (httpd_start(&server, &config) == ESP_OK) {
    // Register URI handlers
    httpd_uri_t index_uri = {.uri = "/",
                             .method = HTTP_GET,
                             .handler = index_get_handler,
                             .user_ctx = NULL};
    httpd_register_uri_handler(server, &index_uri);

    // Register WebSocket handler
    httpd_uri_t ws_uri = {.uri = "/ws",
                          .method = HTTP_GET,
                          .handler = ws_handler,
                          .user_ctx = NULL,
                          .is_websocket = true};
    httpd_register_uri_handler(server, &ws_uri);

    ESP_LOGI(TAG, "Web server started with WebSocket support");
    g_server = server;
    return server;
  }

  ESP_LOGE(TAG, "Error starting server!");
  return NULL;
}

/**
 * @brief Checks if cancellation is requested
 */
static bool is_cancellation_requested(void) {
  return atomic_load(&g_cancel_requested);
}

/**
 * @brief Initialize the inference system with WebSocket support
 */
void init_inference_system(void) {
  // Create the inference semaphore
  g_inference_semaphore = xSemaphoreCreateBinary();
  xSemaphoreGive(g_inference_semaphore);

  // Create WebSocket mutex
  g_ws_mutex = xSemaphoreCreateMutex();
  if (!g_ws_mutex) {
    ESP_LOGE(TAG, "Failed to create WebSocket mutex");
    return;
  }

  // Initialize atomic variables
  atomic_store(&g_token_generated, false);
  atomic_store(&g_inference_running, false);
  atomic_store(&g_cancel_requested, false);

  // Model setup
  char *checkpoint_path = "__EMBEDDED__";       // Example for embedded model
  char *tokenizer_path = "/data/tokenizer.bin"; // Or embedded
  unsigned long long rng_seed = (unsigned int)(time(NULL) % 10000);
  ESP_LOGI(TAG, "Using random seed: %llu", rng_seed);

  // Allocate Transformer
  g_transformer = (Transformer *)malloc(sizeof(Transformer));
  if (!g_transformer) {
    ESP_LOGE(TAG, "Failed to allocate memory for transformer");
    return;
  }

  if (!build_transformer(g_transformer, checkpoint_path)) {
    ESP_LOGE(TAG, "Failed to build transformer");
    free(g_transformer);
    g_transformer = NULL;
    return;
  }

  // Allocate Tokenizer in SPIRAM if available
  g_tokenizer =
      (Tokenizer *)heap_caps_malloc(sizeof(Tokenizer), MALLOC_CAP_SPIRAM);
  if (!g_tokenizer) {
    ESP_LOGE(TAG, "Failed to allocate tokenizer");
    return;
  }
  init_tokenizer(g_tokenizer, g_transformer->config.vocab_size);
  build_tokenizer(g_tokenizer, tokenizer_path);

  // Allocate Sampler
  g_sampler = (Sampler *)malloc(sizeof(Sampler));
  if (!g_sampler) {
    ESP_LOGE(TAG, "Failed to allocate memory for sampler");
    return;
  }
  build_sampler(g_sampler, g_transformer->config.vocab_size, g_temperature,
                g_topp, rng_seed);

  // Register cancellation check
  register_cancellation_check(is_cancellation_requested);

  // Optimize memory usage
  optimize_tokenizer_memory(g_tokenizer);

  // Create inference queue
  g_inference_queue = xQueueCreate(4, sizeof(inference_request_t));
  if (!g_inference_queue) {
    ESP_LOGE(TAG, "Failed to create inference queue");
    return;
  }

  // Create inference task pinned to core 1
  BaseType_t res =
      xTaskCreatePinnedToCore(llm_inference_task, "llm_inference", 8192, NULL,
                              6, &g_inference_task_handle, 1);
  if (res != pdPASS) {
    ESP_LOGE(TAG, "Failed to create inference task");
  }
}

/**
 * @brief Submit an inference request to the LLM
 */
esp_err_t submit_inference_request(const char *prompt, int max_tokens,
                                   httpd_req_t *req) {
  // Check if inference is running
  if (atomic_load(&g_inference_running)) {
    return ESP_FAIL;
  }

  // Mark inference as running
  atomic_store(&g_inference_running, true);

  // Take semaphore
  if (xSemaphoreTake(g_inference_semaphore, pdMS_TO_TICKS(1000)) != pdTRUE) {
    atomic_store(&g_inference_running, false);
    return ESP_ERR_TIMEOUT;
  }

  // Build request
  inference_request_t request;
  memset(&request, 0, sizeof(request));
  strncpy(request.prompt, prompt, sizeof(request.prompt) - 1);
  request.max_tokens = max_tokens;
  request.req = req;

  // Send request
  if (xQueueSend(g_inference_queue, &request, pdMS_TO_TICKS(1000)) != pdTRUE) {
    ESP_LOGE(TAG, "Failed to queue inference request");
    atomic_store(&g_inference_running, false);
    xSemaphoreGive(g_inference_semaphore);
    return ESP_FAIL;
  }

  // Immediately notify the inference task
  xTaskNotifyGive(g_inference_task_handle);
  return ESP_OK;
}