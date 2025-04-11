#include "esp_attr.h"
#include "esp_log.h"
#include "esp_task_wdt.h"
#include "memory_utils.h"
#include "web_server.h"
#include "wifi_manager.h"

static const char *TAG = "MAIN";

/**
 * @brief Configure watchdog with a longer timeout
 */
static void configure_task_watchdog(void) {
  // Increase watchdog timeout to accommodate long inference times
  esp_task_wdt_config_t wdt_config = {
      .timeout_ms = 60000,        // 60 second timeout
      .idle_core_mask = (1 << 0), // Don't monitor idle task on core 0
      .trigger_panic = false      // Don't panic on timeout, just warn
  };

  ESP_LOGI(TAG, "Configuring task watchdog with 60-second timeout");
  esp_err_t err = esp_task_wdt_reconfigure(&wdt_config);
  if (err != ESP_OK) {
    ESP_LOGW(TAG, "Failed to reconfigure watchdog: %s", esp_err_to_name(err));
  }
}

void app_main(void) {
  // Initialize system
  ESP_LOGI(TAG, "Starting ESP32-LLM");
  log_memory_usage(TAG, "Startup", -1);

  // Configure watchdog with extended timeout
  configure_task_watchdog();

  // Initialize WiFi
  ESP_ERROR_CHECK(wifi_init_sta());

  // Initialize the inference system
  init_inference_system();

  // Start web server
  start_webserver();

  log_memory_usage(TAG, "After initialization", -1);
}