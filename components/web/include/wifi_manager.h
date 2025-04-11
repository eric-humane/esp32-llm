#pragma once

#include "esp_err.h"

/**
 * @brief Initialize and connect to WiFi network
 *
 * @return esp_err_t ESP_OK on success
 */
esp_err_t wifi_init_sta(void);