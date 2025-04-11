#pragma once

#include "esp_http_server.h"

/**
 * @brief Start the web server for serving the LLM interface
 *
 * @return httpd_handle_t Handle to the HTTP server
 */
httpd_handle_t start_webserver(void);

/**
 * @brief Submit an inference request to the LLM
 *
 * @param prompt The text prompt to send to the LLM
 * @param max_tokens Maximum number of tokens to generate
 * @param req HTTP request that triggered the inference
 * @return esp_err_t ESP_OK on success
 */
esp_err_t submit_inference_request(const char *prompt, int max_tokens,
                                   httpd_req_t *req);

/**
 * @brief Initialize the inference system
 */
void init_inference_system(void);