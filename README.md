# Running a LLM on the ESP32-S3

## Summary
This project demonstrates running a Large Language Model (LLM) directly on an ESP32-S3 microcontroller. It implements a highly optimized inference engine capable of running small transformer models with extensive optimizations for embedded systems.

The model is a 260K parameter [TinyLlamas checkpoint](https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K) trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. Despite its small size, it generates coherent, simple text suitable for embedded applications.

Based on [llama2.c](https://github.com/karpathy/llama2.c) with extensive ESP32-specific optimizations.

## Hardware Requirements
- **ESP32-S3** with 2MB PSRAM (ESP32-S3FH4R2)
- 8MB flash (7.9MB app partition)
- ~1.5MB available RAM

## Performance
Achieves **~32.83 tokens/second** through:

- **SIMD Acceleration**: ESP32-S3 vector instructions via [ESP-DSP](https://github.com/espressif/esp-dsp)
- **Memory Alignment**: 16-byte aligned allocations for SIMD operations
- **Dual-Core Processing**: Parallel computation across both cores
- **Optimized Clocks**: CPU at 240MHz, PSRAM at 80MHz
- **Assembly Optimizations**: Custom float division routines
- **Efficient Math**: Lookup tables for activation functions

## Features

### Web Interface
- Real-time token streaming via WebSocket
- Adjustable temperature and max tokens
- Mobile-responsive design
- Access at http://192.168.4.1

### Core Components
- **LLM Engine**: Complete transformer implementation with top-p sampling
- **Memory Management**: Custom aligned allocators for SIMD efficiency
- **WiFi Manager**: Station mode connectivity
- **Embedded Model**: 1MB model + 6KB tokenizer built into firmware

## Project Structure

```
├── main/                  # Application entry point
├── components/
│   ├── llm/              # LLM inference engine
│   │   ├── assets/       # Embedded model & tokenizer
│   │   └── src/          # Core implementation + ASM
│   └── web/              # Web server & WiFi
│       └── src/          # HTTP/WebSocket server
└── partitions.csv        # 8MB partition table
```

## Setup and Installation

Requires [ESP-IDF](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/index.html#installation) v5.3.2 or later.

### 1. Configure WiFi
```bash
# Copy the template
cp components/web/include/wifi_config.h.template components/web/include/wifi_config.h

# Edit with your WiFi credentials
# Set WIFI_SSID and WIFI_PASS in wifi_config.h
```

### 2. Build and Flash
```bash
# Set up ESP-IDF environment
. ~/esp/esp-idf/export.sh

# Configure for ESP32-S3
idf.py set-target esp32s3

# Build the project
idf.py build

# Flash and monitor (replace PORT with your device)
idf.py -p /dev/ttyUSB0 flash monitor
```

## Usage

1. After flashing, ESP32 connects to your WiFi network
2. Find the device IP in serial monitor output
3. Navigate to `http://<device-ip>` in your browser
4. Enter prompts and watch tokens stream in real-time

### Serial Output
The device provides detailed logs including:
- Memory usage statistics
- Performance metrics per layer
- Token generation progress

## Configuration

Key options in `idf.py menuconfig`:
- I2C GPIO pins (if using external peripherals)
- I2C clock speed

## Technical Details

- **Model**: 260K parameters, 6 layers, 288 dimensions
- **Vocabulary**: 512 tokens optimized for simple stories
- **Context**: 512 token maximum sequence length
- **Sampling**: Top-p (nucleus) sampling with temperature control
- **Memory**: ~1.5MB runtime memory requirement

## Limitations

- Simple vocabulary suitable for basic stories
- 512 token context window
- No dynamic model loading (embedded in firmware)
- WiFi required for web interface

## Contributing

Contributions welcome! Areas for improvement:
- Further SIMD optimizations
- Model quantization support
- External storage for larger models
- Additional sampling methods

## License

This project is intended to be open source. Please add a LICENSE file to clarify terms.
