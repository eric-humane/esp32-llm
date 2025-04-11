# Embedded Tokenizer

The tokenizer is embedded directly into the application binary using `target_add_binary_data` in the component's CMakeLists.txt.

## Embedded Model Files

Similarly, the LLM model file is now embedded directly into the application binary, eliminating the need for SPIFFS storage.

### How to embed your own model

1. Place your model file named `model.bin` in the `components/llm_core/assets/` directory
2. The model will automatically be embedded in the app partition via `target_add_binary_data(${COMPONENT_LIB} "assets/model.bin" BINARY)`
3. The code already has the necessary declarations to use the embedded model:
   ```c
   extern const uint8_t model_bin_start[] asm("_binary_model_bin_start");
   extern const uint8_t model_bin_end[] asm("_binary_model_bin_end");
   ```
4. No filesystem operations are needed to access the model

### Important Notes

1. Using an embedded model increases the app partition size significantly
2. The partition table has been updated to accommodate the larger app partition
3. If your model is very large, you may need to further adjust the partition table
4. Embedding the model improves reliability as there's no filesystem dependency
5. Initial flash time will be longer due to the larger binary

### Fallback to Filesystem

The code retains the ability to load models from SPIFFS if needed:
- If the embedded model is not available, it will try to load from `/data/model.bin`
- This provides flexibility during development

## Implementation

The tokenizer.bin file is stored in the components/llm_core/assets directory and:

1. It gets embedded in the app partition via `target_add_binary_data(${COMPONENT_LIB} "assets/tokenizer.bin" BINARY)`
2. It doesn't get copied to SPIFFS as it's not in the main data directory

This approach lets the system:
- Access the tokenizer directly from the app binary with no file I/O
- Reduce SPIFFS usage by not having a duplicate copy there

## Benefits
- Faster access from embedded copy (no file I/O needed)
- Reduced SPIFFS space usage
- Simple implementation with no file movement during build 