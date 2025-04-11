set(ASM_SOURCES
    "src/asm/float_div.S"
)

idf_component_register(
    SRCS 
        "src/llm.c"
        "src/memory_utils.c"
        ${ASM_SOURCES}
    INCLUDE_DIRS 
        "include"
    REQUIRES 
        esp-dsp 
    PRIV_REQUIRES
        esp_timer
)

# Disable if-conversion to avoid warnings
target_compile_options(${COMPONENT_LIB} PRIVATE -fno-if-conversion)

# Embed the tokenizer binary data into the application
target_add_binary_data(${COMPONENT_LIB} "assets/tokenizer.bin" BINARY)

# Embed the model binary data into the application
target_add_binary_data(${COMPONENT_LIB} "assets/model.bin" BINARY)

