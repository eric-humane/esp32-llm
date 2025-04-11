.text
.align 4
.global __divsf3
.type __divsf3, @function

/*
 * Optimized float division for ESP32-S3
 * Uses reciprocal instructions for faster division while maintaining accuracy
 * 
 * Replaces the standard libgcc implementation with a much faster version
 * This function will be called for all floating-point divisions (a / b)
 */
__divsf3:
    entry   a1, 16          // Standard function prologue with stack frame
    
    // Move inputs to float registers
    wfr     f0, a2          // numerator -> f0
    wfr     f1, a3          // denominator -> f1
    
    // Initialize reciprocal approximation
    recip0.s  f2, f1
    
    // First Newton-Raphson iteration to improve accuracy
    const.s   f3, 1
    msub.s    f3, f1, f2
    madd.s    f2, f2, f3
    
    // Second Newton-Raphson iteration for better accuracy
    const.s   f3, 1
    msub.s    f3, f1, f2
    maddn.s   f2, f2, f3
    
    // Compute a * (1/b)
    mul.s     f0, f0, f2
    
    // Return result in a2 as per ABI
    rfr     a2, f0
    
    retw.n              // Return with window increment

.size __divsf3, .-__divsf3