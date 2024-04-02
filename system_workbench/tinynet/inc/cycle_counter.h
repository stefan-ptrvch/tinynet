#include <stdint.h>

// Reset the clock cycle counter
void reset_timer();

// Start the clock cycle counter
void start_timer();

// Stop the clock cycle counter
void stop_timer();

// Get the number of counted clock cycles
uint32_t get_num_cycles();
