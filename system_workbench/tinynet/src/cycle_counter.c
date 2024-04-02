// Set of functions for counting the number of clock cycles of a sectioin of
// code.
//
// Author: Stefan Petrovic
// Date: 21. VI 2017.
// Mail: stefan.petrovic@novelic.com

#include <stdint.h>
#include "cycle_counter.h"

// Addresses of registers
volatile uint32_t *DWT_CONTROL = (uint32_t *)0xE0001000;
volatile uint32_t *DWT_CYCCNT = (uint32_t *)0xE0001004;
volatile uint32_t *DEMCR = (uint32_t *)0xE000EDFC;

// Lock access register
volatile uint32_t *LAR  = (uint32_t *) 0xE0001FB0;

// Reset the clock cycle counter
void reset_timer()
{
    // Enable access to DWT registers
    *LAR = 0xC5ACCE55;

    // Clear the count register
    *DWT_CYCCNT = 0;

    // Stop the counter
    *DWT_CONTROL = 0;

    // Enable the DWT
    *DEMCR = *DEMCR | 0x01000000;
}

// Start the clock cycle counter
void start_timer()
{
    // Enable cycle counter
    *DWT_CONTROL = *DWT_CONTROL | 1;
}

// Stop the clock cycle counter
void stop_timer()
{
    // Clear the control register to stop the timer
    *DWT_CONTROL = 0;
}

// Get the number of counted clock cycles
uint32_t get_num_cycles()
{
    return *DWT_CYCCNT;
}
