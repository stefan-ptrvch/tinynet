// Program for line profiling, i. e. calculating the center of gravity for
// every row of an image. The program is used for detecting a line, as well as
// the intensity and continuity of the line.
//
// Author: Stefan Petrovic
// Date: 07. VI 2017.
// Mail: stefan.petrovic@novelic.com

#include "stm32f7xx.h"
#include "stm32f7xx_nucleo_144.h"
#include "hardware_init.h"
#include "cycle_counter.h"

#include "tinynet.h"

// Function for printing flower name based on prediction
void print_flower(float* vec)
{
    // Variables for probabilities
    double a, b, c, max;

    // Get values
    a = vec[0];
    b = vec[1];
    c = vec[2];

    max = a;

    if(max < b)
        max = b;
    if(max < c)
        max = c;

    if(max == a)
        printf("%s\n", "Flower is Iris-setosa");
    else if(max == b)
        printf("%s\n", "Flower is Iris-versicolor");
    else if(max == c)
        printf("%s\n", "Flower is Iris-virginica");

}

int main(void)
{
	  int clk_num;
	  double processing_time;

	// Initialize hardware
	init_hardware();

    while(1)
    {
        printf("Enter flower dimensions:\n");
        scanf("%f %f %f %f", &input[0], &input[1], &input[2], &input[3]);

        // Reset cycle counter                                               //
        reset_timer();                                                       //
                                                                             //
        //Start counting clock cycles                                        //
        start_timer();

        predict();

        // Stop the clock cycle counter                                      //
        stop_timer();                                                        //
                                                                             //
        // Get number of cycles                                              //
        clk_num = get_num_cycles();

        // Calculate time spend on processing                                //
        processing_time = (double) clk_num/(double) 216000000;
        printf("Time spent on processing: %.5f\n", processing_time);
        print_flower(prediction);
        printf("\n");
    }

    return 0;
}
