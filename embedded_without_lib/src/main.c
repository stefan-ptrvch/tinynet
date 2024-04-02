#include <stdio.h>
#include "tinynet.h"


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

int main()
{
    while(1)
    {
        printf("Enter flower dimensions:\n");
        scanf("%f %f %f %f", &input[0], &input[1], &input[2], &input[3]);

        predict();

        print_flower(prediction);
        printf("\n");
    }
}
