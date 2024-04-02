#include <stdio.h>
#include <gsl/gsl_blas.h>
#include "utils.h"

int main (void)
{
    // Iteration variable
    int i;

    // Variable for holding the number of layers
    int numLayers;

    // Variables for hodling matrix dimensions
    int numRows, numCols;

    // Open file that contains the weights
    FILE *f = fopen("./weights.txt", "r");

    // Get number of layers
    fscanf(f, "%d", &numLayers);

    // Variable for holding number of units per layer
    int units[numLayers];

    // Get number of units per layer
    for(i = 0; i < numLayers; i++)
    {
        fscanf(f, "%d", &units[i]);
    }

    // Array for holding weight matrix pointers
    gsl_matrix *weights[numLayers - 1];

    // Get the weights
    for(i = 1; i < numLayers; i++)
    {
        numRows = units[i];
        numCols = units[i - 1] + 1;

        weights[i - 1] = getWeights(f, numRows, numCols);
    }

    // Close file containg the weights
    fclose(f);

    // Define an input vector
    gsl_vector *x = gsl_vector_alloc(4);

    // Get input vector from user
    double input;

    for(i = 0; i < units[0]; i++)
    {
        printf("Feature x%d: ", i + 1);
        scanf("%lf", &input);
        gsl_vector_set(x, i, input);
    }

    // Evaluate forwardProp
    forwardProp(x, weights, numLayers, units);

    // Remove matrices and vectors from memory
    for(i = 0; i < numLayers - 1; i++)
    {
        gsl_matrix_free(weights[i]);
    }

    return 0;
}
