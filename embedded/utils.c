#include <stdio.h>
#include <gsl/gsl_blas.h>
#include <math.h>

/* Function for adding a column of ones
   to a matrix */
gsl_matrix *addOnes(gsl_matrix *mat, int m, int n)
{
    // Variable for iteration
    int i;

    // Allocate memory for new matrix
    gsl_matrix *matOnes = gsl_matrix_alloc(m, n + 1);

    // Allocate memory for help vector
    gsl_vector *helpVec = gsl_vector_alloc(m);

    // Set all elements to 1
    gsl_vector_set_all(helpVec, 1);

    // Copy column vector into new matrix
    gsl_matrix_set_col(matOnes, 0, helpVec);

    // Copy old matrix into new matrix
    for(i = 0; i < n; i++)
    {
        gsl_matrix_get_col(helpVec, mat, i);
        gsl_matrix_set_col(matOnes, i + 1, helpVec);
    }

    // Deallocate old matrix and help vector
    gsl_matrix_free(mat);
    gsl_vector_free(helpVec);

    // Return new matrix
    return matOnes;
}

/* Function for adding a one
   to a vector */
gsl_vector *addOne(gsl_vector *vec, int m)
{
    // Variable for iteration
    int i;

    // Allocate memory for new vector
    gsl_vector *vecOne = gsl_vector_alloc(m + 1);

    // Add one to beginning of vector
    gsl_vector_set(vecOne, 0, 1);

    // Copy old vector into new vector
    for(i = 0; i < m; i++)
    {
        gsl_vector_set(vecOne, i + 1,
                        gsl_vector_get(vec, i));
    }

    // Deallocate old vector
    gsl_vector_free(vec);

    // Return new vector
    return vecOne;
}

/* Function for calculating the sigmoid of
   every element of a vector */
gsl_vector *sigmoid(gsl_vector *vec, int m)
{
    // Declare iteration variable
    int i;

    // Iterate over vector length anc
    // calculate sigmoid
    for(i = 0; i < m; i++)
    {
        gsl_vector_set(vec, i,
                       1/(1 + exp(-gsl_vector_get(vec, i))));
    }

    return vec;
}

/* Function for parsing a file and getting
   the weights of the neural network */
gsl_matrix *getWeights(FILE *f, int numRow, int numCol)
{
    // Variables for iteration
    int j;

    // Variable for holding one weight
    double weight;

    // Allocate memory for weight matrix
    gsl_matrix *weights = gsl_matrix_alloc(numRow, numCol);

    // Read the weights from file
    for(j = 0; j <  numRow * numCol; j++)
    {
        // Read one weight from file
        fscanf(f, "%lf", &weight);

        // Add to appropriate matrix location
        gsl_matrix_set(weights,
                       j/numCol,
                       j % numCol,
                       weight);
    }

    // Return pointer to weight matrix
    return weights;
}

void printFlower(gsl_vector *vec)
{
    // Iterator variable
    int i;

    // Variables for probabilities
    double a, b, c, max;

    // Get values
    a = gsl_vector_get(vec, 0);
    b = gsl_vector_get(vec, 1);
    c = gsl_vector_get(vec, 2);

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

/* Function that returns the number of layers of
   the neural network */
int getNumLayers()
{
    // Variable for holding number of layers
    int numLayers;

    // Open file that contains the weights
    FILE *f = fopen("./weights.txt", "r");

    // Get number of layers
    fscanf(f, "%d", &numLayers);

    // Close file
    fclose(f);

    // Return number of layers
    return numLayers;
}

/* Function that propagates a sample vector
   to the output of the neural network */
/*gsl_vector */void forwardProp(gsl_vector *x, gsl_matrix *weights[],
                        int numLayers, int units[])
{
    // Declare iteration variable
    int i;

    // Allocate memory for all intermediate results
    gsl_vector *helpVecs[numLayers - 1];

    for(i = 0; i < numLayers; i++)
    {
        helpVecs[i] = gsl_vector_alloc(units[i]);
    }

    // Set x as input vector
    gsl_vector_memcpy(helpVecs[0], x);

    // Deallocate old vector
    gsl_vector_free(x);

    // Iterate over layers
    for(i = 0; i < numLayers - 1; i++)
    {

    // Add one to input vector, for bias term
    helpVecs[i] = addOne(helpVecs[i], units[i]);

    // Propagate forward to next layer
    gsl_blas_dgemv(CblasNoTrans, 1, weights[i], helpVecs[i], 0, helpVecs[i + 1]);

    // Calculate sigmoid activation (except for last layer)
//    if(i < numLayers - 2)
//    {
        helpVecs[i + 1] = sigmoid(helpVecs[i + 1], units[i+ 1]);
//    }

    // Deallocate previous vector
    gsl_vector_free(helpVecs[i]);
    }

    // Print predicted values
    printf("Output probabilities: %f, %f, %f\n",
       gsl_vector_get(helpVecs[2], 0),
       gsl_vector_get(helpVecs[2], 1),
       gsl_vector_get(helpVecs[2], 2));

    // Print flower name
    printFlower(helpVecs[2]);

    // Return new vector
    //return helpVecs[numLayers - 1];
}
