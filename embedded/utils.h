#include <gsl/gsl_blas.h>

/* Function for adding a column of ones
   to a matrix */
gsl_matrix * addOnes(gsl_matrix * mat, int m, int n);

/* Function for adding a one
   to a vector */
gsl_vector * addOne(gsl_vector * vec, int m);

/* Function for calculating the sigmoid of
   every element of a vector */
gsl_vector * sigmoid(gsl_vector * vec, int m);

/* Function for parsing a file and getting
   the weights of the neural network */
gsl_matrix *getWeights(FILE *f, int numRow, int numCol);

void printFlower(gsl_vector *vec);

/* Function that returns the number of layers of
   the neural network */
int getNumLayers();

/* Function that propagates a sample vector
   to the output of the neural network */
/*gsl_vector */void forwardProp(gsl_vector *x, gsl_matrix *weights[],
                        int numLayers, int units[]);

