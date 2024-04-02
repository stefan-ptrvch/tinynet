#include <iostream>
#include <armadillo>
#include <vector>
#include <stdio.h>
#include <fstream>

#include "tiny.h"

#define ONE

#ifdef ONE

// Function for parsing dataset
arma::fmat read_input(std::string file_name, int num_features,
                                   int num_samples)
{
    using namespace std;

    // Vector for holding input data values
    arma::fmat input_samples(num_samples, num_features);

    // Declare input file
    ifstream input_file;

    // String for holding line
    string line;

    // Open file
    input_file.open(file_name);

    // Variable for keeping track of read data ponts
    int i = 0;
    int row;
    int column;

    // Read number for number
    while (getline(input_file, line, ','))
    {
        // Calculate row
        row = i/num_features;

        // Calculate column
        column = i % num_features;

        // Put the feature into the feature matrix
        input_samples(row, column) = stof(line);

        // Increment number of data points read
        i++;
    }

    // Return feature matrix
    return input_samples;
}

// Function for parsing dataset
arma::fmat read_target(std::string file_name, int num_samples)
{
    using namespace std;

    // Vector for holding input data values
    arma::fmat target_samples(num_samples, 1);

    // Declare target file
    ifstream target_file;

    // String for holding line
    string line;

    // Open file
    target_file.open(file_name);

    // Variable for keeping track of read data ponts
    int i = 0;

    // Read number for number
    while (getline(target_file, line, '\n'))
    {
        // Put the target sample into the target sample vector (and make just
        // two classes, for binary classification)
        if (stof(line) == 1)
        {
            target_samples(i, 0) = stof(line);
        }

        else
        {
            target_samples(i, 0) = 0;
        }

        // Increment number of data points read
        i++;
    }

    // Return target samples
    return target_samples;
}

int main()
{
    // Read feature matrix
    arma::fmat input_data = read_input("./data/input_data.csv", 4, 150);

    // Read target values
    arma::fmat targets = read_target("./data/targets.csv", 150);

    // Define network
    tiny::Net model;

    // Add input layer
    model.add(tiny::Input(4));

    // Add hidden layer
    model.add(tiny::Dense(40));

    // Set activation
    model.add(tiny::Activation("sigmoid"));

    // Add output layer
    model.add(tiny::Dense(1));

    // Add output activation
    model.add(tiny::Activation("sigmoid"));

    // Build network
    model.build();

    // Now...

    // Initialize a gradient calculation object
    tiny::Gradient gradient("gradient checking", model);

    // Make a cost function object
    tiny::Cost cost("cross entropy");

    // Tell the gradient which cost function to use
    gradient.set_cost(cost);

    // Make optimizer
    tiny::Optimizer optimizer("mini batch", 8, 100, gradient, cost,
                              input_data, targets);

    // Cross fingers
    optimizer.optimize(model, 0.05);

    // Predict for all input samples and compare to targets

    // Variable for holding a single input sample
    arma::fcolvec input_sample;

    // Predictions per epoch
    arma::fcolvec prediction;

    // Terget per epoch
    arma::fcolvec target_sample;

    // Number of samples
    int num_samples = input_data.n_rows;

    // Predict on current weights, for current epoch, to get the cost (the
    // error) for this epoch
    for (int j = 0; j < num_samples; j++)
    {
        // Get one input sample from the input sample matrix (one row)
        input_sample = input_data.row(j).t();

        // Predict on the extracted input sample
        prediction = model.predict(input_sample);

        // Get the corresponding target sample from the target sample
        // matrix (one row)
        target_sample = targets.row(j);

        // Print values to output
        std::cout << "Pred:" << prediction
                  << "Targ:" << target_sample << std::endl;

    }
}

#endif

#ifdef TWO

#include "cost.h"
#include "gradient.h"

int main()
{
    // Make a model and try to get weights and activations out

    // Define network
    tiny::Net model;

    // Add input layer
    model.add(tiny::Input(4));

    // Add hidden layer
    model.add(tiny::Dense(10));

    // Set activation
    model.add(tiny::Activation("sigmoid"));

    // Add output layer
    model.add(tiny::Dense(5));

    // Add output activation
    model.add(tiny::Activation("sigmoid"));

    // Build network
    model.build();

    // Vector for holding weights
    std::vector<arma::fmat*> weights;

    weights = model.get_weights();

    // Vector for holding activations
    std::vector<arma::fcolvec*> activations;

    activations = model.get_activations();

    // Go over weight matrices and print them, as well as activations
    for(int i = 0; i < 2; i++)
    {
        weights[i] -> print("Weights:");
        std::cout << std::endl;

        activations[i] -> print("Activations:");
        std::cout << std::endl;
    }
}

#endif
