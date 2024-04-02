#include "optimizer.h"
#include <math.h>

using namespace tiny;

Optimizer::Optimizer(std::string name, int batch_size, int num_epoch,
                     Gradient& gradient, Cost& cost_function,
                     arma::fmat& input_data, arma::fmat& target_values) :
                     m_gradient(gradient), m_cost_function(cost_function),
                     m_input_data(input_data), m_target_values(target_values)

{
    // Set name of optimizer
    m_optimizer_name = name;

    // Set batch size
    m_batch_size = batch_size;

    // Set number of epochs
    m_num_epoch = num_epoch;
}

void Optimizer::optimize(Net& model, float learning_rate)
{
    // Optimize model with algorithm that was chosen during object
    // initialization
    if (m_optimizer_name == "mini batch")
        mini_batch(model, learning_rate, m_batch_size);

    // Default optimization algorithm
    else
        mini_batch(model, learning_rate, m_batch_size);
}

void Optimizer::mini_batch(Net& model, float learning_rate)
{
    using namespace std;

    // This is actually just gradient descent, not using batches but the whole
    // dataset
    // This function is used to test the gradient checking method

    // Number of samples
    int num_samples = m_input_data.n_rows;

    // Variable for holding a single input sample
    arma::fcolvec input_sample;

    // Variable for cost
    float cost = 0;

    // Vector of matrices for holding gradient for the model per epoch
    std::vector<arma::fmat> gradients;

    // Vector of matrices for holding the cumulative gradient of all samples per
    // epoch
    std::vector<arma::fmat> cumulative_gradient(model.get_num_layers());

    // Initialize the cumulative gradient to zero
    for (unsigned int i = 0; i < cumulative_gradient.size(); i++)
    {
        // Reshape cumulative gradient matrix to fit gradient matrix
        if (i == 0)
        {
            cumulative_gradient[i].reshape(model.get_num_neurons(0),
                                           model.get_input_size() + 1);
        }

        else
        {
            cumulative_gradient[i].reshape(model.get_num_neurons(i),
                                           model.get_num_neurons(i - 1) + 1);
        }

        // Fill all gradient matrices with zeros
        cumulative_gradient[i].fill(0);
    }

    // Predictions per epoch
    arma::fcolvec prediction;

    // Terget per epoch
    arma::fcolvec target_sample;

    // Accuracy per epoch
    float accuracy = 0;

    // Iterate number of epochs
    for (int i = 0; i < m_num_epoch; i++)
    {
        // Predict on current weights, for current epoch, to get the cost (the
        // error) for this epoch
        for (int j = 0; j < num_samples; j++)
        {
            // Get one input sample from the input sample matrix (one row)
            input_sample = m_input_data.row(j).t();

            // Predict on the extracted input sample
            prediction = model.predict(input_sample);

            // Get the corresponding target sample from the target sample
            // matrix (one row)
            target_sample = m_target_values.row(j);

            // Calculate the cost for the current sample, and add it to the
            // cumulative cost of the whole dataset for this epoch
            cost += m_cost_function.evaluate(prediction, target_sample);

            // Calculate the gradient for the current sample, for the model of
            // the current epoch
            gradients = m_gradient.calculate(input_sample, target_sample);

            // Add the gradient for current sample to cumulative gradient
            for (unsigned int k = 0; k < gradients.size(); k++)
            {
                // Iterate over layers and add gradient for every layer to
                // respective cumulative gradient matrix
                cumulative_gradient[k] += gradients[k];
            }

            // Calculate the accuracy per epoch (works only for binary
            // classification)
            if ((prediction.at(0) >= 0.5 and target_sample.at(0) == 1) or
                (prediction.at(0) < 0.5 and target_sample.at(0) == 0))
            {
                accuracy += 1;
            }
        }

        // Calculate average gradient
        for (unsigned int k = 0; k < cumulative_gradient.size(); k++)
        {
            // Iterate over layers and divide gradient for every weight by the
            // number of samples
            cumulative_gradient[k] /= num_samples;
        }

        // Calculate average cost
        cost /= num_samples;

        // Calculate accuracy
        accuracy /= num_samples;

        // Iterate over layers and update weights in each layer
        for (unsigned int k = 0; k < cumulative_gradient.size();  k++)
        {
            model.update_weights(k, -learning_rate*cumulative_gradient[k]);
        }

        // Print average cost for this epoch
        cout << "Epoch " << i << " cost: " << cost;
        cout << endl;

        // Print the accuracy for this epoch
        cout << "Epoch " << i << " accuracy: " << accuracy*100 << "%";
        cout << endl;

        // Set cumulative gradient back to zero, for next epoch
        for (unsigned int k = 0; k < cumulative_gradient.size(); k++)
        {
            // Set all matrices in to zero
            cumulative_gradient[k].fill(0);
        }

        // Set cost back to zero, for next epoch
        cost = 0;

        // Set accuracy back to zero, for next epoch
        accuracy = 0;
    }
}

void Optimizer::mini_batch(Net& model, float learning_rate, int batch_size)
{
    using namespace std;

    // Number of samples
    int num_samples = m_input_data.n_rows;

    // Number of features
    int num_features = m_input_data.n_cols;

    // Number of classes
    int num_classes = m_target_values.n_cols;

    // Variable for holding a single input sample
    arma::fcolvec input_sample;

    // Variable for cost
    float cost = 0;

    // Vector of matrices for holding gradient for the model per epoch
    std::vector<arma::fmat> gradients;

    // Vector of matrices for holding the cumulative gradient of all samples per
    // epoch
    std::vector<arma::fmat> cumulative_gradient(model.get_num_layers());

    // Initialize the cumulative gradient to zero
    for (unsigned int i = 0; i < cumulative_gradient.size(); i++)
    {
        // Reshape cumulative gradient matrix to fit gradient matrix
        if (i == 0)
        {
            cumulative_gradient[i].reshape(model.get_num_neurons(0),
                                           model.get_input_size() + 1);
        }

        else
        {
            cumulative_gradient[i].reshape(model.get_num_neurons(i),
                                           model.get_num_neurons(i - 1) + 1);
        }

        // Fill all gradient matrices with zeros
        cumulative_gradient[i].fill(0);
    }

    // Predictions per epoch
    arma::fcolvec prediction;

    // Terget per epoch
    arma::fcolvec target_sample;

    // Accuracy per epoch
    float accuracy = 0;

    // Matrix for shuffling input samples and target samples for every epoch
    arma::fmat shufflematrix;

    // Reshape shufflematrix, to hold all samples
    shufflematrix.reshape(num_samples, num_features + num_classes);

    // Put samples into shufflematrix
    shufflematrix.cols(0, num_features - 1) = m_input_data;
    shufflematrix.cols(num_features,
                       num_features + num_classes - 1) = m_target_values;

    // Iterate number of epochs
    for (int i = 0; i < m_num_epoch; i++)
    {
        // Shullfe the input data and target values, for current epoch
        shufflematrix = arma::shuffle(shufflematrix, 0);

        // Iterate over number of iterations, for this epoch, disregard last
        // incomplete batch
        for (int k = 0; k < floor(num_samples/batch_size); k++ )
        {
            // Calculate gradients for the current batch, and update the model
            for (int j = 0; j < batch_size; j++)
            {
                // Get one input sample from the input sample matrix (one row)
                input_sample = shufflematrix.row(k*batch_size +
                                                 j).cols(0, num_features -
                                                         1).t();

                // Get the corresponding target sample from the target sample
                // matrix (one row)
                target_sample = shufflematrix.row(k*batch_size +
                                                  j).cols(num_features,
                                                          num_features +
                                                          num_classes - 1);

                // Calculate the gradient for the current sample, for the model
                // of the current epoch
                gradients = m_gradient.calculate(input_sample, target_sample);

                // Add the gradient for current sample to cumulative gradient
                for (unsigned int k = 0; k < gradients.size(); k++)
                {
                    // Iterate over layers and add gradient for every layer to
                    // respective cumulative gradient matrix
                    cumulative_gradient[k] += gradients[k];
                }
            }

            // Calculate average gradient for this iteration and update weights
            for (unsigned int k = 0; k < cumulative_gradient.size(); k++)
            {
                // Iterate over layers and divide gradient for every weight by
                // the number of samples
                cumulative_gradient[k] /= batch_size;

                // Update weights
                model.update_weights(k, -learning_rate*cumulative_gradient[k]);
            }

            // Set cumulative gradient back to zero, for next epoch
            for (unsigned int k = 0; k < cumulative_gradient.size(); k++)
            {
                // Set all matrices in to zero
                cumulative_gradient[k].fill(0);
            }
        }

        // Predict on current weights, for current epoch, to get the cost (the
        // error) for this epoch
        for (int j = 0; j < num_samples; j++)
        {
            // Get one input sample from the input sample matrix (one row)
            input_sample = m_input_data.row(j).t();

            // Predict on the extracted input sample
            prediction = model.predict(input_sample);

            // Get the corresponding target sample from the target sample
            // matrix (one row)
            target_sample = m_target_values.row(j);

            // Calculate the cost for the current sample, and add it to the
            // cumulative cost of the whole dataset for this epoch
            cost += m_cost_function.evaluate(prediction, target_sample);

            // Calculate the accuracy per epoch (works only for binary
            // classification)
            if ((prediction.at(0) >= 0.5 and target_sample.at(0) == 1) or
                (prediction.at(0) < 0.5 and target_sample.at(0) == 0))
            {
                accuracy += 1;
            }
        }

        // Calculate average cost
        cost /= num_samples;

        // Calculate accuracy
        accuracy /= num_samples;

        // Print average cost for this epoch
        cout << "Epoch " << i << " cost: " << cost;
        cout << endl;

        // Print the accuracy for this epoch
        cout << "Epoch " << i << " accuracy: " << accuracy*100 << "%";
        cout << endl;

        // Set cost back to zero, for next epoch
        cost = 0;

        // Set accuracy back to zero, for next epoch
        accuracy = 0;
    }
}
