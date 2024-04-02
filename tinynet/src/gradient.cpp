#include "gradient.h"

using namespace tiny;

Gradient::Gradient(std::string name, Net& model) : m_model(model)
{
    // Set name of algorithm
    m_name = name;
}

std::vector<arma::fmat> Gradient::calculate(arma::fcolvec input_data,
                                            arma::fcolvec target_values)
{
    // Check which algorithm should be used
    if (m_name == "gradient checking")
    {
        return gradient_checking(input_data, target_values);
    }
    else if (m_name == "backprop")
    {
        return backprop(input_data, target_values);
    }

    // Default is backprop
    else
    {
        return backprop(input_data, target_values);
    }
}

std::vector<arma::fmat> Gradient::gradient_checking(arma::fcolvec input_data,
                                                    arma::fcolvec target_values)
{
    // THIS ALGORITHM IS SUPER SLOW AND SHOULD ONLY BE USED FOR VERIFICATION

    // Declare vector of gradient matrices that correspond to every layer
    std::vector<arma::fmat> gradients;

    // Variables for holding upper and lower nudge of gradient
    float upper_nugde;
    float lower_nudge;

    // Variable for holding gradient
    float gradient;

    // Define epsilon
    float epsilon = 1.0e-5;

    // Define variable for hodling neuron input size
    int neuron_input_size;

    // Matrix for updating single weights
    arma::fmat update;

    // Iterate over layers
    for (int i = 0; i < m_model.get_num_layers(); i++)
    {
        // Resize update matrix for current layer, as well as gradient
        // matrix
        if (i == 0)
        {
            // This is for the first hidden layer, since its size depends on
            // the input layer
            update.resize(m_model.get_num_neurons(0),
                          m_model.get_input_size() + 1);

            gradients.push_back(arma::fmat(m_model.get_num_neurons(0),
                                           m_model.get_input_size() + 1));

            // Set neuron input size
            neuron_input_size = m_model.get_input_size() + 1;
        }

        else
        {
            // This is for all other hidden layers
            update.resize(m_model.get_num_neurons(i),
                          m_model.get_num_neurons(i - 1) + 1);

            gradients.push_back(arma::fmat(m_model.get_num_neurons(i),
                                           m_model.get_num_neurons(i - 1) +
                                           1));

            // Set neuron input size
            neuron_input_size = m_model.get_num_neurons(i - 1) + 1;
        }

        // Iterate over neurons
        for (int j = 0; j < m_model.get_num_neurons(i); j++)
        {
            // Iterate over weights
            for (int k = 0; k < neuron_input_size; k++)
            {
                // Set epsilon to influence that specific weight
                update(j, k) = epsilon;

                // Update weight matrix of that layer
                m_model.update_weights(i, update);

                // Calculate error for this input sample
                upper_nugde = m_cost.evaluate(m_model.predict(input_data),
                                              target_values);

                // Set epsilon to influence that specific weight for lower nudge
                update(j, k) = -2*epsilon;

                // Update weight matrix of that layer
                m_model.update_weights(i, update);

                // Calculate error for this input sample
                lower_nudge = m_cost.evaluate(m_model.predict(input_data),
                                              target_values);

                // Calculate gradient for that weight
                gradient = (upper_nugde - lower_nudge)/(2*epsilon);

                // Insert it into the vector of layer gradients
                gradients[i].at(j, k) = gradient;

                // Put weight back to initial value
                update(j, k) = epsilon;

                m_model.update_weights(i, update);

                // Keep the update matrix clean!
                update(j, k) = 0;
            }
        }
    }

    return gradients;
}

void Gradient::set_cost(const Cost& cost )
{
    // Set cost member value
    m_cost = cost;
}

std::vector<arma::fmat> Gradient::backprop(arma::fcolvec input_data,
                                           arma::fcolvec target_values)
{
    // Vector of matrices for holding the calculated gradients
    std::vector<arma::fmat> gradients;

    // Vector of vectors for holding deltas
    std::vector<arma::fcolvec> deltas;

    // Vector of vectors for holding derivative of activations
    std::vector<arma::fcolvec> der_act;

    // Vector of pointers to weights of model
    std::vector<arma::fmat*> weights = m_model.get_weights();

    // Vector of pointers to activations
    std::vector<arma::fcolvec*> activations;

    // Vector for holding number of neurons per layer
    std::vector<int> num_neurons;

    // Number of layers in model
    int num_layers = m_model.get_num_layers();

    // Add gradient matrices, delta vectors and derivative activation
    // vectors
    for (int i = 0; i < num_layers; i++)
    {
        // Get number of neurons for layer, not counting bias neuron
        num_neurons.push_back(m_model.get_num_neurons(i));

        // Add delta vectors
        deltas.push_back(arma::fcolvec(num_neurons[i] + 1));

        // Add derivative activations
        der_act.push_back(arma::fcolvec(num_neurons[i] + 1));

        // Derivative activations for bias neurons are 1
        der_act[i].at(0) = 1;

        // Add gradient matrices
        if (i == 0)
        {
            // This is for the first hidden layer, since its size depends on
            // the input layer
            gradients.push_back(arma::fmat(num_neurons[i],
                                           m_model.get_input_size() + 1));
        }

        else
        {
            // This is for all other hidden layers
            gradients.push_back(arma::fmat(num_neurons[i],
                                           num_neurons[i - 1] + 1));
        }
    }

    // Delete last derivative activation vector, since the output layer does not
    // need one
    der_act.pop_back();

    // Get delta values of output layer
    deltas.back() = m_model.predict(input_data) - target_values;

    // Get activations after sample was propagated
    activations = m_model.get_activations();

    // Calculate gradients for output layer, without bias neurons
    gradients.back().cols(1, num_neurons.end()[-2]) = deltas.back()
                                            * activations.end()[-2]->t();

    // Set bias neuron gradients
    gradients.back().col(0) = deltas.back();

    // Calculate derivative of activation evaluated at pre-activation value
    for (int i = num_layers - 2; i >= 0; i--)
    {
        // Get derivative of activation evaluated at pre-activation value,
        // except for bias neuron
        // ######## WORKS PORBABLY ONLY FOR SIGMOID #######
        if (not (i == 0))
        {
            der_act[i].tail(num_neurons[i]) = *activations[i]
                                              % (1 - *activations[i]);
        }

        else
        {
            der_act[i].tail(m_model.get_input_size()) = input_data
                                                        % (1 - input_data);
        }
    }

    // Calculate delta values
    for (int i = num_layers - 2; i >= 0; i--)
    {
        // Propagate deltas backwards
        deltas[i] = der_act[i] % (weights[i + 1]->t()
                                  * deltas[i + 1].tail(num_neurons[i + 1]));
    }

    // Calculate gradients
    for (int i = num_layers - 1; i >= 0; i--)
    {
        // Calculate gradients, except for first hidden layer
        if (not ( i == 0))
        {
            // Leaving out bias neurons
            gradients[i].cols(1, num_neurons[i - 1]) = deltas[i].tail(num_neurons[i])
                                                       * activations[i - 1]->t();

            // Calculate for bias neuron
            gradients[i].col(0) = deltas[i].tail(num_neurons[i]);

        }

        // Calculate gradients for first hidden layer
        else
        {
            // Leaving out bias neuron
            gradients[i].cols(1, m_model.get_input_size()) = deltas[i].tail(num_neurons[i])
                                                             * input_data.t();

            // Calculate for bias neuron
            gradients[i].col(0) = deltas[i].tail(num_neurons[i]);
        }
    }

    return gradients;
}
