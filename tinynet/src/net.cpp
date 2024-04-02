#include "net.h"

using namespace tiny;

void Net::add(const Dense& layer)
{
    // Add layer to net
    m_layers.push_back(layer);
}

void Net::add(const Activation& activation)
{
    // Add layer to net
    m_activations.push_back(activation);
}

void Net::add(const Input& input_layer)
{
    m_input_layer = input_layer;
}

arma::fcolvec Net::predict(arma::fcolvec& input)
{
    // Set input into first layer
    m_layers[0].set_input(input);
    m_layers[0].propagate();

    // Iterate over layers and activations and propagate data thourh them
    for (unsigned int i = 0; i < m_layers.size() - 1; i++)
    {
        m_layers[i + 1].set_input(m_layers[i].get_output());
        m_layers[i + 1].propagate();
    }

    return m_layers.back().get_output();
}

void Net::build()
{
    // Set the number of layers in the network
    m_num_layers = m_layers.size();

    // Set input size and activation of first hidden layer
    m_layers[0].set_input_size(m_input_layer.get_output_size());
    m_layers[0].set_activation(m_activations[0]);
    m_layers[0].build();

    // Set layer sizes and activation functions of other layers
    for (unsigned int i = 0; i < m_layers.size() - 1; i++)
    {
        m_layers[i + 1].set_input_size(m_layers[i].get_output_size());
        m_layers[i + 1].set_activation(m_activations[i + 1]);
        m_layers[i + 1].build();
    }

}

void Net::update_weights(int layer_number, arma::fmat update)
{
    // Update weights of given layer (not counting input layer)
    m_layers[layer_number].update_weights(update);
}

int Net::get_num_neurons(int layer_number)
{
    // Return number of neurons
    return m_layers[layer_number].get_output_size();
}

int Net::get_input_size()
{
    // Return size of input layer
    return m_input_layer.get_output_size();
}

int Net::get_num_layers()
{
    // Return number of layers in network
    return m_num_layers;
}

std::vector<arma::fmat*> Net::get_weights()
{
    // Vector of pointers to matrices
    std::vector<arma::fmat*> weights;

    // Iterate through all layers and get weight matrices
    for (int i = 0; i < m_num_layers; i++)
    {
        weights.push_back(m_layers[i].get_weights());
    }

    // Return vector of matrix pointers
    return weights;
}

std::vector<arma::fcolvec*> Net::get_activations()
{
    // Vector of pointers to vectors
    std::vector<arma::fcolvec*> activations;

    // Iterate through all layers and get activation vectors
    for (int i = 0; i < m_num_layers; i++)
    {
        activations.push_back(m_layers[i].get_activations());
    }

    // Return vecotr of vector pointers
    return activations;
}
