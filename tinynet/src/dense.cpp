#include "dense.h"

using namespace tiny;

Dense::Dense(int num_outputs)
{
    // Set output size
    m_output_size = num_outputs;
}

void Dense::set_input(const arma::fcolvec& input)
{
    // Set input into layer
    m_input = input;
}

arma::fcolvec& Dense::get_output()
{
    // Return activated values
    return m_activation;
}

void Dense::propagate()
{
    // Calculate input into layer
    m_pre_activation = m_weights.cols(1, m_weights.n_cols - 1)*m_input +
        m_weights.col(0);

    // Calculate output of layer
    activate();
}

void Dense::activate()
{
    m_activation_function.activate(m_pre_activation, m_activation);
}

void Dense::init_weights()
{
    m_weights.randn();
}

void Dense::set_input_size(int input_size)
{
    m_input_size = input_size;
}

int Dense::get_output_size()
{
    return m_output_size;
}

void Dense::set_activation(Activation activation)
{
    // Set activaion function
    m_activation_function = activation;
}

void Dense::build()
{
    // Set size of pre-activation vector
    m_pre_activation.resize(m_output_size);

    // Set size of activation vector
    m_activation.resize(m_output_size);

    // Set size of weight matrix
    m_weights.resize(m_output_size, m_input_size + 1);

    // Initialize weights to random values
    init_weights();
}

void Dense::update_weights(arma::fmat update)
{
    // Update layer weights with update value
    m_weights += update;
}

arma::fcolvec* Dense::get_activations()
{
    // Return the activations of this layer
    return &m_activation;
}

arma::fmat* Dense::get_weights()
{
    // Return the weights of this layer
    return &m_weights;
}
