#include "activation.h"

using namespace tiny;

Activation::Activation(const std::string& type)
{
    // Set type of function to be called
    m_type = type;
}

Activation::Activation()
{
    // Set type of function to be called
    m_type = "sigmoid";
}

void Activation::activate(const arma::fcolvec& input, arma::fcolvec& output )
{
    // Chcekc which activation function should be called
    if (m_type == "sigmoid")
    {
        sigmoid(input, output);
    }

    else if (m_type == "linear")
    {
        linear(input, output);
    }

    // Default activation
    else
    {
        sigmoid(input, output);
    }
}

void Activation::sigmoid(const arma::fcolvec& input, arma::fcolvec& output)
{
    // Calculate sigmoid of every element of input vector
    output = 1/(1 + arma::exp(-input));
}

void Activation::linear(const arma::fcolvec& input, arma::fcolvec& output)
{
    // Calculate linear activation
    output = input;
}
