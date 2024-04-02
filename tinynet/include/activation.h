#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <armadillo>
#include <string>

namespace tiny
{
    class Activation
    {
        public:

            // Constructor takes type of activation funcion
            Activation();

            Activation(const std::string& type);

            // Function for calling activation function
            void activate(const arma::fcolvec& input, arma::fcolvec& output);

        private:

            // For holding the type of activation
            std::string m_type;

            // Sigmoid activation function
            void sigmoid(const arma::fcolvec& input, arma::fcolvec& activation);

            // Linear activation function
            void linear(const arma::fcolvec& input, arma::fcolvec& activation);
    };
}

#endif
