#ifndef DENSE_H
#define DENSE_H

#include <armadillo>
#include <string>
#include "activation.h"

namespace tiny
{
    class Dense
    {
        public:

            // Class constructor, needs layer shape
            Dense(int outputs);

            // Set input into layer
            void set_input(const arma::fcolvec& input);

            // Set input size of layer
            void set_input_size(int input_size);

            // Get output size of layer
            int get_output_size();

            // Set activaton function of layer
            void set_activation(Activation activation);

            // Get output of layer
            arma::fcolvec& get_output();

            // Function that propagates data through the current layer
            void propagate();

            // Initialize all member variables
            void build();

            // Update weights
            void update_weights(arma::fmat update);

            // Get activations
            arma::fcolvec* get_activations();

            // Get weights
            arma::fmat* get_weights();

        private:

            // Pre-activation values of current layer
            arma::fcolvec m_pre_activation;

            // Input values of current layer
            arma::fcolvec m_input;

            // Activations of current layer
            arma::fcolvec m_activation;

            // The weight matrix of the current layer
            arma::fmat m_weights;

            // Delta values of current layer
            arma::fcolvec m_delta;

            // Activation function
            Activation m_activation_function;

            // Initialize weights to random values
            void init_weights();

            // Function for activating all pre-activations of current layer
            void activate();

            // Input size of layer
            int m_input_size;

            // Ouput size of layer
            int m_output_size;
    };
}

#endif
