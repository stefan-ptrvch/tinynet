#ifndef NET_H
#define NET_H

#include <armadillo>
#include <vector>

#include "dense.h"
#include "input.h"
#include "activation.h"

namespace tiny
{
    class Net
    {
        public:

            // Constructor
            Net(){};

            // A function which adds a Dense layer
            void add(const Dense& layer);

            // A function which adds an Input layer
            void add(const Input& layer);

            // A function which adds an activation
            void add(const Activation& activation);

            // Propagate data throught whole network
            arma::fcolvec predict(arma::fcolvec& input);

            // Build the network
            void build();

            // Get number of layers
            int get_num_layers();

            // Get number of neurons in layer
            int get_num_neurons(int layer_number);

            // Get input size of network
            int get_input_size();

            // Update weights of layer in network
            void update_weights(int layer_number, arma::fmat update);

            // Get weights of all layers
            std::vector<arma::fmat*> get_weights();

            // Get activations of all layers
            std::vector<arma::fcolvec*> get_activations();

        private:

            // Vector holding layers
            std::vector<Dense> m_layers;

            // Vecotr holding activations
            std::vector<Activation> m_activations;

            // Input layer
            Input m_input_layer;

            // Number of layers
            int m_num_layers;
    };
}

#endif
