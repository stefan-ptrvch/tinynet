#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include <string>
#include "net.h"
#include "cost.h"
#include "gradient.h"

namespace tiny
{
    class Optimizer
    {
        public:

            // Constructor takes algorithm name as argument
            Optimizer(std::string name, int batch_size, int num_epoch,
                      Gradient& gradient, Cost& cost_function,
                      arma::fmat& input_data, arma::fmat& target_values);

            // Function for calling selected optimizer
            void optimize(Net& model, float learning_rate);

        private:

            // Optimization algorithm name
            std::string m_optimizer_name;

            // Batch size
            int m_batch_size;

            // Number of epochs for algorithm
            int m_num_epoch;

            // Gradient function
            Gradient& m_gradient;

            // Cost function
            Cost& m_cost_function;

            // Input data
            arma::fmat& m_input_data;

            // Target values
            arma::fmat& m_target_values;

            // Print cost to output
            void print_cost();

            // Mini-batch gradient descent
            void mini_batch(Net& model, float learning_rate);

            void mini_batch(Net& model, float learning_rate, int batch_size);
    };
}

#endif
