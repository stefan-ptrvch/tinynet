#ifndef GRADIENT_H
#define GRADIENT_H

#include <string>
#include <vector>
#include <armadillo>
#include "net.h"
#include "cost.h"

namespace tiny
{
    class Gradient
    {
        public:

            // Constructor takes name of gradient calculation algorithm
            Gradient(std::string name, Net& model);

            // Calculate all gradients based on input sample
            std::vector<arma::fmat> calculate(arma::fcolvec input_data,
                                              arma::fcolvec target_values);

            // Set cost function
            void set_cost(const Cost& cost);

        private:

            // Algorithm name
            std::string m_name;

            // Gradient checking algorithm
            std::vector<arma::fmat> gradient_checking(arma::fcolvec input_data,
                                                      arma::fcolvec target_values);

            // Backpropagation algorithm
            std::vector<arma::fmat> backprop(arma::fcolvec input_data,
                                             arma::fcolvec target_values);

            // A reference to the model
            Net& m_model;

            // Cost function
            Cost m_cost;
    };
}

#endif
