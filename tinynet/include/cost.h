#ifndef COST_H
#define COST_H

#include <string>
#include <armadillo>

namespace tiny
{
    class Cost
    {
        public:

            // Default constructor
            Cost(){};

            // Cost function constructor takes function name as argument
            Cost(std::string name);

            // Evaluation of output of model
            float evaluate(const arma::fcolvec& output_of_model,
                           const arma::fcolvec& target_values);

        private:

            // Cost function name
            std::string m_cost_function_name;

            // Squared error cost function
            float squared_error(const arma::fcolvec& output_of_model,
                                const arma::fcolvec& target_values);

            // Cross-entropy cost function
            float cross_entropy(const arma::fcolvec& output_of_model,
                                const arma::fcolvec& target_values);
    };
}

#endif
