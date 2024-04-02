#include "cost.h"

using namespace tiny;

Cost::Cost(std::string name)
{
    // Set name of cost function to be used
    m_cost_function_name = name;
}

float Cost::evaluate(const arma::fcolvec& output_of_model,
                     const arma::fcolvec& target_values)
{
    // Check which cost function should be called
    if (m_cost_function_name == "squared error")
    {
        return squared_error(output_of_model, target_values);

    }

    else if (m_cost_function_name == "cross entropy")
    {
        return cross_entropy(output_of_model, target_values);
    }

    // Default cost is cross entropy error
    else
    {
        return cross_entropy(output_of_model, target_values);
    }
}

float Cost::squared_error(const arma::fcolvec& output_of_model,
                          const arma::fcolvec& target_values)
{
    // Return average squared error of model prediction and target values
    return 1.0/(2.0*target_values.n_elem)*arma::sum(arma::pow(output_of_model -
                                                        target_values, 2));
}

float Cost::cross_entropy(const arma::fcolvec& output_of_model,
                          const arma::fcolvec& target_values)
{
    // Return average cross entropy error of model prediction and target
    // values
    return -1.0/(2.0*target_values.n_elem)*arma::sum(target_values % arma::log(output_of_model) +
            (1.0 - target_values) % arma::log(1.0 - output_of_model));
}

