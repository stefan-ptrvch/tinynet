#ifndef INPUT_H
#define INPUT_H

namespace tiny
{
    class Input
    {
        public:

            Input(){};

            // Constructor takes input layer size
            Input(int size) { m_output_size = size; }

            // Get size of layer
            int& get_output_size() { return m_output_size; }

        private:

            // Input layer size
            int m_output_size;
    };
}

#endif
