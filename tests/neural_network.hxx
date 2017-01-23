#ifndef TEST_NEURAL_NETWORK_HXX
#define TEST_NEURAL_NETWORK_HXX

#include <vector>
using std::vector;

#include "node.hxx"
#include "edge.hxx"

class NeuralNetwork {
    public:
        vector< vector<Node*> > nodes;
        vector<Edge*> edges;

        vector<double> o_expected;

        double learning_rate;
        double total_error;

        NeuralNetwork();

        void forward_pass();
        void backward_pass();
        void update_weights();
};
#endif
