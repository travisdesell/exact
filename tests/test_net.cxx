#include <cmath>

#include <iomanip>
using std::setw;

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "neural_network.hxx"
#include "node.hxx"
#include "edge.hxx"

int main(int argc, char **argv) {
    NeuralNetwork *neural_network = new NeuralNetwork();

    for (uint32_t i = 0; i < 10000; i++) {
        neural_network->forward_pass();
        neural_network->backward_pass();
        neural_network->update_weights();
    }
}
