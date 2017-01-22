#include <cmath>

#include <iostream>
using std::cout;
using std::endl;

#include "node.hxx"

Node::Node() {
    out = 0.0;
    bias = 0.0;
    error = 0.0;

    dtotal_dout = 0.0;
    dout_dnet = 0.0;
}

Node::Node(double _out, double _bias) {
    out = _out;
    bias = _bias;
}

void Node::fire() {
    for (uint32_t i = 0; i < output_edges.size(); i++) {
        //cout << "adding to output out: " << out << " * " << output_edges[i]->weight << endl;
        output_edges[i]->output->out += out * output_edges[i]->weight;
    }
}

void Node::activation_function() {
    out += bias;
    out = 1.0 / (1.0 + exp(-out));
}
