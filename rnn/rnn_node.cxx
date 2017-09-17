#include <cmath>

#include <iostream>
using std::cerr;
using std::endl;

#include <vector>
using std::vector;

#include "rnn_node.hxx"


RNN_Node::RNN_Node(int _innovation_number, int _type) : RNN_Node_Interface(_innovation_number, _type) {
    if (type == RNN_INPUT_NODE) {
        total_inputs = 1;
    }
}

void RNN_Node::input_fired() {
    inputs_fired++;

    if (inputs_fired < total_inputs) return;
    else if (inputs_fired > total_inputs) {
        cerr << "ERROR: inputs_fired on RNN_Node " << innovation_number << " is " << inputs_fired << " and total_inputs is " << total_inputs << endl;
        exit(1);
    }

    //bound_value(input_value);
    //cerr << "activating RNN node: " << innovation_number << ", type: " << type << endl;
    input_value += bias;

    output_value = tanh(input_value);

    if (isnan(output_value) || isinf(output_value)) {
        cerr << "ERROR: output_value became " << output_value << " on RNN node: " << innovation_number << endl;
        cerr << "\tinput_value: " << input_value << endl;
        exit(1);
    }
}

void RNN_Node::output_fired() {
    outputs_fired++;

    if (outputs_fired < total_outputs) return;
    else if (outputs_fired > total_outputs) {
        cerr << "ERROR: outputs_fired on RNN_Node " << innovation_number << " is " << outputs_fired << " and total_outputs is " << total_outputs << endl;
        exit(1);
    }
}

void RNN_Node::reset() {
    input_value = 0;
    output_value = 0;

    inputs_fired = 0;
    outputs_fired = 0;
}

void RNN_Node::full_reset() {
    input_value = 0.0;
    output_value = 0.0;

    inputs_fired = 0;
    outputs_fired = 0;
}


uint32_t RNN_Node::get_number_weights() {
    return 1;
}

void RNN_Node::set_weights(uint32_t &offset, const vector<double> &parameters) {
    //no weights to set in a basic RNN node
    bias = parameters[offset++];
}
