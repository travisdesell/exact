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

void RNN_Node::input_fired(const vector<double> &incoming_outputs) {
    inputs_fired++;

    for (uint32_t i = 0; i < input_values.size(); i++) {
        input_values[i] += incoming_outputs[i];
    }

    if (inputs_fired < total_inputs) return;
    else if (inputs_fired > total_inputs) {
        cerr << "ERROR: inputs_fired on RNN_Node " << innovation_number << " is " << inputs_fired << " and total_inputs is " << total_inputs << endl;
        exit(1);
    }

    //bound_value(input_value);
    //cerr << "activating RNN node: " << innovation_number << ", type: " << type << endl;

    for (uint32_t i = 0; i < input_values.size(); i++) {
        //output_values[i] = input_values[i] + bias;
        //ld_output[i] = 1.0;

        output_values[i] = tanh(input_values[i] + bias);
        ld_output[i] = tanh_derivative(output_values[i]);

        //output_values[i] = sigmoid(input_values[i] + bias);
        //ld_output[i] = sigmoid_derivative(output_values[i]);

#ifdef NAN_CHECKS
        if (isnan(output_values[i]) || isinf(output_values[i])) {
            cerr << "ERROR: output_value[" << i << "] became " << output_values[i] << " on RNN node: " << innovation_number << endl;
            cerr << "\tinput_value[" << i << "]: " << input_values[i] << endl;
            cerr << "\tnode bias: " << bias << endl;
            exit(1);
        }
#endif
    }
}

void RNN_Node::try_update_deltas() {
    if (outputs_fired < total_outputs) return;
    else if (outputs_fired > total_outputs) {
        cerr << "ERROR: outputs_fired on RNN_Node " << innovation_number << " is " << outputs_fired << " and total_outputs is " << total_outputs << endl;
        exit(1);
    }

    for (uint32_t i = 0; i < series_length; i++) {
        d_input[i] *= ld_output[i];
    }

    d_bias = 0.0;
    for (uint32_t i = 0; i < series_length; i++) {
        d_bias += d_input[i];
    }
}

void RNN_Node::output_fired(double error) {
    outputs_fired++;

    for (uint32_t i = 0; i < series_length; i++) {
        d_input[i] *= error;
    }

    try_update_deltas();
}

void RNN_Node::output_fired(const vector<double> &deltas) {
    outputs_fired++;

    for (uint32_t i = 0; i < series_length; i++) {
        d_input[i] += deltas[i];
    }

    try_update_deltas();
}

void RNN_Node::reset(int _series_length) {
    series_length = _series_length;

    ld_output.assign(series_length, 0.0);
    d_input.assign(series_length, 0.0);
    input_values.assign(series_length, 0.0);
    output_values.assign(series_length, 0.0);

    inputs_fired = 0;
    outputs_fired = 0;
}

void RNN_Node::get_gradients(vector<double> &gradients) {
    gradients.assign(1, d_bias);
}

uint32_t RNN_Node::get_number_weights() {
    return 1;
}

void RNN_Node::get_weights(uint32_t &offset, vector<double> &parameters) {
    //no weights to set in a basic RNN node, only a bias
    parameters[offset++] = bias;
}

void RNN_Node::set_weights(uint32_t &offset, const vector<double> &parameters) {
    //no weights to set in a basic RNN node, only a bias
    bias = parameters[offset++];
}

RNN_Node_Interface* RNN_Node::copy() {
    RNN_Node* n = new RNN_Node(innovation_number, type);

    //copy RNN_Node values
    n->bias = bias;
    n->d_bias = d_bias;
    n->ld_output = ld_output;

    //copy RNN_Node_Interface values
    n->series_length = series_length;
    n->input_values = input_values;
    n->output_values = output_values;
    n->error_values = error_values;
    n->d_input = d_input;

    n->inputs_fired = inputs_fired;
    n->total_inputs = total_inputs;
    n->outputs_fired = outputs_fired;
    n->total_outputs = total_outputs;

    return n;
}
