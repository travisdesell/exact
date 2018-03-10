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

void RNN_Node::input_fired(int time, double incoming_output) {
    inputs_fired[time]++;

    input_values[time] += incoming_output;

    if (inputs_fired[time] < total_inputs) return;
    else if (inputs_fired[time] > total_inputs) {
        cerr << "ERROR: inputs_fired on RNN_Node " << innovation_number << " at time " << time << " is " << inputs_fired[time] << " and total_inputs is " << total_inputs << endl;
        exit(1);
    }


    output_values[time] = tanh(input_values[time] + bias);
    ld_output[time] = tanh_derivative(output_values[time]);

    //output_values[time] = sigmoid(input_values[time] + bias);
    //ld_output[time] = sigmoid_derivative(output_values[time]);

#ifdef NAN_CHECKS
    if (isnan(output_values[time]) || isinf(output_values[time])) {
        cerr << "ERROR: output_value[" << time << "] became " << output_values[time] << " on RNN node: " << innovation_number << endl;
        cerr << "\tinput_value[" << time << "]: " << input_values[time] << endl;
        cerr << "\tnode bias: " << bias << endl;
        exit(1);
    }
#endif
}

void RNN_Node::try_update_deltas(int time) {
    if (outputs_fired[time] < total_outputs) return;
    else if (outputs_fired[time] > total_outputs) {
        cerr << "ERROR: outputs_fired on RNN_Node " << innovation_number << " at time " << time << " is " << outputs_fired[time] << " and total_outputs is " << total_outputs << endl;
        exit(1);
    }

    d_input[time] *= ld_output[time];

    d_bias += d_input[time];
}

void RNN_Node::error_fired(int time, double error) {
    outputs_fired[time]++;

    d_input[time] *= error;

    try_update_deltas(time);
}

void RNN_Node::output_fired(int time, double delta) {
    outputs_fired[time]++;

    d_input[time] += delta;

    try_update_deltas(time);
}

void RNN_Node::reset(int _series_length) {
    series_length = _series_length;

    ld_output.assign(series_length, 0.0);
    d_input.assign(series_length, 0.0);
    input_values.assign(series_length, 0.0);
    output_values.assign(series_length, 0.0);

    inputs_fired.assign(series_length, 0);
    outputs_fired.assign(series_length, 0);

    d_bias = 0.0;
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
