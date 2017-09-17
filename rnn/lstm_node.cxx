#include <cmath>

#include <iostream>
using std::cerr;
using std::endl;

#include <vector>
using std::vector;

#include "rnn_node_interface.hxx"
#include "lstm_node.hxx"


LSTM_Node::LSTM_Node(int _innovation_number, int _type) : RNN_Node_Interface(_innovation_number, _type) {
}

void LSTM_Node::input_fired() {
    inputs_fired++;

    if (inputs_fired < total_inputs) return;
    else if (inputs_fired > total_inputs) {
        cerr << "ERROR: inputs_fired on LSTM_Node " << innovation_number << " is " << inputs_fired << " and total_inputs is " << total_inputs << endl;
        exit(1);
    }

    //cerr << "activating LSTM node: " << innovation_number << ", type: " << type << endl;

    input_gate_value = sigmoid(input_gate_weight * input_value + input_gate_update_weight * previous_cell_value + input_gate_bias);
    output_gate_value = sigmoid(output_gate_weight * input_value + output_gate_update_weight * previous_cell_value + output_gate_bias);
    forget_gate_value = sigmoid(forget_gate_weight * input_value + forget_gate_update_weight * previous_cell_value + forget_gate_bias);

    bound_value(input_gate_value);
    bound_value(output_gate_value);
    bound_value(forget_gate_value);

    previous_cell_value = cell_value;
    cell_value = forget_gate_value * previous_cell_value + input_gate_value * tanh(cell_weight * input_value + cell_bias);

    bound_value(cell_value);

    //The original is a hyperbolic tangent, but the peephole[clarification needed] LSTM paper suggests the activation function be linear -- activation(x) = x

    output_value = output_gate_value * tanh(cell_value);
    //output = output_gate_value * sigmoid(cell_value);
    //output = output_gate_value * sigmoid(cell_value);

    if (isnan(output_value) || isinf(output_value)) {
        cerr << "ERROR: output_value became " << output_value << " on LSTM node: " << innovation_number << endl;
        print_cell_values();
        exit(1);
    }

    if (isnan(forget_gate_value) || isinf(forget_gate_value)) {
        cerr << "ERROR: forget_gate_value became " << forget_gate_value << " on LSTM node: " << innovation_number << endl;
        print_cell_values();
        exit(1);
    }

    if (isnan(input_gate_value) || isinf(input_gate_value)) {
        cerr << "ERROR: input_gate_value became " << input_gate_value << " on LSTM node: " << innovation_number << endl;
        print_cell_values();
        exit(1);
    }

    //if (type == RNN_OUTPUT_NODE) cerr << "output: " << output_value << endl;
}

void LSTM_Node::print_cell_values() {
    cerr << "\tinput_value: " << input_value << endl;
    cerr << "\tinput_gate_value: " << input_gate_value << ", input_gate_update_weight: " << input_gate_update_weight << ", input_gate_bias: " << input_gate_bias << endl;
    cerr << "\toutput_gate_value: " << output_gate_value << ", output_gate_update_weight: " << output_gate_update_weight << ", output_gate_bias: " << output_gate_bias << endl;
    cerr << "\tforget_gate_value: " << forget_gate_value << ", forget_gate_update_weight: " << forget_gate_update_weight << "\tforget_gate_bias: " << forget_gate_bias << endl;
    cerr << "\tcell_value: " << cell_value << ", previous_cell_value: " << previous_cell_value << ", cell_bias: " << cell_bias << endl;
    exit(1);
}

void LSTM_Node::output_fired() {
    //this gonna be fun

    outputs_fired++;

    if (outputs_fired < total_outputs) return;
    else if (outputs_fired > total_outputs) {
        cerr << "ERROR: outputs_fired on LSTM_Node " << innovation_number << " is " << outputs_fired << " and total_outputs is " << outputs_fired << endl;
        exit(1);
    }

}

uint32_t LSTM_Node::get_number_weights() {
    return 11;
}

void LSTM_Node::set_weights(uint32_t &offset, const vector<double> &parameters) {
    //uint32_t start_offset = offset;

    input_gate_weight = parameters[offset++];
    output_gate_weight = parameters[offset++];
    forget_gate_weight = parameters[offset++];
    cell_weight = parameters[offset++];

    input_gate_update_weight = parameters[offset++];
    output_gate_update_weight = parameters[offset++];
    forget_gate_update_weight = parameters[offset++];

    input_gate_bias = parameters[offset++];
    output_gate_bias = parameters[offset++];
    forget_gate_bias = parameters[offset++];
    cell_bias = parameters[offset++];

    //uint32_t end_offset = offset;

    //cerr << "set weights from offset " << start_offset << " to " << end_offset << " on LSTM_Node " << innovation_number << endl;
}

void LSTM_Node::reset() {
    input_value = 0.0;
    output_value = 0.0;

    inputs_fired = 0;
    outputs_fired = 0;
}

void LSTM_Node::full_reset() {
    input_value = 0.0;
    output_value = 0.0;

    input_gate_value = 0.0;
    output_gate_value = 0.0;
    forget_gate_value = 0.0;
    cell_value = 0.0;
    previous_cell_value = 0.0;

    inputs_fired = 0;
    outputs_fired = 0;
}
