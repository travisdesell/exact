#include <cmath>

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <vector>
using std::vector;

#include "rnn_node.hxx"



RNN_Node::RNN_Node(int _innovation_number, int _layer_type, double _depth, int _node_type) : RNN_Node_Interface(_innovation_number, _layer_type, _depth) {
    if (layer_type == INPUT_LAYER) {
        total_inputs = 1;
    }
    //cout << "created node: " << innovation_number << ", type: " << type << endl;

    node_type = _node_type;
}

RNN_Node::~RNN_Node() {
}

// RNN_Node(const RNN_Node_Interface& _rnn_node_interface) : RNN_Node_Interface(_rnn_node_interface){
// }

void RNN_Node::initialize_randomly(minstd_rand0 &generator, NormalDistribution &normal_distribution, double mu, double sigma) {
    bias = bound(normal_distribution.random(generator, mu, sigma));
}

void RNN_Node::input_fired(int time, double incoming_output) {
    inputs_fired[time]++;

    input_values[time] += incoming_output;

    cerr << "inputs_fired on RNN_Node " << innovation_number << " at time " << time << " is " << inputs_fired[time] << " and total_inputs is " << total_inputs << endl;

    if (inputs_fired[time] < total_inputs) return;
    else if (inputs_fired[time] > total_inputs) {
        cerr << "ERROR: inputs_fired on RNN_Node " << innovation_number << " at time " << time << " is " << inputs_fired[time] << " and total_inputs is " << total_inputs << endl;
        exit(1);
    }

    //cout << "node " << innovation_number << " - input value[" << time << "]: " << input_values[time] << endl;


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

    /*
    cout << "error fired at time: " << time << " on node " << innovation_number
        << ", d_input: " << d_input[time]
        << ", ld_output: " << ld_output[time]
        << ", error_values: " << error_values[time]
        << ", output_values: " << output_values[time]
        << endl;
    */


    d_input[time] += error_values[time] * error;

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
    error_values.assign(series_length, 0.0);

    inputs_fired.assign(series_length, 0);
    outputs_fired.assign(series_length, 0);

    d_bias = 0.0;
}

void RNN_Node::get_gradients(vector<double> &gradients) {
    gradients.assign(1, d_bias);
}

uint32_t RNN_Node::get_number_weights() const {
    return 1;
}

void RNN_Node::get_weights(vector<double> &parameters) const {
    parameters.resize(get_number_weights());
    //no weights to set in a basic RNN node, only a bias
    uint32_t offset = 0;
    get_weights(offset, parameters);
}

void RNN_Node::set_weights(const vector<double> &parameters) {
    //no weights to set in a basic RNN node, only a bias
    uint32_t offset = 0;
    set_weights(offset, parameters);
}

void RNN_Node::get_weights(uint32_t &offset, vector<double> &parameters) const {
    //no weights to set in a basic RNN node, only a bias
    parameters[offset++] = bias;
}

void RNN_Node::set_weights(uint32_t &offset, const vector<double> &parameters) {
    //no weights to set in a basic RNN node, only a bias
    bias = parameters[offset++];
}

RNN_Node_Interface* RNN_Node::copy() const {
    RNN_Node* n = new RNN_Node(innovation_number, layer_type, depth, node_type);

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
    n->enabled = enabled;
    n->forward_reachable = forward_reachable;
    n->backward_reachable = backward_reachable;

    return n;
}

void RNN_Node::write_to_stream(ostream &out) {
    RNN_Node_Interface::write_to_stream(out);
}
