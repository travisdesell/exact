#include <cmath>

#include <vector>
using std::vector;

#include "rnn_node.hxx"

#include "common/log.hxx"


RNN_Node::RNN_Node(int _innovation_number, int _layer_type, double _depth, int _node_type) : RNN_Node_Interface(_innovation_number, _layer_type, _depth), bias(0) {

    //node type will be simple, jordan or elman
    node_type = _node_type;
    Log::trace("created node: %d, layer type: %d, node type: %d\n", innovation_number, layer_type, node_type);
}


RNN_Node::RNN_Node(int _innovation_number, int _layer_type, double _depth, int _node_type, string _parameter_name) : RNN_Node_Interface(_innovation_number, _layer_type, _depth, _parameter_name), bias(0) {

    //node type will be simple, jordan or elman
    node_type = _node_type;
    Log::trace("created node: %d, layer type: %d, node type: %d\n", innovation_number, layer_type, node_type);
}


RNN_Node::~RNN_Node() {
}

void RNN_Node::initialize_lamarckian(minstd_rand0 &generator, NormalDistribution &normal_distribution, double mu, double sigma) {
    bias = bound(normal_distribution.random(generator, mu, sigma));
}

void RNN_Node::initialize_xavier(minstd_rand0 &generator, uniform_real_distribution<double> &rng_1_1, double range) {

    bias = range * (rng_1_1(generator));
}

void RNN_Node::initialize_kaiming(minstd_rand0 &generator, NormalDistribution &normal_distribution, double range) {

    bias = range * normal_distribution.random(generator, 0, 1);
}

void RNN_Node::initialize_uniform_random(minstd_rand0 &generator, uniform_real_distribution<double> &rng) {
    bias = rng(generator);
}

void RNN_Node::input_fired(int time, double incoming_output) {
    inputs_fired[time]++;

    input_values[time] += incoming_output;

    if (inputs_fired[time] < total_inputs) return;
    else if (inputs_fired[time] > total_inputs) {
        Log::fatal("ERROR: inputs_fired on RNN_Node %d at time %d is %d and total_inputs is %d\n", innovation_number, time, inputs_fired[time], total_inputs);
        exit(1);
    }

    //Log::trace("node %d - input value[%d]: %lf\n", innovation_number, time, input_values[time]);

    output_values[time] = tanh(input_values[time] + bias);
    ld_output[time] = tanh_derivative(output_values[time]);

    //output_values[time] = sigmoid(input_values[time] + bias);
    //ld_output[time] = sigmoid_derivative(output_values[time]);

#ifdef NAN_CHECKS
    if (isnan(output_values[time]) || isinf(output_values[time])) {
        Log::fatal("ERROR: output_value[%d] becaome %lf on RNN node: %d\n", time, output_values[time], innovation_number);
        Log::fatal("\tinput_value[%dd]: %lf\n", time, input_values[time]);
        Log::Fatal("\tnode bias: %lf", bias);
        exit(1);
    }
#endif
}

void RNN_Node::try_update_deltas(int time) {
    if (outputs_fired[time] < total_outputs) return;
    else if (outputs_fired[time] > total_outputs) {
        Log::fatal("ERROR: outputs_fired on RNN_Node %d at time %d is %d and total_outputs is %d\n", innovation_number, time, outputs_fired[time], total_outputs);
        exit(1);
    }

    d_input[time] *= ld_output[time];

    d_bias += d_input[time];
}

void RNN_Node::error_fired(int time, double error) {
    outputs_fired[time]++;

    //Log::trace("error fired at time: %d on node %d, d_input: %lf, ld_output %lf, error_values: %lf, output_values: %lf\n", time, innovation_number, d_input[time], ld_output[time], error_values[time], output_values[time]);

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
    bias = bound(parameters[offset++]);
}

RNN_Node_Interface* RNN_Node::copy() const {
    RNN_Node *n = NULL;
    if (layer_type == HIDDEN_LAYER) {
        n = new RNN_Node(innovation_number, layer_type, depth, node_type);
    } else {
        n = new RNN_Node(innovation_number, layer_type, depth, node_type, parameter_name);
    }

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
