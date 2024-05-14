#include <cmath>
#include <vector>
using std::vector;

#include "common/log.hxx"
#include "sigmoid_node_gp.hxx"

SIGMOID_Node_GP::SIGMOID_Node_GP(int32_t _innovation_number, int32_t _layer_type, double _depth)
    : RNN_Node(_innovation_number, _layer_type, _depth, SIGMOID_NODE_GP) {
    Log::debug("created node: %d, layer type: %d, node type: SIGMOID_NODE_GP\n", innovation_number, layer_type);
}

SIGMOID_Node_GP::~SIGMOID_Node_GP() {
}

double SIGMOID_Node_GP::activation_function(double input) {
    double exp_value = exp(-input);
    return 1.0 / (1.0 + exp_value);
}

double SIGMOID_Node_GP::derivative_function(double input) {
    return activation_function(input) * (1 - activation_function(input));
}

RNN_Node_Interface* SIGMOID_Node_GP::copy() const {
    SIGMOID_Node_GP* n = new SIGMOID_Node_GP(innovation_number, layer_type, depth);
    // copy RNN_Node values
    n->bias = bias;
    n->d_bias = d_bias;
    n->ld_output = ld_output;

    // copy RNN_Node_Interface values
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
