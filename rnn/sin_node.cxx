#include <cmath>
#include <vector>
using std::vector;

#include "common/log.hxx"
#include "sin_node.hxx"

SIN_Node::SIN_Node(int32_t _innovation_number, int32_t _layer_type, double _depth)
    : RNN_Node(_innovation_number, _layer_type, _depth, SIN_NODE) {
    Log::debug("created node: %d, layer type: %d, node type: SIN_NODE\n", innovation_number, layer_type);
}

SIN_Node::~SIN_Node() {
}

double SIN_Node::activation_function(double input) {
    return sin(input);
}

double SIN_Node::derivative_function(double input) {
    return cos(input);
}

RNN_Node_Interface* SIN_Node::copy() const {
    SIN_Node* n = new SIN_Node(innovation_number, layer_type, depth);
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
