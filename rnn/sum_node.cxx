#include <cmath>
#include <vector>
using std::vector;

#include "common/log.hxx"
#include "sum_node.hxx"

SUM_Node::SUM_Node(int32_t _innovation_number, int32_t _layer_type, double _depth)
    : RNN_Node(_innovation_number, _layer_type, _depth, SUM_NODE) {
    Log::debug("created node: %d, layer type: %d, node type: SUM_NODE\n", innovation_number, layer_type);
}

SUM_Node::~SUM_Node() {
}

double SUM_Node::activation_function(double input) {
    return input;
}

double SUM_Node::derivative_function(double input) {
    return 1.0;
}

RNN_Node_Interface* SUM_Node::copy() const {
    SUM_Node* n = new SUM_Node(innovation_number, layer_type, depth);
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
