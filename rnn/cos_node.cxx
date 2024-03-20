#include <cmath>
#include <vector>
using std::vector;

#include "common/log.hxx"
#include "cos_node.hxx"

COS_Node::COS_Node(int32_t _innovation_number, int32_t _layer_type, double _depth)
    : RNN_Node(_innovation_number, _layer_type, _depth, COS_NODE) {
    Log::debug("created node: %d, layer type: %d, node type: COS_NODE\n", innovation_number, layer_type);
}

COS_Node::~COS_Node() {
}

double COS_Node::activation_function(double input) {
    return cos(input);
}

double COS_Node::derivative_function(double input) {
    return -1 * sin(input);
}

RNN_Node_Interface* COS_Node::copy() const {
    COS_Node* n = new COS_Node(innovation_number, layer_type, depth);
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
