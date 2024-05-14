#include <cmath>
#include <vector>
using std::vector;

#include "common/log.hxx"
#include "inverse_node_gp.hxx"

INVERSE_Node_GP::INVERSE_Node_GP(int32_t _innovation_number, int32_t _layer_type, double _depth)
    : RNN_Node(_innovation_number, _layer_type, _depth, INVERSE_NODE_GP) {
    Log::debug("created node: %d, layer type: %d, node type: INVERSE_NODE_GP\n", innovation_number, layer_type);
}

INVERSE_Node_GP::~INVERSE_Node_GP() {
}

double INVERSE_Node_GP::activation_function(double input) {
    return 1.0 / (input);
}

double INVERSE_Node_GP::derivative_function(double input) {
    double gradient = -1.0 / ((input) * (input));
    if (isnan(gradient) || isinf(gradient)) {
        gradient = -1000.0;
    }
    return gradient;
}

RNN_Node_Interface* INVERSE_Node_GP::copy() const {
    INVERSE_Node_GP* n = new INVERSE_Node_GP(innovation_number, layer_type, depth);
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
