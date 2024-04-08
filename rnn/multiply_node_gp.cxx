#include <cmath>
#include <limits>
using std::numeric_limits;

#include <vector>
using std::vector;

#include "common/log.hxx"
#include "multiply_node_gp.hxx"

MULTIPLY_Node_GP::MULTIPLY_Node_GP(int32_t _innovation_number, int32_t _layer_type, double _depth)
    : MULTIPLY_Node(_innovation_number, _layer_type, _depth) {
    node_type = MULTIPLY_NODE_GP;
    Log::debug("created node: %d, layer type: %d, node type: MULTIPLY_NODE_GP\n", innovation_number, layer_type);
}

MULTIPLY_Node_GP::~MULTIPLY_Node_GP() {
}

void MULTIPLY_Node_GP::input_fired(int32_t time, double incoming_output) {
    inputs_fired[time]++;

    ordered_input[time].push_back(incoming_output);

    if (inputs_fired[time] == 1) {
        input_values[time] = incoming_output;
    } else {
        input_values[time] *= incoming_output;
    }
    if (inputs_fired[time] < total_inputs) {
        return;
    } else if (inputs_fired[time] > total_inputs) {
        Log::fatal(
            "ERROR: inputs_fired on RNN_Node %d at time %d is %d and total_inputs is %d\n", innovation_number, time,
            inputs_fired[time], total_inputs
        );
        exit(1);
    }

    Log::debug("node %d - input value[%d]: %lf\n", innovation_number, time, input_values[time]);

    output_values[time] = input_values[time] * bias;

    if (ordered_input[time].size() != inputs_fired[time]) {
        Log::fatal("ERROR: size of total_input is not the same as ordered_inputs\n");
        Log::fatal("total: %d ordered: %d\n", total_inputs, ordered_input.size());
        exit(1);
    }

    double total;
    for (int i = 0; i < total_inputs; i++) {
        total = 1.0;
        for (int j = 0; j < total_inputs; j++) {
            if (j != i) {
                total *= ordered_input[time][j];
            }
        }
        ordered_d_input[time].push_back(total * bias);
    }

#ifdef NAN_CHECKS
    if (isnan(output_values[time]) || isinf(output_values[time])) {
        Log::fatal(
            "ERROR: output_value[%d] becaome %lf on RNN node: %d\n", time, output_values[time], innovation_number
        );
        Log::fatal("\tinput_value[%dd]: %lf\n", time, input_values[time]);
        Log::Fatal("\tnode bias: %lf", bias);
        exit(1);
    }
#endif
}

void MULTIPLY_Node_GP::try_update_deltas(int32_t time) {
    if (outputs_fired[time] < total_outputs) {
        return;
    } else if (outputs_fired[time] > total_outputs) {
        Log::fatal(
            "ERROR: outputs_fired on RNN_Node %d at time %d is %d and total_outputs is %d\n", innovation_number, time,
            outputs_fired[time], total_outputs
        );
        exit(1);
    }

    d_bias += (d_input[time] * input_values[time]);
    for (double& num : ordered_d_input[time]) {
        num *= d_input[time];

        // most likely gradient got huge, so clip it
        if (num == NAN || num == -numeric_limits<double>::infinity()) {
            num = -1000.0;
        } else if (num == NAN || num == numeric_limits<double>::infinity()) {
            num = 1000.0;
        }
    }
}

RNN_Node_Interface* MULTIPLY_Node_GP::copy() const {
    MULTIPLY_Node_GP* n = NULL;
    if (layer_type == HIDDEN_LAYER) {
        n = new MULTIPLY_Node_GP(innovation_number, layer_type, depth);
    }
    n->bias = bias;
    n->d_bias = d_bias;
    n->ordered_d_input = ordered_d_input;
    n->ordered_input = ordered_input;

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
