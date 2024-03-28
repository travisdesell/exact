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

void SIGMOID_Node_GP::input_fired(int32_t time, double incoming_output) {
    inputs_fired[time]++;

    input_values[time] += incoming_output;

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
    //for gp nodes bias is added and trained only on sum and multiply node
    this->bias = 0;
    output_values[time] = activation_function(input_values[time]);
    ld_output[time] = derivative_function(input_values[time]);

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

void SIGMOID_Node_GP::try_update_deltas(int32_t time) {
    if (outputs_fired[time] < total_outputs) {
        return;
    } else if (outputs_fired[time] > total_outputs) {
        Log::fatal(
            "ERROR: outputs_fired on RNN_Node %d at time %d is %d and total_outputs is %d\n", innovation_number, time,
            outputs_fired[time], total_outputs
        );
        exit(1);
    }

    d_input[time] *= ld_output[time];
    //gp bias only trains for sum and multiply nodes
    this->d_bias = 0;
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
