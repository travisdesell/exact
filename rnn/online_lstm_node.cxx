#include <cmath>

#include <fstream>
using std::ostream;

#include <iomanip>
using std::setw;

#include <string>
using std::string;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <vector>
using std::vector;

#include "common/random.hxx"
#include "common/log.hxx"

#include "rnn_node_interface.hxx"
#include "mse.hxx"
#include "online_lstm_node.hxx"


Online_LSTM_Node::Online_LSTM_Node(int _innovation_number, int _type, double _depth) : RNN_Node_Interface(_innovation_number, _type, _depth) {
    node_type = ONLINE_LSTM_NODE;
}

Online_LSTM_Node::~Online_LSTM_Node() {
}

void Online_LSTM_Node::initialize_lamarckian(minstd_rand0 &generator, NormalDistribution &normal_distribution, double mu, double sigma) {

    output_gate_update_weight = bound(normal_distribution.random(generator, mu, sigma));
    output_gate_weight = bound(normal_distribution.random(generator, mu, sigma));
    output_gate_bias = bound(normal_distribution.random(generator, mu, sigma));
    //output_gate_bias = 0.0;

    input_gate_update_weight = bound(normal_distribution.random(generator, mu, sigma));
    input_gate_weight = bound(normal_distribution.random(generator, mu, sigma));
    input_gate_bias = bound(normal_distribution.random(generator, mu, sigma));
    //input_gate_bias = 0.0;

    forget_gate_update_weight = bound(normal_distribution.random(generator, mu, sigma));
    forget_gate_weight = bound(normal_distribution.random(generator, mu, sigma));
    forget_gate_bias = bound(normal_distribution.random(generator, mu, sigma));
    //forget_gate_bias = 1.0 + bound(normal_distribution.random(generator, mu, sigma));

    cell_weight = bound(normal_distribution.random(generator, mu, sigma));
    cell_bias = bound(normal_distribution.random(generator, mu, sigma));
    //cell_bias = 0.0;
}

void Online_LSTM_Node::initialize_xavier(minstd_rand0 &generator, uniform_real_distribution<double> &rng_1_1, double range) {

    output_gate_update_weight = range * (rng_1_1(generator));
    output_gate_weight = range * (rng_1_1(generator));
    output_gate_bias = range * (rng_1_1(generator));
    //output_gate_bias = 0.0;

    input_gate_update_weight = range * (rng_1_1(generator));
    input_gate_weight = range * (rng_1_1(generator));
    input_gate_bias = range * (rng_1_1(generator));
    //input_gate_bias = 0.0;

    forget_gate_update_weight = range * (rng_1_1(generator));
    forget_gate_weight = range * (rng_1_1(generator));
    forget_gate_bias = range * (rng_1_1(generator));
    //forget_gate_bias = 1.0 + bound(normal_distribution.random(generator, mu, sigma));

    cell_weight = range * (rng_1_1(generator));
    cell_bias = range * (rng_1_1(generator));
}

void Online_LSTM_Node::initialize_kaiming(minstd_rand0 &generator, NormalDistribution &normal_distribution, double range) {
    output_gate_update_weight = range * normal_distribution.random(generator, 0, 1);
    output_gate_weight = range * normal_distribution.random(generator, 0, 1);
    output_gate_bias = range * normal_distribution.random(generator, 0, 1);
    //output_gate_bias = 0.0;

    input_gate_update_weight = range * normal_distribution.random(generator, 0, 1);
    input_gate_weight = range * normal_distribution.random(generator, 0, 1);
    input_gate_bias = range * normal_distribution.random(generator, 0, 1);
    //input_gate_bias = 0.0;

    forget_gate_update_weight = range * normal_distribution.random(generator, 0, 1);
    forget_gate_weight = range * normal_distribution.random(generator, 0, 1);
    forget_gate_bias = range * normal_distribution.random(generator, 0, 1);
    //forget_gate_bias = 1.0 + bound(normal_distribution.random(generator, mu, sigma));

    cell_weight = range * normal_distribution.random(generator, 0, 1);
    cell_bias = range * normal_distribution.random(generator, 0, 1);
}

void Online_LSTM_Node::initialize_uniform_random(minstd_rand0 &generator, uniform_real_distribution<double> &rng) {
    output_gate_update_weight = rng(generator);
    output_gate_weight = rng(generator);
    output_gate_bias = rng(generator);
    //output_gate_bias = 0.0;

    input_gate_update_weight = rng(generator);
    input_gate_weight = rng(generator);
    input_gate_bias = rng(generator);
    //input_gate_bias = 0.0;

    forget_gate_update_weight = rng(generator);
    forget_gate_weight = rng(generator);
    forget_gate_bias = rng(generator);
    //forget_gate_bias = 1.0 + bound(normal_distribution.random(generator, mu, sigma));

    cell_weight = rng(generator);
    cell_bias = rng(generator);
}

double Online_LSTM_Node::get_gradient(string gradient_name) {
    double gradient_sum = 0.0;

    // for (uint32_t i = 0; i < series_length; i++ ) {
        if (gradient_name == "output_gate_update_weight") {
            gradient_sum += d_output_gate_update_weight;
        } else if (gradient_name == "output_gate_weight") {
            gradient_sum += d_output_gate_weight;
        } else if (gradient_name == "output_gate_bias") {
            gradient_sum += d_output_gate_bias;
        } else if (gradient_name == "input_gate_update_weight") {
            gradient_sum += d_input_gate_update_weight;
        } else if (gradient_name == "input_gate_weight") {
            gradient_sum += d_input_gate_weight;
        } else if (gradient_name == "input_gate_bias") {
            gradient_sum += d_input_gate_bias;
        } else if (gradient_name == "forget_gate_update_weight") {
            gradient_sum += d_forget_gate_update_weight;
        } else if (gradient_name == "forget_gate_weight") {
            gradient_sum += d_forget_gate_weight;
        } else if (gradient_name == "forget_gate_bias") {
            gradient_sum += d_forget_gate_bias;
        } else if (gradient_name == "cell_weight") {
            gradient_sum += d_cell_weight;
        } else if (gradient_name == "cell_bias") {
            gradient_sum += d_cell_bias;
        } else {
            Log::fatal("ERROR: tried to get unknown gradient: '%s'\n", gradient_name.c_str());
            exit(1);
        }
    // }

    return gradient_sum;
}

void Online_LSTM_Node::print_gradient(string gradient_name) {
    Log::info("\tgradient['%s']: %lf\n", gradient_name.c_str(), get_gradient(gradient_name));
}

void Online_LSTM_Node::input_fired(int32_t time, double incoming_output) {
    inputs_fired ++;

    input_values += incoming_output;

    if (inputs_fired < total_inputs) return;
    else if (inputs_fired > total_inputs) {
        Log::fatal("ERROR: inputs_fired on LSTM_Node %d at time %d is %d and total_inputs is %d\n", innovation_number, time, inputs_fired, total_inputs);
        exit(1);
    }

    double input_value = input_values;

    // double previous_cell_value = 0.0;
    // if (time > 0) previous_cell_value = previous_cell_values;

    // only previous cell values need to be saved
    // save previous cell values for propagate forward and backward, before cell_values get updated
    previous_cell_values = cell_values;

    //forget gate bias should be around 1.0 intead of 0, but we do it here to not throw
    //off the mu/sigma of the parameters
    forget_gate_bias = forget_gate_bias + 1.0;

    output_gate_values = sigmoid(output_gate_weight * input_value + output_gate_update_weight * previous_cell_values + output_gate_bias);
    input_gate_values = sigmoid(input_gate_weight * input_value + input_gate_update_weight * previous_cell_values + input_gate_bias);
    forget_gate_values = sigmoid(forget_gate_weight * input_value + forget_gate_update_weight * previous_cell_values + forget_gate_bias);

    ld_output_gate = sigmoid_derivative(output_gate_values);
    ld_input_gate = sigmoid_derivative(input_gate_values);
    ld_forget_gate = sigmoid_derivative(forget_gate_values);

    cell_in_tanh = tanh(cell_weight * input_value + cell_bias);
    ld_cell_in = tanh_derivative(cell_in_tanh);

    cell_values = (forget_gate_values * previous_cell_values) + (input_gate_values * cell_in_tanh);
    

    //The original is a hyperbolic tangent, but the peephole[clarification needed] LSTM paper suggests the activation function be linear -- activation(x) = x
    cell_out_tanh = cell_values;
    ld_cell_out = 1.0;

    output_values = output_gate_values * cell_out_tanh;

    forget_gate_bias -= 1.0;
}

void Online_LSTM_Node::try_update_deltas(int time) {
    if (outputs_fired < total_outputs) return;
    else if (outputs_fired > total_outputs) {
        Log::fatal("ERROR: outputs_fired on LSTM_Node %d at time %d is %d and total_outputs is %d\n", innovation_number, time, outputs_fired, total_outputs);
        exit(1);
    }

    double error = error_values;
    double input_value = input_values;

    // double previous_cell_value = 0.00;
    // if (time > 0) previous_cell_value = cell_values[time - 1];

    // need to save the error from the next cell if it exists (from time t+1 )
    // d_next_cell = d_prev_cell;

    //backprop output gate
    double d_output_gate = error * cell_out_tanh * ld_output_gate;
    d_output_gate_bias = d_output_gate;
    d_output_gate_update_weight = d_output_gate * previous_cell_values;
    d_output_gate_weight = d_output_gate * input_value;
    d_prev_cell += d_output_gate * output_gate_update_weight;
    d_input += d_output_gate * output_gate_weight;

    //backprop the cell path

    double d_cell_out = error * output_gate_values * ld_cell_out;
    //propagate error back from the next cell value if there is one
    // if (time < (series_length - 1)) d_cell_out += d_next_cell;

    //backprop forget gate
    d_prev_cell += d_cell_out * forget_gate_values;

    double d_forget_gate = d_cell_out * previous_cell_values * ld_forget_gate;
    d_forget_gate_bias = d_forget_gate;
    d_forget_gate_update_weight = d_forget_gate * previous_cell_values;
    d_forget_gate_weight = d_forget_gate * input_value;
    d_prev_cell += d_forget_gate * forget_gate_update_weight;
    d_input += d_forget_gate * forget_gate_weight;

    //backprob input gate
    double d_input_gate = d_cell_out * cell_in_tanh * ld_input_gate;
    d_input_gate_bias = d_input_gate;
    d_input_gate_update_weight = d_input_gate * previous_cell_values;
    d_input_gate_weight = d_input_gate * input_value;
    d_prev_cell += d_input_gate * input_gate_update_weight;
    d_input += d_input_gate * input_gate_weight;

    //backprop cell input
    double d_cell_in = d_cell_out * input_gate_values * ld_cell_in;
    d_cell_bias = d_cell_in;
    d_cell_weight = d_cell_in * input_value;
    d_input += d_cell_in * cell_weight;
}

void Online_LSTM_Node::error_fired(int time, double error) {
    outputs_fired++;

    error_values *= error;

    try_update_deltas(time);
}

void Online_LSTM_Node::output_fired(int time, double delta) {
    outputs_fired++;

    error_values += delta;

    try_update_deltas(time);
}

uint32_t Online_LSTM_Node::get_number_weights() const {
    return 11;
}

void Online_LSTM_Node::get_weights(vector<double> &parameters) const {
    parameters.resize(get_number_weights());
    uint32_t offset = 0;
    get_weights(offset, parameters);
}

double Online_LSTM_Node::get_output_value(int32_t time) const {
    return output_values;
}

double Online_LSTM_Node::get_input_value(int32_t time) const {
    return input_values;
}

double Online_LSTM_Node::get_error_value(int32_t time) const {
    return error_values;
}

double Online_LSTM_Node::get_d_input(int32_t time) const {
    return d_input;
}

int32_t Online_LSTM_Node::get_input_fired(int32_t time) const {
    return inputs_fired;
}
int32_t Online_LSTM_Node::get_output_fired(int32_t time) const{
    return outputs_fired;
}

void Online_LSTM_Node::set_weights(const vector<double> &parameters) {
    uint32_t offset = 0;
    set_weights(offset, parameters);
}


void Online_LSTM_Node::set_weights(uint32_t &offset, const vector<double> &parameters) {
    //uint32_t start_offset = offset;

    output_gate_update_weight = bound(parameters[offset++]);
    output_gate_weight = bound(parameters[offset++]);
    output_gate_bias = bound(parameters[offset++]);

    input_gate_update_weight = bound(parameters[offset++]);
    input_gate_weight = bound(parameters[offset++]);
    input_gate_bias = bound(parameters[offset++]);

    forget_gate_update_weight = bound(parameters[offset++]);
    forget_gate_weight = bound(parameters[offset++]);
    forget_gate_bias = bound(parameters[offset++]);

    cell_weight = bound(parameters[offset++]);
    cell_bias = bound(parameters[offset++]);

    //uint32_t end_offset = offset;
    //Log::trace("set weights from offset %d to %d on LSTM_Node %d\n", start_offset, end_offset, innovation_number);
}

void Online_LSTM_Node::get_weights(uint32_t &offset, vector<double> &parameters) const {
    //uint32_t start_offset = offset;

    parameters[offset++] = output_gate_update_weight;
    parameters[offset++] = output_gate_weight;
    parameters[offset++] = output_gate_bias;

    parameters[offset++] = input_gate_update_weight;
    parameters[offset++] = input_gate_weight;
    parameters[offset++] = input_gate_bias;

    parameters[offset++] = forget_gate_update_weight;
    parameters[offset++] = forget_gate_weight;
    parameters[offset++] = forget_gate_bias;

    parameters[offset++] = cell_weight;
    parameters[offset++] = cell_bias;

    //uint32_t end_offset = offset;
    //Log::trace("got weights from offset %d to %d on LSTM_Node %d\n", start_offset, end_offset, innovation_number);
}


void Online_LSTM_Node::get_gradients(vector<double> &gradients) {
    gradients.assign(11, 0.0);

    for (uint32_t i = 0; i < 11; i++) {
        gradients[i] = 0.0;
    }

    // for (uint32_t i = 0; i < series_length; i++) {
        gradients[0] += d_output_gate_update_weight;
        gradients[1] += d_output_gate_weight;
        gradients[2] += d_output_gate_bias;

        gradients[3] += d_input_gate_update_weight;
        gradients[4] += d_input_gate_weight;
        gradients[5] += d_input_gate_bias;

        gradients[6] += d_forget_gate_update_weight;
        gradients[7] += d_forget_gate_weight;
        gradients[8] += d_forget_gate_bias;

        gradients[9] += d_cell_weight;
        gradients[10] += d_cell_bias;
    // }
}

void Online_LSTM_Node::reset(int _series_length) {
    series_length = _series_length;

    ld_output_gate = 0;
    ld_input_gate = 0;
    ld_forget_gate = 0;

    cell_in_tanh = 0;
    cell_out_tanh = 0;
    ld_cell_in = 0;
    ld_cell_out = 0;

    d_input = 0;
    d_prev_cell = 0;

    d_output_gate_update_weight = 0;
    d_output_gate_weight = 0;
    d_output_gate_bias = 0;

    d_input_gate_update_weight = 0;
    d_input_gate_weight = 0;
    d_input_gate_bias = 0;

    d_forget_gate_update_weight = 0;
    d_forget_gate_weight = 0;
    d_forget_gate_bias = 0;

    d_cell_weight = 0;
    d_cell_bias = 0;

    output_gate_values = 0;
    input_gate_values = 0;
    forget_gate_values = 0;
    cell_values = 0;

    error_values = 0;

    input_values = 0;
    output_values = 0;

    inputs_fired = 0;
    outputs_fired = 0;
}

RNN_Node_Interface* Online_LSTM_Node::copy() const {
    Online_LSTM_Node* n = new Online_LSTM_Node(innovation_number, layer_type, depth);

    //copy LSTM_Node values
    n->output_gate_update_weight = output_gate_update_weight;
    n->output_gate_weight = output_gate_weight;
    n->output_gate_bias = output_gate_bias;

    n->input_gate_update_weight = input_gate_update_weight;
    n->input_gate_weight = input_gate_weight;
    n->input_gate_bias = input_gate_bias;

    n->forget_gate_update_weight = forget_gate_update_weight;
    n->forget_gate_weight = forget_gate_weight;
    n->forget_gate_bias = forget_gate_bias;

    n->cell_weight = cell_weight;
    n->cell_bias = cell_bias;

    n->output_gate_values = output_gate_values;
    n->input_gate_values = input_gate_values;
    n->forget_gate_values = forget_gate_values;
    n->cell_values = cell_values;
    n->previous_cell_values = previous_cell_values;

    n->ld_output_gate = ld_output_gate;
    n->ld_input_gate = ld_input_gate;
    n->ld_forget_gate = ld_forget_gate;

    n->cell_in_tanh = cell_in_tanh;
    n->cell_out_tanh = cell_out_tanh;
    n->ld_cell_in = ld_cell_in;
    n->ld_cell_out = ld_cell_out;

    n->d_prev_cell = d_prev_cell;
    // n->d_next_cell = d_next_cell;

    n->d_output_gate_update_weight = d_output_gate_update_weight;
    n->d_output_gate_weight = d_output_gate_weight;
    n->d_output_gate_bias = d_output_gate_bias;

    n->d_input_gate_update_weight = d_input_gate_update_weight;
    n->d_input_gate_weight = d_input_gate_weight;
    n->d_input_gate_bias = d_input_gate_bias;

    n->d_forget_gate_update_weight = d_forget_gate_update_weight;
    n->d_forget_gate_weight = d_forget_gate_weight;
    n->d_forget_gate_bias = d_forget_gate_bias;

    n->d_cell_weight = d_cell_weight;
    n->d_cell_bias = d_cell_bias;


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

void Online_LSTM_Node::write_to_stream(ostream &out) {
    RNN_Node_Interface::write_to_stream(out);
}
