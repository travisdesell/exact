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
#include "delta_node.hxx"


#define NUMBER_DELTA_WEIGHTS 6

Delta_Node::Delta_Node(int _innovation_number, int _type, double _depth) : RNN_Node_Interface(_innovation_number, _type, _depth) {
    node_type = DELTA_NODE;
}

Delta_Node::~Delta_Node() {
}

void Delta_Node::initialize_lamarckian(minstd_rand0 &generator, NormalDistribution &normal_distribution, double mu, double sigma) {

    alpha = bound(normal_distribution.random(generator, mu, sigma));
    beta1 = bound(normal_distribution.random(generator, mu, sigma));
    beta2 = bound(normal_distribution.random(generator, mu, sigma));
    v = bound(normal_distribution.random(generator, mu, sigma));
    r_bias = bound(normal_distribution.random(generator, mu, sigma));
    z_hat_bias = bound(normal_distribution.random(generator, mu, sigma));
}

void Delta_Node::initialize_xavier(minstd_rand0 &generator, uniform_real_distribution<double> &rng_1_1, double range) {

    alpha = range * (rng_1_1(generator));
    beta1 = range * (rng_1_1(generator));
    beta2 = range * (rng_1_1(generator));
    v = range * (rng_1_1(generator));
    r_bias = range * (rng_1_1(generator));
    z_hat_bias = range * (rng_1_1(generator));
}

void Delta_Node::initialize_kaiming(minstd_rand0 &generator, NormalDistribution &normal_distribution, double range){
    alpha = range * normal_distribution.random(generator, 0, 1);
    beta1 = range * normal_distribution.random(generator, 0, 1);
    beta2 = range * normal_distribution.random(generator, 0, 1);
    v = range * normal_distribution.random(generator, 0, 1);
    r_bias = range * normal_distribution.random(generator, 0, 1);
    z_hat_bias = range * normal_distribution.random(generator, 0, 1);
}

void Delta_Node::initialize_uniform_random(minstd_rand0 &generator, uniform_real_distribution<double> &rng) {
    alpha = rng(generator);
    beta1 = rng(generator);
    beta2 = rng(generator);
    v = rng(generator);
    r_bias = rng(generator);
    z_hat_bias = rng(generator);
}

double Delta_Node::get_gradient(string gradient_name) {
    double gradient_sum = 0.0;

    for (uint32_t i = 0; i < series_length; i++ ) {
        if (gradient_name == "alpha") {
            gradient_sum += d_alpha[i];
        } else if (gradient_name == "beta1") {
            gradient_sum += d_beta1[i];
        } else if (gradient_name == "beta2") {
            gradient_sum += d_beta2[i];
        } else if (gradient_name == "v") {
            gradient_sum += d_v[i];
        } else if (gradient_name == "r_bias") {
            gradient_sum += d_r_bias[i];
        } else if (gradient_name == "z_hat_bias") {
            gradient_sum += d_z_hat_bias[i];
        } else {
            Log::fatal("ERROR: tried to get unknown gradient: '%s'\n", gradient_name.c_str());
            exit(1);
        }
    }

    return gradient_sum;
}

void Delta_Node::print_gradient(string gradient_name) {
    Log::info("\tgradient['%s']: %lf\n", gradient_name.c_str(), get_gradient(gradient_name));
}

void Delta_Node::input_fired(int time, double incoming_output) {
    inputs_fired[time]++;

    input_values[time] += incoming_output;

    if (inputs_fired[time] < total_inputs) return;
    else if (inputs_fired[time] > total_inputs) {
        Log::fatal("ERROR: inputs_fired on Delta_Node %d at time %d is %d and total_inputs is %d\n", innovation_number, time, inputs_fired[time], total_inputs);
        exit(1);
    }

    //update alpha, beta1, beta2 so they're centered around 2, 1 and 1
    alpha += 2;
    beta1 += 1;
    beta2 += 1;

    double d2 = input_values[time];

    double z_prev = 0.0;
    if (time > 0) z_prev = output_values[time - 1];

    double d1 = v * z_prev;

    double z_hat_1 = d1 * d2 * alpha;
    double z_hat_2 = d1 * beta1;

    double z_hat_3 = d2 * beta2;
    double z_hat_sum = z_hat_1 + z_hat_2 + z_hat_3 + z_hat_bias;
    z_cap[time] = tanh(z_hat_sum);
    ld_z_cap[time] = tanh_derivative(z_cap[time]);

    double input_r_bias = d2 + r_bias;
    r[time] = sigmoid(input_r_bias);
    ld_r[time] = sigmoid_derivative(r[time]);

    double z_1 = z_cap[time] * (1 - r[time]);
    double z_2 = r[time] * z_prev;

    //TODO:
    //try this with RELU(0 to 6)) or identity

    output_values[time] = tanh(z_1 + z_2);
    ld_z[time] = tanh_derivative(output_values[time]);

    //reset alpha, beta1, beta2 so they don't mess with mean/stddev calculations for
    //parameter generation
    alpha -= 2.0;
    beta1 -= 1.0;
    beta2 -= 1.0;
}

void Delta_Node::try_update_deltas(int time) {
    if (outputs_fired[time] < total_outputs) return;
    else if (outputs_fired[time] > total_outputs) {
        Log::fatal("ERROR: outputs_fired on Delta_Node %d at time %d is %d and total_outputs is %d\n", innovation_number, time, outputs_fired[time], total_outputs);
        exit(1);
    }

    //update the alpha and betas to be their actual value
    alpha += 2.0;
    beta1 += 1.0;
    beta2 += 1.0;

    double error = error_values[time];
    double d2 = input_values[time];

    double z_prev = 0.0;
    if (time > 0) z_prev = output_values[time - 1];


    //backprop output gate
    double d_z = error;
    if (time < (series_length - 1)) d_z += d_z_prev[time + 1];
    //get the error into the output (z), it's the error from ahead in the network
    //as well as from the previous output of the cell

    d_z *= ld_z[time];

    d_z_prev[time] = d_z * r[time];

    double d_r = ((d_z * z_cap[time] * -1) + (d_z * z_prev)) * ld_r[time];
    d_r_bias[time] = d_r;
    d_input[time] = d_r;

    double d_z_cap = d_z * ld_z_cap[time] * (1 - r[time]);
    //d_z_hat_bias route
    d_z_hat_bias[time] = d_z_cap;

    //z_hat_3 route
    d_input[time] += d_z_cap * beta2;
    d_beta2[time] = d_z_cap * d2;

    //z_hat_1 route
    double d1 = v * z_prev;
    d_input[time] += d_z_cap * alpha * d1;
    d_alpha[time] = d_z_cap * d2 * d1;

    //z_hat_2 route
    d_beta1[time] = d_z_cap * d1;
    double d_d1 = (d_z_cap * beta1) + (d2 * alpha * d_z_cap);
    d_v[time] = d_d1 * z_prev;
    d_z_prev[time] += d_d1 * v;

    //reset the alpha/betas to be around 0
    alpha -= 2.0;
    beta1 -= 1.0;
    beta2 -= 1.0;
}

void Delta_Node::error_fired(int time, double error) {
    outputs_fired[time]++;

    error_values[time] *= error;

    try_update_deltas(time);
}

void Delta_Node::output_fired(int time, double delta) {
    outputs_fired[time]++;

    error_values[time] += delta;

    try_update_deltas(time);
}


uint32_t Delta_Node::get_number_weights() const {
    return NUMBER_DELTA_WEIGHTS;
}

void Delta_Node::get_weights(vector<double> &parameters) const {
    parameters.resize(get_number_weights());
    uint32_t offset = 0;
    get_weights(offset, parameters);
}

void Delta_Node::set_weights(const vector<double> &parameters) {
    uint32_t offset = 0;
    set_weights(offset, parameters);
}


void Delta_Node::set_weights(uint32_t &offset, const vector<double> &parameters) {
    //uint32_t start_offset = offset;

    alpha = bound(parameters[offset++]);
    beta1 = bound(parameters[offset++]);
    beta2 = bound(parameters[offset++]);
    v = bound(parameters[offset++]);

    r_bias = bound(parameters[offset++]);
    z_hat_bias = bound(parameters[offset++]);

    //uint32_t end_offset = offset;
    //Log::trace("set weights from offset %d to %d on Delta_node %d\n", start_offset, end_offset, innovation_number);
}

void Delta_Node::get_weights(uint32_t &offset, vector<double> &parameters) const {
    //uint32_t start_offset = offset;

    parameters[offset++] = alpha;
    parameters[offset++] = beta1;
    parameters[offset++] = beta2;
    parameters[offset++] = v;

    parameters[offset++] = r_bias;
    parameters[offset++] = z_hat_bias;

    //uint32_t end_offset = offset;
    //Log::trace("got weights from offset %d to %d on Delta_node %d\n", start_offset, end_offset, innovation_number);
}


void Delta_Node::get_gradients(vector<double> &gradients) {
    gradients.assign(NUMBER_DELTA_WEIGHTS, 0.0);

    for (uint32_t i = 0; i < NUMBER_DELTA_WEIGHTS; i++) {
        gradients[i] = 0.0;
    }

    for (uint32_t i = 0; i < series_length; i++) {
        gradients[0] += d_alpha[i];
        gradients[1] += d_beta1[i];
        gradients[2] += d_beta2[i];
        gradients[3] += d_v[i];

        gradients[4] += d_r_bias[i];
        gradients[5] += d_z_hat_bias[i];
    }
}

void Delta_Node::reset(int _series_length) {
    series_length = _series_length;

    d_alpha.assign(series_length, 0.0);
    d_beta1.assign(series_length, 0.0);
    d_beta2.assign(series_length, 0.0);
    d_v.assign(series_length, 0.0);
    d_r_bias.assign(series_length, 0.0);
    d_z_hat_bias.assign(series_length, 0.0);
    d_z_prev.assign(series_length, 0.0);

    r.assign(series_length, 0.0);
    ld_r.assign(series_length, 0.0);
    z_cap.assign(series_length, 0.0);
    ld_z_cap.assign(series_length, 0.0);
    ld_z.assign(series_length, 0.0);

    d_input.assign(series_length, 0.0);
    error_values.assign(series_length, 0.0);

    input_values.assign(series_length, 0.0);
    output_values.assign(series_length, 0.0);

    inputs_fired.assign(series_length, 0);
    outputs_fired.assign(series_length, 0);
}

RNN_Node_Interface* Delta_Node::copy() const {
    Delta_Node* n = new Delta_Node(innovation_number, layer_type, depth);

    //copy Delta_Node values
    n->d_alpha = d_alpha;
    n->d_beta1 = d_beta1;
    n->d_beta2 = d_beta2;
    n->d_v = d_v;
    n->d_r_bias = d_r_bias;
    n->d_z_hat_bias = d_z_hat_bias;
    n->d_z_prev = d_z_prev;

    n->r = r;
    n->ld_r = ld_r;
    n->z_cap = z_cap;
    n->ld_z_cap = ld_z_cap;
    n->ld_z = ld_z;

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

void Delta_Node::write_to_stream(ostream &out) {
    RNN_Node_Interface::write_to_stream(out);
}
