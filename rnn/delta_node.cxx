#include <cmath>

#include <fstream>
using std::ostream;

#include <iomanip>
using std::setw;

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <string>
using std::string;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <vector>
using std::vector;

#include "common/random.hxx"

#include "rnn_node_interface.hxx"
#include "mse.hxx"
#include "delta_node.hxx"


#define NUMBER_DELTA_WEIGHTS 6

Delta_Node::Delta_Node(int _innovation_number, int _type, double _depth) : RNN_Node_Interface(_innovation_number, _type, _depth) {
    node_type = DELTA_NODE;
}

Delta_Node::~Delta_Node() {
}

void Delta_Node::initialize_randomly(minstd_rand0 &generator, NormalDistribution &normal_distribution, double mu, double sigma) {

    alpha = bound(normal_distribution.random(generator, mu, sigma));
    beta1 = bound(normal_distribution.random(generator, mu, sigma));
    beta2 = bound(normal_distribution.random(generator, mu, sigma));
    v = bound(normal_distribution.random(generator, mu, sigma));
    r_bias = bound(normal_distribution.random(generator, mu, sigma));
    z_hat_bias = bound(normal_distribution.random(generator, mu, sigma));
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
            cerr << "ERROR: tried to get unknown gradient: '" << gradient_name << "'" << endl;
            exit(1);
        }
    }

    return gradient_sum;
}

void Delta_Node::print_gradient(string gradient_name) {
    cout << "\tgradient['" << gradient_name << "']: " << get_gradient(gradient_name) << endl;
}

void Delta_Node::input_fired(int time, double incoming_output) {
    inputs_fired[time]++;

    input_values[time] += incoming_output;

    if (inputs_fired[time] < total_inputs) return;
    else if (inputs_fired[time] > total_inputs) {
        cerr << "ERROR: inputs_fired on Delta_Node " << innovation_number << " at time " << time << " is " << inputs_fired[time] << " and total_inputs is " << total_inputs << endl;
        exit(1);
    }

    //update alpha, beta1, beta2 so they're centered around 2, 1 and 1
    alpha += 2;
    beta1 += 1;
    beta2 += 1;


    //cout << "PROPAGATING FORWARD" << endl;

    double d2 = input_values[time];
    //cout << "node " << innovation_number << " - input value[" << time << "] (d2): " << d2 << endl;

    double z_prev = 0.0;
    if (time > 0) z_prev = output_values[time - 1];
    //cout << "node " << innovation_number << " - prev_output_value[" << time << "] (z_prev): " << z_prev << endl;

    //cout << "r_bias: " << r_bias << endl;

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

    output_values[time] = tanh(z_1 + z_2);
    ld_z[time] = tanh_derivative(output_values[time]);

    //reset alpha, beta1, beta2 so they don't mess with mean/stddev calculations for
    //parameter generation
    alpha -= 2.0;
    beta1 -= 1.0;
    beta2 -= 1.0;

    //cout << "node " << innovation_number << " - output_values[" << time << "]: " << output_values[time] << endl;
}

void Delta_Node::try_update_deltas(int time) {
    if (outputs_fired[time] < total_outputs) return;
    else if (outputs_fired[time] > total_outputs) {
        cerr << "ERROR: outputs_fired on Delta_Node " << innovation_number << " at time " << time << " is " << outputs_fired[time] << " and total_outputs is " << total_outputs << endl;
        exit(1);
    }

    //cout << "PROPAGATING BACKWARDS" << endl;
    //update the alpha and betas to be their actual value
    alpha += 2.0;
    beta1 += 1.0;
    beta2 += 1.0;

    double error = error_values[time];
    //cout << "error_values[time]: " << error << endl;
    double d2 = input_values[time];
    //cout << "input value[" << time << "]:" << d2 << endl;

    double z_prev = 0.0;
    if (time > 0) z_prev = output_values[time - 1];
    //cout << "z_prev[" << (time - 1) << "]: " << z_prev << endl;


    //backprop output gate
    double d_z = error;
    if (time < (series_length - 1)) d_z += d_z_prev[time + 1];
    //get the error into the output (z), it's the error from ahead in the network
    //as well as from the previous output of the cell

    //cout << "d_z: " << d_z << endl;
    //cout << "ld_z_cap[" << time << "]: " << ld_z_cap[time] << endl;

    d_z *= ld_z[time];

    d_z_prev[time] = d_z * r[time];

    double d_r = ((d_z * z_cap[time] * -1) + (d_z * z_prev)) * ld_r[time];
    d_r_bias[time] = d_r;
    d_input[time] = d_r;

    double d_z_cap = d_z * ld_z_cap[time] * (1 - r[time]);
    //d_z_hat_bias route
    d_z_hat_bias[time] = d_z_cap;
    //cout << "d_z_hat_bias[" << time << "]: " << d_z_hat_bias[time] << endl;

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

    //cout << "d_input: " << d_input[time] << endl;
    //cout << "d_beta2: " << d_beta2[time] << endl;

    //cout << endl << endl;

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


void Delta_Node::print_cell_values() {
    /*
    cerr << "\tinput_value: " << input_value << endl;
    cerr << "\tinput_gate_value: " << input_gate_value << ", input_gate_update_weight: " << input_gate_update_weight << ", input_gate_bias: " << input_gate_bias << endl;
    cerr << "\toutput_gate_value: " << output_gate_value << ", output_gate_update_weight: " << output_gate_update_weight << ", output_gate_bias: " << output_gate_bias << endl;
    cerr << "\tforget_gate_value: " << forget_gate_value << ", forget_gate_update_weight: " << forget_gate_update_weight << "\tforget_gate_bias: " << forget_gate_bias << endl;
    cerr << "\tcell_value: " << cell_value << ", cell_bias: " << cell_bias << endl;
    */
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

    alpha = parameters[offset++];
    beta1 = parameters[offset++];
    beta2 = parameters[offset++];
    v = parameters[offset++];

    r_bias = parameters[offset++];
    z_hat_bias = parameters[offset++];

    //uint32_t end_offset = offset;

    //cerr << "set weights from offset " << start_offset << " to " << end_offset << " on Delta_Node " << innovation_number << endl;
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

    //cerr << "set weights from offset " << start_offset << " to " << end_offset << " on Delta_Node " << innovation_number << endl;
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
    Delta_Node* n = new Delta_Node(innovation_number, type, depth);

    //cout << "COPYING!" << endl;

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
