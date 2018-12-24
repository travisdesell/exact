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
#include "gru_node.hxx"


#define NUMBER_GRU_WEIGHTS 9

GRU_Node::GRU_Node(int _innovation_number, int _type, double _depth) : RNN_Node_Interface(_innovation_number, _type, _depth) {
    node_type = GRU_NODE;
}

GRU_Node::~GRU_Node() {
}

void GRU_Node::initialize_randomly(minstd_rand0 &generator, NormalDistribution &normal_distribution, double mu, double sigma) {

    zw = bound(normal_distribution.random(generator, mu, sigma));
    zu = bound(normal_distribution.random(generator, mu, sigma));
    z_bias = bound(normal_distribution.random(generator, mu, sigma));

    rw = bound(normal_distribution.random(generator, mu, sigma));
    ru = bound(normal_distribution.random(generator, mu, sigma));
    r_bias = bound(normal_distribution.random(generator, mu, sigma));

    hw = bound(normal_distribution.random(generator, mu, sigma));
    hu = bound(normal_distribution.random(generator, mu, sigma));
    h_bias = bound(normal_distribution.random(generator, mu, sigma));
}

double GRU_Node::get_gradient(string gradient_name) {
    double gradient_sum = 0.0;

    for (uint32_t i = 0; i < series_length; i++ ) {
        if (gradient_name == "zw") {
            gradient_sum += d_zw[i];
        } else if (gradient_name == "zu") {
            gradient_sum += d_zu[i];
        } else if (gradient_name == "z_bias") {
            gradient_sum += d_z_bias[i];
        } else if (gradient_name == "rw") {
            gradient_sum += d_rw[i];
        } else if (gradient_name == "ru") {
            gradient_sum += d_ru[i];
        } else if (gradient_name == "r_bias") {
            gradient_sum += d_r_bias[i];
        } else if (gradient_name == "hw") {
            gradient_sum += d_hw[i];
        } else if (gradient_name == "hu") {
            gradient_sum += d_hu[i];
        } else if (gradient_name == "h_bias") {
            gradient_sum += d_h_bias[i];
        } else {
            cerr << "ERROR: tried to get unknown gradient: '" << gradient_name << "'" << endl;
            exit(1);
        }
    }

    return gradient_sum;
}

void GRU_Node::print_gradient(string gradient_name) {
    cout << "\tgradient['" << gradient_name << "']: " << get_gradient(gradient_name) << endl;
}

void GRU_Node::input_fired(int time, double incoming_output) {
    inputs_fired[time]++;

    input_values[time] += incoming_output;

    if (inputs_fired[time] < total_inputs) return;
    else if (inputs_fired[time] > total_inputs) {
        cerr << "ERROR: inputs_fired on GRU_Node " << innovation_number << " at time " << time << " is " << inputs_fired[time] << " and total_inputs is " << total_inputs << endl;
        exit(1);
    }

    //update the reset gate bias so its centered around 1
    //r_bias += 1;

    //cout << "PROPAGATING FORWARD" << endl;

    double x = input_values[time];
    //cout << "node " << innovation_number << " - input value[" << time << "] (x): " << x << endl;

    double h_prev = 0.0;
    if (time > 0) h_prev = output_values[time - 1];
    //cout << "node " << innovation_number << " - prev_output_value[" << time << "] (h_prev): " << h_prev << endl;

    //cout << "r_bias: " << r_bias << endl;

    double hzu = h_prev * zu;
    double xzw = x * zw;
    double z_sum = z_bias + hzu + xzw;

    z[time] = sigmoid(z_sum);
    ld_z[time] = sigmoid_derivative(z[time]);

    double z_h_prev = h_prev * z[time];

    double xhw = x * hw;
    double xrw = x * rw;
    double hru = h_prev * ru;

    double r_sum = r_bias + xrw + hru;

    r[time] = sigmoid(r_sum);
    ld_r[time] = sigmoid_derivative(r[time]);

    double hu_r_h_prev = hu * r[time] * h_prev;

    double h_sum = h_bias + xhw + hu_r_h_prev;

    h_tanh[time] = tanh(h_sum);
    ld_h_tanh[time] = tanh_derivative(h_tanh[time]);

    output_values[time] = z_h_prev + (1 - z[time]) * h_tanh[time];

    //reset alpha, beta1, beta2 so they don't mess with mean/stddev calculations for
    //parameter generation
    //r_bias -= 1.0;

    //cout << "node " << innovation_number << " - output_values[" << time << "]: " << output_values[time] << endl;
}

void GRU_Node::try_update_deltas(int time) {
    if (outputs_fired[time] < total_outputs) return;
    else if (outputs_fired[time] > total_outputs) {
        cerr << "ERROR: outputs_fired on GRU_Node " << innovation_number << " at time " << time << " is " << outputs_fired[time] << " and total_outputs is " << total_outputs << endl;
        exit(1);
    }

    //cout << "PROPAGATING BACKWARDS" << endl;
    //update the reset gate bias so its centered around 1   
    //r_bias += 1.0;

    double error = error_values[time];
    //cout << "error_values[time]: " << error << endl;
    double x = input_values[time];
    //cout << "input value[" << time << "]:" << x << endl;

    double h_prev = 0.0;
    if (time > 0) h_prev = output_values[time - 1];
    //cout << "h_prev[" << (time - 1) << "]: " << h_prev << endl;


    //backprop output gate
    double d_h = error;
    if (time < (series_length - 1)) d_h += d_h_prev[time + 1];
    //get the error into the output (z), it's the error from ahead in the network
    //as well as from the previous output of the cell

    //cout << "d_z: " << d_z << endl;
    //cout << "ld_z_cap[" << time << "]: " << ld_z_cap[time] << endl;

    d_h_prev[time] = d_h * z[time];

    double d_z = ((d_h * h_prev) - (d_h * h_tanh[time])) * ld_z[time];
    d_z_bias[time] = d_z;
    d_zu[time] = d_z * h_prev;
    d_h_prev[time] += d_z * zu;
    d_zw[time] = d_z * x;
    d_input[time] = d_z * zw;

    double d_h_tanh = (1 - z[time]) * d_h * ld_h_tanh[time];

    d_input[time] += d_h_tanh * hw;
    d_hw[time] = d_h_tanh * x;

    d_h_bias[time] = d_h_tanh;

    d_hu[time] = d_h_tanh * r[time] * h_prev;
    double d_r = d_h_tanh * hu * h_prev * ld_r[time];

    d_h_prev[time] += d_h_tanh * hu * r[time];

    d_r_bias[time] = d_r;
    d_ru[time] = d_r * h_prev;
    d_h_prev[time] += d_r * ru;

    d_rw[time] = d_r * x;
    d_input[time] += d_r * rw;

    //cout << "d_input: " << d_input[time] << endl;
    //cout << "d_beta2: " << d_beta2[time] << endl;

    //cout << endl << endl;

    //reset the reset gate bias to be around 0
    //r_bias -= 1.0;
}

void GRU_Node::error_fired(int time, double error) {
    outputs_fired[time]++;

    error_values[time] *= error;

    try_update_deltas(time);
}

void GRU_Node::output_fired(int time, double delta) {
    outputs_fired[time]++;

    error_values[time] += delta;

    try_update_deltas(time);
}


void GRU_Node::print_cell_values() {
    /*
    cerr << "\tinput_value: " << input_value << endl;
    cerr << "\tinput_gate_value: " << input_gate_value << ", input_gate_update_weight: " << input_gate_update_weight << ", input_gate_bias: " << input_gate_bias << endl;
    cerr << "\toutput_gate_value: " << output_gate_value << ", output_gate_update_weight: " << output_gate_update_weight << ", output_gate_bias: " << output_gate_bias << endl;
    cerr << "\tforget_gate_value: " << forget_gate_value << ", forget_gate_update_weight: " << forget_gate_update_weight << "\tforget_gate_bias: " << forget_gate_bias << endl;
    cerr << "\tcell_value: " << cell_value << ", cell_bias: " << cell_bias << endl;
    */
}


uint32_t GRU_Node::get_number_weights() const {
    return NUMBER_GRU_WEIGHTS;
}

void GRU_Node::get_weights(vector<double> &parameters) const {
    parameters.resize(get_number_weights());
    uint32_t offset = 0;
    get_weights(offset, parameters);
}

void GRU_Node::set_weights(const vector<double> &parameters) {
    uint32_t offset = 0;
    set_weights(offset, parameters);
}


void GRU_Node::set_weights(uint32_t &offset, const vector<double> &parameters) {
    //uint32_t start_offset = offset;

    zw = parameters[offset++];
    zu = parameters[offset++];
    z_bias = parameters[offset++];

    rw = parameters[offset++];
    ru = parameters[offset++];
    r_bias = parameters[offset++];

    hw = parameters[offset++];
    hu = parameters[offset++];
    h_bias = parameters[offset++];


    //uint32_t end_offset = offset;

    //cerr << "set weights from offset " << start_offset << " to " << end_offset << " on GRU_Node " << innovation_number << endl;
}

void GRU_Node::get_weights(uint32_t &offset, vector<double> &parameters) const {
    //uint32_t start_offset = offset;

    parameters[offset++] = zw;
    parameters[offset++] = zu;
    parameters[offset++] = z_bias;

    parameters[offset++] = rw;
    parameters[offset++] = ru;
    parameters[offset++] = r_bias;

    parameters[offset++] = hw;
    parameters[offset++] = hu;
    parameters[offset++] = h_bias;

    //uint32_t end_offset = offset;

    //cerr << "set weights from offset " << start_offset << " to " << end_offset << " on GRU_Node " << innovation_number << endl;
}


void GRU_Node::get_gradients(vector<double> &gradients) {
    gradients.assign(NUMBER_GRU_WEIGHTS, 0.0);

    for (uint32_t i = 0; i < NUMBER_GRU_WEIGHTS; i++) {
        gradients[i] = 0.0;
    }

    for (uint32_t i = 0; i < series_length; i++) {
        gradients[0] += d_zw[i];
        gradients[1] += d_zu[i];
        gradients[2] += d_z_bias[i];

        gradients[3] += d_rw[i];
        gradients[4] += d_ru[i];
        gradients[5] += d_r_bias[i];

        gradients[6] += d_hw[i];
        gradients[7] += d_hu[i];
        gradients[8] += d_h_bias[i];
    }
}

void GRU_Node::reset(int _series_length) {
    series_length = _series_length;

    d_zw.assign(series_length, 0.0);
    d_zu.assign(series_length, 0.0);
    d_z_bias.assign(series_length, 0.0);

    d_rw.assign(series_length, 0.0);
    d_ru.assign(series_length, 0.0);
    d_r_bias.assign(series_length, 0.0);

    d_hw.assign(series_length, 0.0);
    d_hu.assign(series_length, 0.0);
    d_h_bias.assign(series_length, 0.0);

    d_h_prev.assign(series_length, 0.0);

    z.assign(series_length, 0.0);
    ld_z.assign(series_length, 0.0);
    r.assign(series_length, 0.0);
    ld_r.assign(series_length, 0.0);
    h_tanh.assign(series_length, 0.0);
    ld_h_tanh.assign(series_length, 0.0);

    //reset values from rnn_node_interface
    d_input.assign(series_length, 0.0);
    error_values.assign(series_length, 0.0);

    input_values.assign(series_length, 0.0);
    output_values.assign(series_length, 0.0);

    inputs_fired.assign(series_length, 0);
    outputs_fired.assign(series_length, 0);
}

RNN_Node_Interface* GRU_Node::copy() const {
    GRU_Node* n = new GRU_Node(innovation_number, type, depth);

    //cout << "COPYING!" << endl;

    //copy GRU_Node values
    n->zw = zw;
    n->zu = zu;
    n->z_bias = z_bias;
    n->rw = rw;
    n->ru = ru;
    n->r_bias = r_bias;
    n->hw = hw;
    n->hu = hu;
    n->h_bias = h_bias;

    n->d_zw = d_zw;
    n->d_zu = d_zu;
    n->d_z_bias = d_z_bias;
    n->d_rw = d_rw;
    n->d_ru = d_ru;
    n->d_r_bias = d_r_bias;
    n->d_hw = d_hw;
    n->d_hu = d_hu;
    n->d_h_bias = d_h_bias;

    n->d_h_prev = d_h_prev;

    n->z = z;
    n->ld_z = ld_z;
    n->r = r;
    n->ld_r = ld_r;
    n->h_tanh = h_tanh;
    n->ld_h_tanh = ld_h_tanh;

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
