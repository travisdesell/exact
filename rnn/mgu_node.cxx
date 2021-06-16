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
#include "mgu_node.hxx"


#define NUMBER_MGU_WEIGHTS 6

MGU_Node::MGU_Node(int _innovation_number, int _layer_type, double _depth) : RNN_Node_Interface(_innovation_number, _layer_type, _depth) {
    node_type = MGU_NODE;
}

MGU_Node::~MGU_Node() {
}

void MGU_Node::initialize_lamarckian(minstd_rand0 &generator, NormalDistribution &normal_distribution, double mu, double sigma) {

    fw = bound(normal_distribution.random(generator, mu, sigma));
    fu = bound(normal_distribution.random(generator, mu, sigma));
    f_bias = bound(normal_distribution.random(generator, mu, sigma));

    hw = bound(normal_distribution.random(generator, mu, sigma));
    hu = bound(normal_distribution.random(generator, mu, sigma));
    h_bias = bound(normal_distribution.random(generator, mu, sigma));
}

void MGU_Node::initialize_xavier(minstd_rand0 &generator, uniform_real_distribution<double> &rng_1_1, double range) {

    fw = range * (rng_1_1(generator));
    fu = range * (rng_1_1(generator));
    f_bias = range * (rng_1_1(generator));

    hw = range * (rng_1_1(generator));
    hu = range * (rng_1_1(generator));
    h_bias = range * (rng_1_1(generator));
    
}

void MGU_Node::initialize_kaiming(minstd_rand0 &generator, NormalDistribution &normal_distribution, double range){
    fw = range * normal_distribution.random(generator, 0, 1);
    fu = range * normal_distribution.random(generator, 0, 1);
    f_bias = range * normal_distribution.random(generator, 0, 1);

    hw = range * normal_distribution.random(generator, 0, 1);
    hu = range * normal_distribution.random(generator, 0, 1);
    h_bias = range * normal_distribution.random(generator, 0, 1);
}

void MGU_Node::initialize_uniform_random(minstd_rand0 &generator, uniform_real_distribution<double> &rng) {
    fw = rng(generator);
    fu = rng(generator);
    f_bias = rng(generator);

    hw = rng(generator);
    hu = rng(generator);
    h_bias = rng(generator);
}

double MGU_Node::get_gradient(string gradient_name) {
    double gradient_sum = 0.0;

    for (uint32_t i = 0; i < series_length; i++ ) {
        if (gradient_name == "fw") {
            gradient_sum += d_fw[i];
        } else if (gradient_name == "fu") {
            gradient_sum += d_fu[i];
        } else if (gradient_name == "f_bias") {
            gradient_sum += d_f_bias[i];
        } else if (gradient_name == "hw") {
            gradient_sum += d_hw[i];
        } else if (gradient_name == "hu") {
            gradient_sum += d_hu[i];
        } else if (gradient_name == "h_bias") {
            gradient_sum += d_h_bias[i];
        } else {
            Log::fatal("ERROR: tried to get unknown gradient: '%s'\n", gradient_name.c_str());
            exit(1);
        }
    }

    return gradient_sum;
}

void MGU_Node::print_gradient(string gradient_name) {
    Log::info("\tgradient['%s']: %lf\n", gradient_name.c_str(), get_gradient(gradient_name));
}

void MGU_Node::input_fired(int time, double incoming_output) {
    inputs_fired[time]++;

    input_values[time] += incoming_output;

    if (inputs_fired[time] < total_inputs) return;
    else if (inputs_fired[time] > total_inputs) {
        Log::fatal("ERROR: inputs_fired on MGU_Node %d at time %d is %d and total_inputs is %d\n", innovation_number, time, inputs_fired[time], total_inputs);
        exit(1);
    }

    //update the reset gate bias so its centered around 1
    //r_bias += 1;

    double x = input_values[time];

    double h_prev = 0.0;
    if (time > 0) h_prev = output_values[time - 1];

    double hfu = h_prev * fu;
    double xfw = x * fw;
    double f_sum = f_bias + hfu + xfw;
    f[time] = sigmoid(f_sum);
    ld_f[time] = sigmoid_derivative(f[time]);

    double xhw = x * hw;
    double hu_f_h_prev = hu * f[time] * h_prev;
    double h_sum = h_bias + xhw + hu_f_h_prev;

    h_tanh[time] = tanh(h_sum);
    ld_h_tanh[time] = tanh_derivative(h_tanh[time]);

    output_values[time] = (1 - f[time]) * h_prev   +   f[time] * h_tanh[time];
}

void MGU_Node::try_update_deltas(int time) {
    if (outputs_fired[time] < total_outputs) return;
    else if (outputs_fired[time] > total_outputs) {
        Log::fatal("ERROR: outputs_fired on MGU_Node %d at time %d is %d and total_outputs is %d\n:", innovation_number, time, outputs_fired[time], total_outputs);
        exit(1);
    }

    double error = error_values[time];

    double x = input_values[time];

    double h_prev = 0.0;
    if (time > 0) h_prev = output_values[time - 1];

    //backprop output gate
    double d_out = error;
    if (time < (series_length - 1)) d_out += d_h_prev[time + 1];


    d_h_prev[time] = d_out * (1-f[time]);

    double d_h_tanh  = d_out * f[time] * ld_h_tanh[time];
    d_h_bias[time]   = d_h_tanh;
    d_hw[time]       = d_h_tanh * x;
    d_hu[time]       = d_h_tanh * f[time] * h_prev;
    d_input[time]    += d_h_tanh * hw;
    d_h_prev[time]   += d_h_tanh * hu * f[time];

    double d_f_sigmoid  = ((d_out * h_tanh[time]) - (d_out * h_prev));
    d_f_sigmoid         += d_h_tanh * hu * h_prev;

    double d_f = d_f_sigmoid * ld_f[time];

    d_f_bias[time]  = d_f;
    d_fu[time]      = d_f * h_prev;
    d_fw[time]      = d_f * x;
    d_input[time]   += d_f * fw;
    d_h_prev[time]  += d_f * fu;

}

void MGU_Node::error_fired(int time, double error) {
    outputs_fired[time]++;

    error_values[time] *= error;

    try_update_deltas(time);
}

void MGU_Node::output_fired(int time, double delta) {
    outputs_fired[time]++;

    error_values[time] += delta;

    try_update_deltas(time);
}


uint32_t MGU_Node::get_number_weights() const {
    return NUMBER_MGU_WEIGHTS;
}

void MGU_Node::get_weights(vector<double> &parameters) const {
    parameters.resize(get_number_weights());
    uint32_t offset = 0;
    get_weights(offset, parameters);
}

void MGU_Node::set_weights(const vector<double> &parameters) {
    uint32_t offset = 0;
    set_weights(offset, parameters);
}


void MGU_Node::set_weights(uint32_t &offset, const vector<double> &parameters) {
    //uint32_t start_offset = offset;

    fw = bound(parameters[offset++]);
    fu = bound(parameters[offset++]);
    f_bias = bound(parameters[offset++]);

    hw = bound(parameters[offset++]);
    hu = bound(parameters[offset++]);
    h_bias = bound(parameters[offset++]);


    //uint32_t end_offset = offset;
    //Log::trace("set weights from offset %d to %d on MGU_Node %d\n", start_offset, end_offset, innovation_number);
}

void MGU_Node::get_weights(uint32_t &offset, vector<double> &parameters) const {
    //uint32_t start_offset = offset;

    parameters[offset++] = fw;
    parameters[offset++] = fu;
    parameters[offset++] = f_bias;

    parameters[offset++] = hw;
    parameters[offset++] = hu;
    parameters[offset++] = h_bias;

    //uint32_t end_offset = offset;
    //Log::trace("got weights from offset %d to %d on MGU_Node %d\n", start_offset, end_offset, innovation_number);
}


void MGU_Node::get_gradients(vector<double> &gradients) {
    gradients.assign(NUMBER_MGU_WEIGHTS, 0.0);

    for (uint32_t i = 0; i < NUMBER_MGU_WEIGHTS; i++) {
        gradients[i] = 0.0;
    }

    for (uint32_t i = 0; i < series_length; i++) {
        gradients[0] += d_fw[i];
        gradients[1] += d_fu[i];
        gradients[2] += d_f_bias[i];
        gradients[3] += d_hw[i];
        gradients[4] += d_hu[i];
        gradients[5] += d_h_bias[i];
    }
}

void MGU_Node::reset(int _series_length) {
    series_length = _series_length;

    d_fw.assign(series_length, 0.0);
    d_fu.assign(series_length, 0.0);
    d_f_bias.assign(series_length, 0.0);

    d_hw.assign(series_length, 0.0);
    d_hu.assign(series_length, 0.0);
    d_h_bias.assign(series_length, 0.0);

    d_h_prev.assign(series_length, 0.0);

    f.assign(series_length, 0.0);
    ld_f.assign(series_length, 0.0);
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

RNN_Node_Interface* MGU_Node::copy() const {
    MGU_Node* n = new MGU_Node(innovation_number, layer_type, depth);

    //copy MGU_Node values
    n->fw = fw;
    n->fu = fu;
    n->f_bias = f_bias;
    n->hw = hw;
    n->hu = hu;
    n->h_bias = h_bias;

    n->d_fw = d_fw;
    n->d_fu = d_fu;
    n->d_f_bias = d_f_bias;
    n->d_hw = d_hw;
    n->d_hu = d_hu;
    n->d_h_bias = d_h_bias;

    n->d_h_prev = d_h_prev;

    n->f = f;
    n->ld_f = ld_f;
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

void MGU_Node::write_to_stream(ostream &out) {
    RNN_Node_Interface::write_to_stream(out);
}
