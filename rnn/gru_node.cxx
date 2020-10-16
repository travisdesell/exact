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
#include "gru_node.hxx"


#define NUMBER_GRU_WEIGHTS 9

GRU_Node::GRU_Node(int _innovation_number, int _type, double _depth) : RNN_Node_Interface(_innovation_number, _type, _depth) {
    node_type = GRU_NODE;
}

GRU_Node::~GRU_Node() {
}

void GRU_Node::initialize_lamarckian(minstd_rand0 &generator, NormalDistribution &normal_distribution, double mu, double sigma) {

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

void GRU_Node::initialize_xavier(minstd_rand0 &generator, uniform_real_distribution<double> &rng_1_1, double range) {
    
    zw = range * (rng_1_1(generator));
    zu = range * (rng_1_1(generator));
    z_bias = range * (rng_1_1(generator));

    rw = range * (rng_1_1(generator));
    ru = range * (rng_1_1(generator));
    r_bias = range * (rng_1_1(generator));

    hw = range * (rng_1_1(generator));
    hu = range * (rng_1_1(generator));
    h_bias = range * (rng_1_1(generator));

}

void GRU_Node::initialize_kaiming(minstd_rand0 &generator, NormalDistribution &normal_distribution, double range){
    zw = range * normal_distribution.random(generator, 0, 1);
    zu = range * normal_distribution.random(generator, 0, 1);
    z_bias = range * normal_distribution.random(generator, 0, 1);

    rw = range * normal_distribution.random(generator, 0, 1);
    ru = range * normal_distribution.random(generator, 0, 1);
    r_bias = range * normal_distribution.random(generator, 0, 1);

    hw = range * normal_distribution.random(generator, 0, 1);
    hu = range * normal_distribution.random(generator, 0, 1);
    h_bias = range * normal_distribution.random(generator, 0, 1);
}

void GRU_Node::initialize_uniform_random(minstd_rand0 &generator, uniform_real_distribution<double> &rng) {
    zw = rng(generator);
    zu = rng(generator);
    z_bias = rng(generator);

    rw = rng(generator);
    ru = rng(generator);
    r_bias = rng(generator);

    hw = rng(generator);
    hu = rng(generator);
    h_bias = rng(generator);
    
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
            Log::fatal("ERROR: tried to get unknown gradient: '%s'\n", gradient_name.c_str()); 
            exit(1);
        }
    }

    return gradient_sum;
}

void GRU_Node::print_gradient(string gradient_name) {
    Log::info("\tgradient['%s']: %lf\n", gradient_name.c_str(), get_gradient(gradient_name));
}

void GRU_Node::input_fired(int time, double incoming_output) {
    inputs_fired[time]++;

    input_values[time] += incoming_output;

    if (inputs_fired[time] < total_inputs) return;
    else if (inputs_fired[time] > total_inputs) {
        Log::fatal("ERROR: inputs_fired on GRU_Node %d at time %d is %d and total_inputs is %d\n", innovation_number, time, inputs_fired[time], total_inputs);
        exit(1);
    }

    //update the reset gate bias so its centered around 1
    //r_bias += 1;

    double x = input_values[time];

    double h_prev = 0.0;
    if (time > 0) h_prev = output_values[time - 1];

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

    //r_bias so it doesn't mess with mean/stddev calculations for
    //parameter generation
    //r_bias -= 1.0;
}

void GRU_Node::try_update_deltas(int time) {
    if (outputs_fired[time] < total_outputs) return;
    else if (outputs_fired[time] > total_outputs) {
        Log::fatal("ERROR: outputs_fired on GRU_Node %d at time %d is %d and total_outputs is %d\n", innovation_number, time, outputs_fired[time], total_outputs);
        exit(1);
    }

    //update the reset gate bias so its centered around 1   
    //r_bias += 1.0;

    double error = error_values[time];
    double x = input_values[time];

    double h_prev = 0.0;
    if (time > 0) h_prev = output_values[time - 1];

    //backprop output gate
    double d_h = error;
    if (time < (series_length - 1)) d_h += d_h_prev[time + 1];
    //get the error into the output (z), it's the error from ahead in the network
    //as well as from the previous output of the cell

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

    zw = bound(parameters[offset++]);
    zu = bound(parameters[offset++]);
    z_bias = bound(parameters[offset++]);

    rw = bound(parameters[offset++]);
    ru = bound(parameters[offset++]);
    r_bias = bound(parameters[offset++]);

    hw = bound(parameters[offset++]);
    hu = bound(parameters[offset++]);
    h_bias = bound(parameters[offset++]);


    //uint32_t end_offset = offset;
    //Log::trace("set weights from offset %d to %d on GRU_Node %d\n", start_offset, end_offset, innovation_number);
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
    //Log::trace("got weights from offset %d to %d on GRU_Node %d\n", start_offset, end_offset, innovation_number);
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
    GRU_Node* n = new GRU_Node(innovation_number, layer_type, depth);

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

void GRU_Node::write_to_stream(ostream &out) {
    RNN_Node_Interface::write_to_stream(out);
}

