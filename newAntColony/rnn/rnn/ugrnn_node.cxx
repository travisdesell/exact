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
#include "ugrnn_node.hxx"


#define NUMBER_UGRNN_WEIGHTS 6

UGRNN_Node::UGRNN_Node(int _innovation_number, int _type, double _depth) : RNN_Node_Interface(_innovation_number, _type, _depth) {
    node_type = UGRNN_NODE;
}

UGRNN_Node::~UGRNN_Node() {
}

void UGRNN_Node::initialize_randomly(minstd_rand0 &generator, NormalDistribution &normal_distribution, double mu, double sigma) {

    cw = bound(normal_distribution.random(generator, mu, sigma));
    ch = bound(normal_distribution.random(generator, mu, sigma));
    c_bias = bound(normal_distribution.random(generator, mu, sigma));

    gw = bound(normal_distribution.random(generator, mu, sigma));
    gh = bound(normal_distribution.random(generator, mu, sigma));
    g_bias = bound(normal_distribution.random(generator, mu, sigma));
}

double UGRNN_Node::get_gradient(string gradient_name) {
    double gradient_sum = 0.0;

    for (uint32_t i = 0; i < series_length; i++ ) {
        if (gradient_name == "cw") {
            gradient_sum += d_cw[i];
        } else if (gradient_name == "ch") {
            gradient_sum += d_ch[i];
        } else if (gradient_name == "c_bias") {
            gradient_sum += d_c_bias[i];
        } else if (gradient_name == "gw") {
            gradient_sum += d_gw[i];
        } else if (gradient_name == "gh") {
            gradient_sum += d_gh[i];
        } else if (gradient_name == "g_bias") {
            gradient_sum += d_g_bias[i];
        } else {
            cerr << "ERROR: tried to get unknown gradient: '" << gradient_name << "'" << endl;
            exit(1);
        }
    }

    return gradient_sum;
}

void UGRNN_Node::print_gradient(string gradient_name) {
    cout << "\tgradient['" << gradient_name << "']: " << get_gradient(gradient_name) << endl;
}

void UGRNN_Node::input_fired(int time, double incoming_output) {
    inputs_fired[time]++;

    input_values[time] += incoming_output;

    if (inputs_fired[time] < total_inputs) return;
    else if (inputs_fired[time] > total_inputs) {
        cerr << "ERROR: inputs_fired on UGRNN_Node " << innovation_number << " at time " << time << " is " << inputs_fired[time] << " and total_inputs is " << total_inputs << endl;
        exit(1);
    }

    //update the reset gate bias so its centered around 1
    //g_bias += 1;

    //cout << "PROPAGATING FORWARD" << endl;

    double x = input_values[time];
    //cout << "node " << innovation_number << " - input value[" << time << "] (x): " << x << endl;

    double h_prev = 0.0;
    if (time > 0) h_prev = output_values[time - 1];
    //cout << "node " << innovation_number << " - prev_output_value[" << time << "] (h_prev): " << h_prev << endl;

    //cout << "g_bias: " << g_bias << endl;

    double xcw = x * cw;
    double hch = h_prev * ch;
    double c_sum = xcw + hch + c_bias;
    c[time] = tanh(c_sum);
    ld_c[time] = tanh_derivative(c[time]);

    double xgw = x * gw;
    double hgh = h_prev * gh;
    double g_sum = xgw + hgh + g_bias;

    g[time] = sigmoid(g_sum);
    ld_g[time] = sigmoid_derivative(g[time]);

    output_values[time] = (g[time] * h_prev) + ((1 - g[time]) * c[time]);

    //reset alpha, beta1, beta2 so they don't mess with mean/stddev calculations for
    //parameter generation
    //g_bias -= 1.0;

    //cout << "node " << innovation_number << " - output_values[" << time << "]: " << output_values[time] << endl;
}

void UGRNN_Node::try_update_deltas(int time) {
    if (outputs_fired[time] < total_outputs) return;
    else if (outputs_fired[time] > total_outputs) {
        cerr << "ERROR: outputs_fired on UGRNN_Node " << innovation_number << " at time " << time << " is " << outputs_fired[time] << " and total_outputs is " << total_outputs << endl;
        exit(1);
    }

    //cout << "PROPAGATING BACKWARDS" << endl;
    //update the reset gate bias so its centered around 1   
    //g_bias += 1.0;

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

    d_h_prev[time] = d_h * g[time];

    double d_g = ((d_h * h_prev) - (d_h * c[time])) * ld_g[time];
    d_g_bias[time] = d_g;
    d_gh[time] = d_g * h_prev;
    d_h_prev[time] += d_g * gh;
    d_gw[time] = d_g * x;
    d_input[time] = d_g * gw;

    double d_c = (1 - g[time]) * d_h * ld_c[time];

    d_input[time] += d_c * cw;
    d_cw[time] = d_c * x;

    d_c_bias[time] = d_c;

    d_ch[time] = d_c * h_prev;
    d_h_prev[time] += d_c * ch;

    //cout << "d_input: " << d_input[time] << endl;
    //cout << "d_beta2: " << d_beta2[time] << endl;

    //cout << endl << endl;

    //reset the reset gate bias to be around 0
    //g_bias -= 1.0;
}

void UGRNN_Node::error_fired(int time, double error) {
    outputs_fired[time]++;

    error_values[time] *= error;

    try_update_deltas(time);
}

void UGRNN_Node::output_fired(int time, double delta) {
    outputs_fired[time]++;

    error_values[time] += delta;

    try_update_deltas(time);
}


void UGRNN_Node::print_cell_values() {
    /*
    cerr << "\tinput_value: " << input_value << endl;
    cerr << "\tinput_gate_value: " << input_gate_value << ", input_gate_update_weight: " << input_gate_update_weight << ", input_gate_bias: " << input_gate_bias << endl;
    cerr << "\toutput_gate_value: " << output_gate_value << ", output_gate_update_weight: " << output_gate_update_weight << ", output_gate_bias: " << output_gate_bias << endl;
    cerr << "\tforget_gate_value: " << forget_gate_value << ", forget_gate_update_weight: " << forget_gate_update_weight << "\tforget_gate_bias: " << forget_gate_bias << endl;
    cerr << "\tcell_value: " << cell_value << ", cell_bias: " << cell_bias << endl;
    */
}


uint32_t UGRNN_Node::get_number_weights() const {
    return NUMBER_UGRNN_WEIGHTS;
}

void UGRNN_Node::get_weights(vector<double> &parameters) const {
    parameters.resize(get_number_weights());
    uint32_t offset = 0;
    get_weights(offset, parameters);
}

void UGRNN_Node::set_weights(const vector<double> &parameters) {
    uint32_t offset = 0;
    set_weights(offset, parameters);
}


void UGRNN_Node::set_weights(uint32_t &offset, const vector<double> &parameters) {
    //uint32_t start_offset = offset;

    cw = parameters[offset++];
    ch = parameters[offset++];
    c_bias = parameters[offset++];

    gw = parameters[offset++];
    gh = parameters[offset++];
    g_bias = parameters[offset++];

    //uint32_t end_offset = offset;

    //cerr << "set weights from offset " << start_offset << " to " << end_offset << " on UGRNN_Node " << innovation_number << endl;
}

void UGRNN_Node::get_weights(uint32_t &offset, vector<double> &parameters) const {
    //uint32_t start_offset = offset;

    parameters[offset++] = cw;
    parameters[offset++] = ch;
    parameters[offset++] = c_bias;

    parameters[offset++] = gw;
    parameters[offset++] = gh;
    parameters[offset++] = g_bias;

    //uint32_t end_offset = offset;

    //cerr << "set weights from offset " << start_offset << " to " << end_offset << " on UGRNN_Node " << innovation_number << endl;
}


void UGRNN_Node::get_gradients(vector<double> &gradients) {
    gradients.assign(NUMBER_UGRNN_WEIGHTS, 0.0);

    for (uint32_t i = 0; i < NUMBER_UGRNN_WEIGHTS; i++) {
        gradients[i] = 0.0;
    }

    for (uint32_t i = 0; i < series_length; i++) {
        gradients[0] += d_cw[i];
        gradients[1] += d_ch[i];
        gradients[2] += d_c_bias[i];

        gradients[3] += d_gw[i];
        gradients[4] += d_gh[i];
        gradients[5] += d_g_bias[i];
    }
}

void UGRNN_Node::reset(int _series_length) {
    series_length = _series_length;

    d_cw.assign(series_length, 0.0);
    d_ch.assign(series_length, 0.0);
    d_c_bias.assign(series_length, 0.0);

    d_gw.assign(series_length, 0.0);
    d_gh.assign(series_length, 0.0);
    d_g_bias.assign(series_length, 0.0);

    d_h_prev.assign(series_length, 0.0);

    c.assign(series_length, 0.0);
    ld_c.assign(series_length, 0.0);
    g.assign(series_length, 0.0);
    ld_g.assign(series_length, 0.0);

    //reset values from rnn_node_interface
    d_input.assign(series_length, 0.0);
    error_values.assign(series_length, 0.0);

    input_values.assign(series_length, 0.0);
    output_values.assign(series_length, 0.0);

    inputs_fired.assign(series_length, 0);
    outputs_fired.assign(series_length, 0);
}

RNN_Node_Interface* UGRNN_Node::copy() const {
    UGRNN_Node* n = new UGRNN_Node(innovation_number, layer_type, depth);

    //cout << "COPYING!" << endl;

    //copy UGRNN_Node values
    n->cw = cw;
    n->ch = ch;
    n->c_bias = c_bias;
    n->gw = gw;
    n->gh = gh;
    n->g_bias = g_bias;

    n->d_cw = d_cw;
    n->d_ch = d_ch;
    n->d_c_bias = d_c_bias;
    n->d_gw = d_gw;
    n->d_gh = d_gh;
    n->d_g_bias = d_g_bias;

    n->d_h_prev = d_h_prev;

    n->c = c;
    n->ld_c = ld_c;
    n->g = g;
    n->ld_g = ld_g;

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

void UGRNN_Node::write_to_stream(ostream &out) {
    RNN_Node_Interface::write_to_stream(out);
}

