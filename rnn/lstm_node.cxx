#include <cmath>

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <vector>
using std::vector;

#include "rnn_node_interface.hxx"
#include "lstm_node.hxx"


LSTM_Node::LSTM_Node(int _innovation_number, int _type) : RNN_Node_Interface(_innovation_number, _type) {
}

void LSTM_Node::input_fired() {
    inputs_fired++;

    if (inputs_fired < total_inputs) return;
    else if (inputs_fired > total_inputs) {
        cerr << "ERROR: inputs_fired on LSTM_Node " << innovation_number << " is " << inputs_fired << " and total_inputs is " << total_inputs << endl;
        exit(1);
    }

    //cerr << "activating LSTM node: " << innovation_number << ", type: " << type << endl;

    input_gate_value = sigmoid(input_gate_weight * input_value + input_gate_update_weight * previous_cell_value + input_gate_bias);
    output_gate_value = sigmoid(output_gate_weight * input_value + output_gate_update_weight * previous_cell_value + output_gate_bias);
    forget_gate_value = sigmoid(forget_gate_weight * input_value + forget_gate_update_weight * previous_cell_value + forget_gate_bias);

    //if (bound_value(input_gate_value)) ld_input_gate = 0;
    //if (bound_value(output_gate_value)) ld_output_gate = 0;
    //if (bound_value(forget_gate_value)) ld_forget_gate = 0;

    ld_input_gate = sigmoid_derivative(input_gate_value);
    ld_output_gate = sigmoid_derivative(output_gate_value);
    ld_forget_gate = sigmoid_derivative(forget_gate_value);

    //previous_cell_value = cell_value;

    cell_in_tanh = tanh(cell_weight * input_value + cell_bias);
    cell_value = forget_gate_value * previous_cell_value + input_gate_value * cell_in_tanh;

    ld_cell_in = tanh_derivative(cell_in_tanh);

    //if (bound_value(cell_value)) ld_cell_in = 0;

    //The original is a hyperbolic tangent, but the peephole[clarification needed] LSTM paper suggests the activation function be linear -- activation(x) = x

    cell_out_tanh = tanh(cell_value);
    output_value = output_gate_value * cell_out_tanh;
    ld_cell_out = tanh_derivative(cell_out_tanh);

    //output = output_gate_value * sigmoid(cell_value);
    //output = output_gate_value * sigmoid(cell_value);

    if (isnan(output_value) || isinf(output_value)) {
        cerr << "ERROR: output_value became " << output_value << " on LSTM node: " << innovation_number << endl;
        print_cell_values();
        exit(1);
    }

    if (isnan(forget_gate_value) || isinf(forget_gate_value)) {
        cerr << "ERROR: forget_gate_value became " << forget_gate_value << " on LSTM node: " << innovation_number << endl;
        print_cell_values();
        exit(1);
    }

    if (isnan(input_gate_value) || isinf(input_gate_value)) {
        cerr << "ERROR: input_gate_value became " << input_gate_value << " on LSTM node: " << innovation_number << endl;
        print_cell_values();
        exit(1);
    }

    //if (type == RNN_OUTPUT_NODE) cerr << "output: " << output_value << endl;
}


void LSTM_Node::print_cell_values() {
    cerr << "\tinput_value: " << input_value << endl;
    cerr << "\tinput_gate_value: " << input_gate_value << ", input_gate_update_weight: " << input_gate_update_weight << ", input_gate_bias: " << input_gate_bias << endl;
    cerr << "\toutput_gate_value: " << output_gate_value << ", output_gate_update_weight: " << output_gate_update_weight << ", output_gate_bias: " << output_gate_bias << endl;
    cerr << "\tforget_gate_value: " << forget_gate_value << ", forget_gate_update_weight: " << forget_gate_update_weight << "\tforget_gate_bias: " << forget_gate_bias << endl;
    cerr << "\tcell_value: " << cell_value << ", previous_cell_value: " << previous_cell_value << ", cell_bias: " << cell_bias << endl;
}

void LSTM_Node::output_fired() {
    //this gonna be fun

    outputs_fired++;

    if (outputs_fired < total_outputs) return;
    else if (outputs_fired > total_outputs) {
        cerr << "ERROR: outputs_fired on LSTM_Node " << innovation_number << " is " << outputs_fired << " and total_outputs is " << outputs_fired << endl;
        exit(1);
    }

}

uint32_t LSTM_Node::get_number_weights() {
    return 11;
}

void LSTM_Node::set_weights(uint32_t &offset, const vector<double> &parameters) {
    //uint32_t start_offset = offset;

    output_gate_update_weight = parameters[offset++];
    output_gate_weight = parameters[offset++];
    output_gate_bias = parameters[offset++];

    input_gate_update_weight = parameters[offset++];
    input_gate_weight = parameters[offset++];
    input_gate_bias = parameters[offset++];

    forget_gate_update_weight = parameters[offset++];
    forget_gate_weight = parameters[offset++];
    forget_gate_bias = parameters[offset++];

    cell_weight = parameters[offset++];
    cell_bias = parameters[offset++];

    //uint32_t end_offset = offset;

    //cerr << "set weights from offset " << start_offset << " to " << end_offset << " on LSTM_Node " << innovation_number << endl;
}

void LSTM_Node::reset() {
    input_value = 0.0;
    output_value = 0.0;

    inputs_fired = 0;
    outputs_fired = 0;
}

void LSTM_Node::full_reset() {
    input_value = 0.0;
    output_value = 0.0;

    input_gate_value = 0.0;
    output_gate_value = 0.0;
    forget_gate_value = 0.0;
    cell_value = 0.0;
    previous_cell_value = 0.0;

    inputs_fired = 0;
    outputs_fired = 0;
}


#ifdef LSTM_TEST
#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <string>
using std::string;

int main(int argc, char **argv) {
    LSTM_Node *node = new LSTM_Node(0, RNN_HIDDEN_NODE);

    int number_of_weights = 11;
    vector<double> parameters(number_of_weights, 0.0);
    vector<double> next_parameters;
    vector<double> min_bound(number_of_weights, -1.0);
    vector<double> max_bound(number_of_weights, 1.0);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    minstd_rand0 generator(seed);
    for (uint32_t i = 0; i < parameters.size(); i++) {
        uniform_real_distribution<double> rng(min_bound[i], max_bound[i]);
        parameters[i] = rng(generator);
    }

    vector<string> parameter_names;
    parameter_names.push_back("output_gate_update_weight");
    parameter_names.push_back("output_gate_weight");
    parameter_names.push_back("output_gate_bias");

    parameter_names.push_back("input_gate_update_weight");
    parameter_names.push_back("input_gate_weight");
    parameter_names.push_back("input_gate_bias");

    parameter_names.push_back("forget_gate_update_weight");
    parameter_names.push_back("forget_gate_weight");
    parameter_names.push_back("forget_gate_bias");

    parameter_names.push_back("cell_weight");
    parameter_names.push_back("cell_bias");

    double expected_value = -1.0;

    for (uint32_t iteration = 0; iteration < 100; iteration++) {
        cout << "\n\niteration: " << iteration << endl;
        uint32_t offset = 0;
        node->full_reset();
        node->set_weights(offset, parameters);
        node->input_value = 0.93;
        node->total_inputs = 1;
        node->input_fired();

        double error = node->output_value - expected_value;

        next_parameters = parameters;

        cout << "empriical gradient:" << endl;

        double diff = 0.00001;
        double save;
        for (uint32_t i = 0; i < parameters.size(); i++) {
            save = parameters[i];
            parameters[i] -= diff;

            offset = 0;
            LSTM_Node *n1 = new LSTM_Node(0, RNN_HIDDEN_NODE);
            n1->full_reset();
            n1->set_weights(offset, parameters);
            n1->input_value = 0.93;
            n1->total_inputs = 1;
            n1->input_fired();

            parameters[i] += 2.0 * diff;

            offset = 0;
            LSTM_Node *n2 = new LSTM_Node(0, RNN_HIDDEN_NODE);
            n2->full_reset();
            n2->set_weights(offset, parameters);
            n2->input_value = 0.93;
            n2->total_inputs = 1;
            n2->input_fired();
            
            double gradient = (n2->output_value - n1->output_value) / (2.0 * diff);

            //cout << "node->output_value: " << node->output_value << endl;
            //cout << "n1->output_value:   " << n1->output_value << endl;
            //cout << "n2->output_value:   " << n2->output_value << endl;
            cout << "\tgradient['" << parameter_names[i] << "']: " << gradient * error << endl;
            parameters[i] = save;
            next_parameters[i] -= 0.001 * gradient * error;
        }

        cout << "analytical gradient:" << endl;
        //d : delta
        //ld : local derivative

        double d_prev_cell = 0.0;
        double d_input = 0.0;

        //backprop output gate
        double d_output_gate = error * node->cell_out_tanh * node->ld_output_gate;
        double d_output_gate_bias = d_output_gate;
        double d_output_gate_update_weight = d_output_gate * node->previous_cell_value;
        double d_output_gate_weight = d_output_gate * node->input_value;
        d_prev_cell += d_output_gate * node->output_gate_update_weight;
        d_input += d_output_gate * node->output_gate_weight;


        //backprop the cell path

        double d_cell_out = error * node->output_gate_value * node->ld_cell_out;

        //backprop forget gate
        double d_forget_gate = d_cell_out * node->previous_cell_value * node->ld_forget_gate; 
        double d_forget_gate_bias = d_forget_gate;
        double d_forget_gate_update_weight = d_forget_gate * node->previous_cell_value;
        double d_forget_gate_weight = d_forget_gate * node->input_value;
        d_prev_cell += d_forget_gate * node->forget_gate_update_weight;
        d_input += d_forget_gate * node->forget_gate_weight;

        //backprob input gate
        double d_input_gate = d_cell_out * node->cell_in_tanh * node->ld_input_gate;
        double d_input_gate_bias = d_input_gate;
        double d_input_gate_update_weight = d_input_gate * node->previous_cell_value;
        double d_input_gate_weight = d_input_gate * node->input_value;
        d_prev_cell += d_input_gate * node->input_gate_update_weight;
        d_input += d_input_gate * node->input_gate_weight;


        //backprop cell input
        double d_cell_in = d_cell_out * node->input_gate_value * node->ld_cell_in;
        double d_cell_bias = d_cell_in;
        double d_cell_weight = d_cell_in * node->input_value;
        d_input += d_cell_in * node->cell_weight;

        cout << "\tgradient['output_gate_update_weight']: " << d_output_gate_update_weight << endl;
        cout << "\tgradient['output_gate_weight']: " << d_output_gate_weight << endl;
        cout << "\tgradient['output_gate_bias']: " << d_output_gate_bias << endl;

        cout << "\tgradient['input_gate_update_weight']: " << d_input_gate_update_weight << endl;
        cout << "\tgradient['input_gate_weight']: " << d_input_gate_weight << endl;
        cout << "\tgradient['input_gate_bias']: " << d_input_gate_bias << endl;

        cout << "\tgradient['forget_gate_update_weight']: " << d_forget_gate_update_weight << endl;
        cout << "\tgradient['forget_gate_weight']: " << d_forget_gate_weight << endl;
        cout << "\tgradient['forget_gate_bias']: " << d_forget_gate_bias << endl;

        cout << "\tgradient['cell_weight']: " << d_cell_weight << endl;
        cout << "\tgradient['cell_bias']: " << d_cell_bias << endl;


        cout << "error: " << error << endl;

        parameters = next_parameters;

    }
}

#endif

