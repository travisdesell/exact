#include <cmath>

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "rnn_node_interface.hxx"
#include "mse.hxx"
#include "lstm_node.hxx"


LSTM_Node::LSTM_Node(int _innovation_number, int _type) : RNN_Node_Interface(_innovation_number, _type) {
}

double LSTM_Node::get_gradient(string gradient_name) {
    double gradient_sum = 0.0;

    for (uint32_t i = 0; i < series_length; i++ ) {
        if (gradient_name == "output_gate_update_weight") {
            gradient_sum += d_output_gate_update_weight[i];
        } else if (gradient_name == "output_gate_weight") {
            gradient_sum += d_output_gate_weight[i];
        } else if (gradient_name == "output_gate_bias") {
            gradient_sum += d_output_gate_bias[i];
        } else if (gradient_name == "input_gate_update_weight") {
            gradient_sum += d_input_gate_update_weight[i];
        } else if (gradient_name == "input_gate_weight") {
            gradient_sum += d_input_gate_weight[i];
        } else if (gradient_name == "input_gate_bias") {
            gradient_sum += d_input_gate_bias[i];
        } else if (gradient_name == "forget_gate_update_weight") {
            gradient_sum += d_forget_gate_update_weight[i];
        } else if (gradient_name == "forget_gate_weight") {
            gradient_sum += d_forget_gate_weight[i];
        } else if (gradient_name == "forget_gate_bias") {
            gradient_sum += d_forget_gate_bias[i];
        } else if (gradient_name == "cell_weight") {
            gradient_sum += d_cell_weight[i];
        } else if (gradient_name == "cell_bias") {
            gradient_sum += d_cell_bias[i];
        } else {
            cerr << "ERROR: tried to get unknown gradient: '" << gradient_name << "'" << endl;
            exit(1);
        }
    }

    return gradient_sum;
}

void LSTM_Node::print_gradient(string gradient_name) {
    cout << "\tgradient['" << gradient_name << "']: " << get_gradient(gradient_name) << endl;
}

void LSTM_Node::input_fired(const vector<double> &incoming_outputs) {
    inputs_fired++;

    for (int32_t i = 0; i < series_length; i++) {
        input_values[i] += incoming_outputs[i];
    }

    if (inputs_fired < total_inputs) return;
    else if (inputs_fired > total_inputs) {
        cerr << "ERROR: inputs_fired on LSTM_Node " << innovation_number << " is " << inputs_fired << " and total_inputs is " << total_inputs << endl;
        exit(1);
    }

    for (int32_t i = 0; i < series_length; i++) {
        double input_value = input_values[i];
        //cout << "input value[" << i << "]:" << input_value << endl;

        double previous_cell_value = 0.0;
        if (i > 0) previous_cell_value = cell_values[i - 1];
        //previous_cell_value = 0.33;
        //cout << "previous_cell_value[" << i << "]: " << previous_cell_value << endl;

        output_gate_values[i] = sigmoid(output_gate_weight * input_value + output_gate_update_weight * previous_cell_value + output_gate_bias);
        input_gate_values[i] = sigmoid(input_gate_weight * input_value + input_gate_update_weight * previous_cell_value + input_gate_bias);
        forget_gate_values[i] = sigmoid(forget_gate_weight * input_value + forget_gate_update_weight * previous_cell_value + forget_gate_bias);

        ld_output_gate[i] = sigmoid_derivative(output_gate_values[i]);
        ld_input_gate[i] = sigmoid_derivative(input_gate_values[i]);
        ld_forget_gate[i] = sigmoid_derivative(forget_gate_values[i]);

        /*
        output_gate_values[i] = output_gate_weight * input_value + output_gate_update_weight * previous_cell_value + output_gate_bias;
        input_gate_values[i] = input_gate_weight * input_value + input_gate_update_weight * previous_cell_value + input_gate_bias;
        forget_gate_values[i] = forget_gate_weight * input_value + forget_gate_update_weight * previous_cell_value + forget_gate_bias;

        ld_output_gate[i] = 1.0;
        ld_input_gate[i] = 1.0;
        ld_forget_gate[i] = 1.0;
        */

        cell_in_tanh[i] = tanh(cell_weight * input_value + cell_bias);
        ld_cell_in[i] = tanh_derivative(cell_in_tanh[i]);

        cell_values[i] = (forget_gate_values[i] * previous_cell_value) + (input_gate_values[i] * cell_in_tanh[i]);

        //The original is a hyperbolic tangent, but the peephole[clarification needed] LSTM paper suggests the activation function be linear -- activation(x) = x

        cell_out_tanh[i] = cell_values[i];
        ld_cell_out[i] = 1.0;
        //cell_out_tanh[i] = tanh(cell_values[i]);
        //ld_cell_out[i] = tanh_derivative(cell_out_tanh[i]);

        output_values[i] = output_gate_values[i] * cell_out_tanh[i];
    }
}


void LSTM_Node::print_cell_values() {
    /*
    cerr << "\tinput_value: " << input_value << endl;
    cerr << "\tinput_gate_value: " << input_gate_value << ", input_gate_update_weight: " << input_gate_update_weight << ", input_gate_bias: " << input_gate_bias << endl;
    cerr << "\toutput_gate_value: " << output_gate_value << ", output_gate_update_weight: " << output_gate_update_weight << ", output_gate_bias: " << output_gate_bias << endl;
    cerr << "\tforget_gate_value: " << forget_gate_value << ", forget_gate_update_weight: " << forget_gate_update_weight << "\tforget_gate_bias: " << forget_gate_bias << endl;
    cerr << "\tcell_value: " << cell_value << ", cell_bias: " << cell_bias << endl;
    */
}

void LSTM_Node::try_update_deltas() {
    if (outputs_fired < total_outputs) return;
    else if (outputs_fired > total_outputs) {
        cerr << "ERROR: outputs_fired on LSTM_Node " << innovation_number << " is " << outputs_fired << " and total_outputs is " << outputs_fired << endl;
        exit(1);
    }

    for (int32_t i = series_length - 1; i >= 0; i--) {
        double error = error_values[i];
        double input_value = input_values[i];
        //cout << "input value[" << i << "]:" << input_value << endl;

        double previous_cell_value = 0.00;
        if (i > 0) previous_cell_value = cell_values[i - 1];
        //previous_cell_value = 0.33;
        //cout << "previous_cell_value[" << i << "]: " << previous_cell_value << endl;


        //backprop output gate
        double d_output_gate = error * cell_out_tanh[i] * ld_output_gate[i];
        d_output_gate_bias[i] = d_output_gate;
        d_output_gate_update_weight[i] = d_output_gate * previous_cell_value;
        d_output_gate_weight[i] = d_output_gate * input_value;
        d_prev_cell[i] += d_output_gate * output_gate_update_weight;
        d_input[i] += d_output_gate * output_gate_weight;

        //backprop the cell path

        double d_cell_out = error * output_gate_values[i] * ld_cell_out[i];
        //propagate error back from the next cell value if there is one
        if (i < (series_length - 1)) d_cell_out += d_prev_cell[i + 1];

        //backprop forget gate
        d_prev_cell[i] += d_cell_out * forget_gate_values[i];

        double d_forget_gate = d_cell_out * previous_cell_value * ld_forget_gate[i]; 
        d_forget_gate_bias[i] = d_forget_gate;
        d_forget_gate_update_weight[i] = d_forget_gate * previous_cell_value;
        d_forget_gate_weight[i] = d_forget_gate * input_value;
        d_prev_cell[i] += d_forget_gate * forget_gate_update_weight;
        d_input[i] += d_forget_gate * forget_gate_weight;

        //backprob input gate
        double d_input_gate = d_cell_out * cell_in_tanh[i] * ld_input_gate[i];
        d_input_gate_bias[i] = d_input_gate;
        d_input_gate_update_weight[i] = d_input_gate * previous_cell_value;
        d_input_gate_weight[i] = d_input_gate * input_value;
        d_prev_cell[i] += d_input_gate * input_gate_update_weight;
        d_input[i] += d_input_gate * input_gate_weight;


        //backprop cell input
        double d_cell_in = d_cell_out * input_gate_values[i] * ld_cell_in[i];
        d_cell_bias[i] = d_cell_in;
        d_cell_weight[i] = d_cell_in * input_value;
        d_input[i] += d_cell_in * cell_weight;
    }
}

void LSTM_Node::output_fired(double error) {
    outputs_fired++;

    for (int32_t i = 0; i < series_length; i++) {
        error_values[i] *= error;
    }

    try_update_deltas();
}

void LSTM_Node::output_fired(const vector<double> &deltas) {
    outputs_fired++;

    for (int32_t i = 0; i < series_length; i++) {
        error_values[i] += deltas[i];
    }

    try_update_deltas();
}

uint32_t LSTM_Node::get_number_weights() {
    return 11;
}

void LSTM_Node::set_weights(const vector<double> &parameters) {
    uint32_t offset = 0;
    set_weights(offset, parameters);
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

void LSTM_Node::get_weights(uint32_t &offset, vector<double> &parameters) {
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

    //cerr << "set weights from offset " << start_offset << " to " << end_offset << " on LSTM_Node " << innovation_number << endl;
}


void LSTM_Node::get_gradients(vector<double> &gradients) {
    gradients.assign(11, 0.0);

    for (uint32_t i = 0; i < 11; i++) {
        gradients[i] = 0.0;
    }

    for (uint32_t i = 0; i < series_length; i++) {
        gradients[0] += d_output_gate_update_weight[i];
        gradients[1] += d_output_gate_weight[i];
        gradients[2] += d_output_gate_bias[i];

        gradients[3] += d_input_gate_update_weight[i];
        gradients[4] += d_input_gate_weight[i];
        gradients[5] += d_input_gate_bias[i];

        gradients[6] += d_forget_gate_update_weight[i];
        gradients[7] += d_forget_gate_weight[i];
        gradients[8] += d_forget_gate_bias[i];

        gradients[9] += d_cell_weight[i];
        gradients[10] += d_cell_bias[i];
    }
}

void LSTM_Node::reset(int _series_length) {
    series_length = _series_length;

    ld_output_gate.assign(series_length, 0.0);
    ld_input_gate.assign(series_length, 0.0);
    ld_forget_gate.assign(series_length, 0.0);

    cell_in_tanh.assign(series_length, 0.0);
    cell_out_tanh.assign(series_length, 0.0);
    ld_cell_in.assign(series_length, 0.0);
    ld_cell_out.assign(series_length, 0.0);

    d_input.assign(series_length, 0.0);
    d_prev_cell.assign(series_length, 0.0);

    d_output_gate_update_weight.assign(series_length, 0.0);
    d_output_gate_weight.assign(series_length, 0.0);
    d_output_gate_bias.assign(series_length, 0.0);

    d_input_gate_update_weight.assign(series_length, 0.0);
    d_input_gate_weight.assign(series_length, 0.0);
    d_input_gate_bias.assign(series_length, 0.0);

    d_forget_gate_update_weight.assign(series_length, 0.0);
    d_forget_gate_weight.assign(series_length, 0.0);
    d_forget_gate_bias.assign(series_length, 0.0);

    d_cell_weight.assign(series_length, 0.0);
    d_cell_bias.assign(series_length, 0.0);

    output_gate_values.assign(series_length, 0.0);
    input_gate_values.assign(series_length, 0.0);
    forget_gate_values.assign(series_length, 0.0);
    cell_values.assign(series_length, 0.0);

    error_values.assign(series_length, 0.0);

    input_values.assign(series_length, 0.0);
    output_values.assign(series_length, 0.0);

    inputs_fired = 0;
    outputs_fired = 0;
}

RNN_Node_Interface* LSTM_Node::copy() {
    LSTM_Node* n = new LSTM_Node(innovation_number, type);

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

    n->ld_output_gate = ld_output_gate;
    n->ld_input_gate = ld_input_gate;
    n->ld_forget_gate = ld_forget_gate;

    n->cell_in_tanh = cell_in_tanh;
    n->cell_out_tanh = cell_out_tanh;
    n->ld_cell_in = ld_cell_in;
    n->ld_cell_out = ld_cell_out;

    n->d_prev_cell = d_prev_cell;

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

    return n;
}

#ifdef LSTM_TEST
#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <string>
using std::string;

int main(int argc, char **argv) {

    int number_of_weights = 11 * 2;
    vector<double> parameters(number_of_weights, 0.0);
    vector<double> next_parameters;
    vector<double> min_bound(number_of_weights, -2.0);
    vector<double> max_bound(number_of_weights, 2.0);

    //unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    unsigned seed = 1337;
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

    LSTM_Node *node1 = new LSTM_Node(0, RNN_HIDDEN_NODE);
    LSTM_Node *node2 = new LSTM_Node(0, RNN_HIDDEN_NODE);
    LSTM_Node *n11  = new LSTM_Node(0, RNN_HIDDEN_NODE);
    LSTM_Node *n12  = new LSTM_Node(0, RNN_HIDDEN_NODE);
    LSTM_Node *n21 = new LSTM_Node(0, RNN_HIDDEN_NODE);
    LSTM_Node *n22 = new LSTM_Node(0, RNN_HIDDEN_NODE);
    node1->total_inputs = 1;
    node1->total_outputs = 1;
    node2->total_inputs = 1;
    node2->total_outputs = 1;
    n11->total_inputs = 1;
    n12->total_inputs = 1;
    n21->total_inputs = 1;
    n22->total_inputs = 1;

    uint32_t offset;
    double mse;
    vector<double> deltas;

    vector<double> inputs;
    inputs.push_back(-0.88);
    inputs.push_back(-0.83);
    inputs.push_back(-0.89);
    inputs.push_back(-0.87);

    vector<double> outputs;
    outputs.push_back(0.73);
    outputs.push_back(0.63);
    outputs.push_back(0.59);
    outputs.push_back(0.55);

    for (uint32_t iteration = 0; iteration < 2000000; iteration++) {
        bool print = (iteration % 10000) == 0;

        if (print) cout << "\n\niteration: " << iteration << endl;
       
        //cout << "firing node1" << endl;
        offset = 0;
        node1->reset(inputs.size());
        node1->set_weights(offset, parameters);
        node1->input_fired(inputs);
        node2->reset(inputs.size());
        node2->set_weights(offset, parameters);
        node2->input_fired(node1->output_values);
        get_mse(node2->output_values, outputs, mse, deltas);
        node2->output_fired(deltas);
        node1->output_fired(node2->d_input);

        if (print) {
            cout << "mean squared error: " << mse << endl;

            cout << "outputs: " << endl;
            for (uint32_t i = 0; i < inputs.size(); i++) {
                cout << "\toutput[" << i << "]: " << node2->output_values[i] << ", expected[" << i << "]: " << outputs[i] << ", diff: " << node2->output_values[i] - outputs[i] << endl;
            }

            cout << "analytical gradient:" << endl;
            for (uint32_t i = 0; i < parameter_names.size(); i++) {
                node1->print_gradient(parameter_names[i % 11]);
            }
            for (uint32_t i = 0; i < parameter_names.size(); i++) {
                node2->print_gradient(parameter_names[i % 11]);
            }
        }

        next_parameters = parameters;

        if (print) cout << "empirical gradient:" << endl;

        double learning_rate = 0.01;
        double diff = 0.00001;
        double save;
        double mse1, mse2;
        for (uint32_t i = 0; i < parameters.size(); i++) {
            save = parameters[i];
            parameters[i] = save - diff;

            offset = 0;
            n11->reset(inputs.size());
            n11->set_weights(offset, parameters);
            n11->input_fired(inputs);
            n12->reset(inputs.size());
            n12->set_weights(offset, parameters);
            n12->input_fired(n11->output_values);
            get_mse(n12->output_values, outputs, mse1, deltas);

            parameters[i] = save + diff;

            offset = 0;
            n21->reset(inputs.size());
            n21->set_weights(offset, parameters);
            n21->input_fired(inputs);
            n22->reset(inputs.size());
            n22->set_weights(offset, parameters);
            n22->input_fired(n21->output_values);
            get_mse(n22->output_values, outputs, mse2, deltas);

            double gradient = (mse2 - mse1) / (2.0 * diff);

            gradient *= mse;
            if (print) cout << "\tgradient['" << parameter_names[i % 11] << "']: " << gradient << endl;

            parameters[i] = save;
        }

        for (int32_t j = 0; j < parameters.size(); j++) {
            if (j < 11) {
                next_parameters[j] -= learning_rate * node1->get_gradient(parameter_names[j % 11]);
            } else {
                next_parameters[j] -= learning_rate * node2->get_gradient(parameter_names[j % 11]);
            }
        }
        parameters = next_parameters;
    }
}

#endif

