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


GRU_Node::GRU_Node(int _innovation_number, int _type, double _depth) : RNN_Node_Interface(_innovation_number, _type, _depth) {
    node_type = GRU_NODE;
}

GRU_Node::~GRU_Node() {
}

double Bound(double value) {
    if (value < -10.0) value = -10.0;
    else if (value > 10.0) value = 10.0;
    return value;
}

void GRU_Node::initialize_randomly(minstd_rand0 &generator, NormalDistribution &normal_distribution, double mu, double sigma) {

    update_gate_update_weight = Bound(normal_distribution.random(generator, mu, sigma));
    update_gate_weight        = Bound(normal_distribution.random(generator, mu, sigma));
    update_gate_bias          = Bound(normal_distribution.random(generator, mu, sigma));

    reset_gate_update_weight  = Bound(normal_distribution.random(generator, mu, sigma));
    reset_gate_weight         = Bound(normal_distribution.random(generator, mu, sigma));
    reset_gate_bias           = Bound(normal_distribution.random(generator, mu, sigma));
    // reset_gate_bias        = 1.0 + normal_distribution.random(generator, mu, sigma);

    memory_gate_update_weight = Bound(normal_distribution.random(generator, mu, sigma));
    memory_gate_weight        = Bound(normal_distribution.random(generator, mu, sigma));
    memory_gate_bias          = Bound(normal_distribution.random(generator, mu, sigma));


}

double GRU_Node::get_gradient(string gradient_name) {
    double gradient_sum = 0.0;

    for (uint32_t i = 0; i < series_length; i++ ) {

        if (gradient_name == "update_gate_update_weight") {
            gradient_sum += d_update_gate_update_weight[i];
        } else if (gradient_name == "update_gate_weight") {
            gradient_sum += d_update_gate_weight[i];
        } else if (gradient_name == "update_gate_bias") {
            gradient_sum += d_update_gate_bias[i];

        } else if (gradient_name == "reset_gate_update_weight") {
            gradient_sum += d_reset_gate_update_weight[i];
        } else if (gradient_name == "reset_gate_weight") {
            gradient_sum += d_reset_gate_weight[i];
        } else if (gradient_name == "reset_gate_bias") {
            gradient_sum += d_reset_gate_bias[i];

        } else if (gradient_name == "memory_gate_update_weight") {
            gradient_sum += d_memory_gate_update_weight[i];
        } else if (gradient_name == "memory_gate_weight") {
            gradient_sum += d_memory_gate_weight[i];
        } else if (gradient_name == "memory_gate_bias") {
            gradient_sum += d_memory_gate_bias[i];
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

    double input_value = input_values[time];
    //cout << "input value[" << i << "]:" << input_value << endl;

    double previous_out_value = 0.0;
    if (time > 0) previous_out_value = output_values[time - 1];

    update_gate_values[time] = sigmoid(update_gate_weight * input_value  + update_gate_update_weight                              * previous_out_value + update_gate_bias);
    reset_gate_values[time]  = sigmoid(reset_gate_weight  * input_value  + reset_gate_update_weight                               * previous_out_value + reset_gate_bias);
    memory_gate_values[time] = tanh(memory_gate_weight    * input_value  + memory_gate_update_weight * reset_gate_values[time]    * previous_out_value + memory_gate_bias);

    ld_update_gate[time] = sigmoid_derivative(update_gate_values[time]);
    ld_reset_gate[time]  = sigmoid_derivative(reset_gate_values[time]);
    ld_memory_gate[time] = tanh_derivative(memory_gate_values[time]);

    output_values[time] = previous_out_value * update_gate_values[time]    +    (1 - update_gate_values[time]) * memory_gate_values[time];
}


void GRU_Node::try_update_deltas(int time) {
    if (outputs_fired[time] < total_outputs) return;
    else if (outputs_fired[time] > total_outputs) {
        cerr << "ERROR: outputs_fired on GRU_Node " << innovation_number << " at time " << time << " is " << outputs_fired[time] << " and total_outputs is " << total_outputs << endl;
        exit(1);
    }

    double error = error_values[time];
    double input_value = input_values[time];
    //cout << "input value[" << i << "]:" << input_value << endl;

    double previous_out_value = 0.00;
    if (time > 0) previous_out_value = output_values[time - 1];
    //previous_out_value = 0.33;
    //cout << "previous_out_value[" << i << "]: " << previous_out_value << endl;


    //backprop output gate
    double d_update_gate              = error * (previous_out_value - memory_gate_values[time]) * ld_update_gate[time];
    d_update_gate_bias[time]          = d_update_gate;
    d_update_gate_update_weight[time] = d_update_gate  * previous_out_value;
    d_update_gate_weight[time]        = d_update_gate  * input_value;
    d_prev_out[time]                  += d_update_gate * update_gate_update_weight;
    d_input[time]                     += d_update_gate * update_gate_weight;

    //backprop memory gate
    double d_memory_gate              = error * (1 - update_gate_values[time]) * ld_memory_gate[time];
    d_memory_gate_bias[time]          = d_memory_gate;
    d_memory_gate_update_weight[time] = d_memory_gate  * reset_gate_values[time] * previous_out_value;
    d_memory_gate_weight[time]        = d_memory_gate  * input_value;
    d_prev_out[time]                  += d_memory_gate * memory_gate_update_weight * reset_gate_values[time];
    d_input[time]                     += d_memory_gate * memory_gate_weight;

    //backprob Reset gate
    double d_reset_gate               = d_memory_gate * memory_gate_update_weight * previous_out_value * ld_reset_gate[time];
    d_reset_gate_bias[time]           = d_reset_gate;
    d_reset_gate_update_weight[time]  = d_reset_gate * previous_out_value;
    d_reset_gate_weight[time]         = d_reset_gate * input_value;
    d_prev_out[time]                  += d_reset_gate * reset_gate_update_weight;
    d_input[time]                     += d_reset_gate * reset_gate_weight;

    d_prev_out[time]                  += error * update_gate_values[time];

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
    return 9;
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

double get_bias(){
  return 999;
}

void GRU_Node::set_weights(uint32_t &offset, const vector<double> &parameters) {
    //uint32_t start_offset = offset;

    update_gate_update_weight = parameters[offset++];
    update_gate_weight = parameters[offset++];
    update_gate_bias = parameters[offset++];

    reset_gate_update_weight = parameters[offset++];
    reset_gate_weight = parameters[offset++];
    reset_gate_bias = parameters[offset++];

    memory_gate_update_weight = parameters[offset++];
    memory_gate_weight = parameters[offset++];
    memory_gate_bias = parameters[offset++];

    //uint32_t end_offset = offset;
    //cerr << "set weights from offset " << start_offset << " to " << end_offset << " on GRU_Node " << innovation_number << endl;
}

void GRU_Node::get_weights(uint32_t &offset, vector<double> &parameters) const {
    //uint32_t start_offset = offset;

    parameters[offset++] = update_gate_update_weight;
    parameters[offset++] = update_gate_weight;
    parameters[offset++] = update_gate_bias;

    parameters[offset++] = reset_gate_update_weight;
    parameters[offset++] = reset_gate_weight;
    parameters[offset++] = reset_gate_bias;

    parameters[offset++] = memory_gate_update_weight;
    parameters[offset++] = memory_gate_weight;
    parameters[offset++] = memory_gate_bias;

    //uint32_t end_offset = offset;
    //cerr << "set weights from offset " << start_offset << " to " << end_offset << " on GRU_Node " << innovation_number << endl;
}


void GRU_Node::get_gradients(vector<double> &gradients) {
    gradients.assign(9, 0.0);

    for (uint32_t i = 0; i < 9; i++) {
        gradients[i] = 0.0;
    }

    for (uint32_t i = 0; i < series_length; i++) {
        gradients[0] += d_update_gate_update_weight[i];
        gradients[1] += d_update_gate_weight[i];
        gradients[2] += d_update_gate_bias[i];

        gradients[3] += d_reset_gate_update_weight[i];
        gradients[4] += d_reset_gate_weight[i];
        gradients[5] += d_reset_gate_bias[i];

        gradients[6] += d_memory_gate_update_weight[i];
        gradients[7] += d_memory_gate_weight[i];
        gradients[8] += d_memory_gate_bias[i];
    }
}

void GRU_Node::reset(int _series_length) {
    series_length = _series_length;

    ld_update_gate.assign(series_length, 0.0);
    ld_reset_gate.assign(series_length, 0.0);
    ld_memory_gate.assign(series_length, 0.0);

    d_input.assign(series_length, 0.0);
    d_prev_out.assign(series_length, 0.0);

    d_update_gate_update_weight.assign(series_length, 0.0);
    d_update_gate_weight.assign(series_length, 0.0);
    d_update_gate_bias.assign(series_length, 0.0);

    d_reset_gate_update_weight.assign(series_length, 0.0);
    d_reset_gate_weight.assign(series_length, 0.0);
    d_reset_gate_bias.assign(series_length, 0.0);

    d_memory_gate_update_weight.assign(series_length, 0.0);
    d_memory_gate_weight.assign(series_length, 0.0);
    d_memory_gate_bias.assign(series_length, 0.0);

    update_gate_values.assign(series_length, 0.0);
    reset_gate_values.assign(series_length, 0.0);
    memory_gate_values.assign(series_length, 0.0);

    error_values.assign(series_length, 0.0);

    input_values.assign(series_length, 0.0);
    output_values.assign(series_length, 0.0);

    inputs_fired.assign(series_length, 0);
    outputs_fired.assign(series_length, 0);
}

RNN_Node_Interface* GRU_Node::copy() const {
    GRU_Node* n = new GRU_Node(innovation_number, type, depth);

    //copy GRU_Node values
    n->update_gate_update_weight = update_gate_update_weight;
    n->update_gate_weight        = update_gate_weight;
    n->update_gate_bias          = update_gate_bias;

    n->reset_gate_update_weight  = reset_gate_update_weight;
    n->reset_gate_weight         = reset_gate_weight;
    n->reset_gate_bias           = reset_gate_bias;

    n->memory_gate_update_weight = memory_gate_update_weight;
    n->memory_gate_weight        = memory_gate_weight;
    n->memory_gate_bias          = memory_gate_bias;

    n->update_gate_values        = update_gate_values;
    n->reset_gate_values         = reset_gate_values;
    n->memory_gate_values        = memory_gate_values;

    n->ld_update_gate            = ld_update_gate;
    n->ld_reset_gate             = ld_reset_gate;
    n->ld_memory_gate            = ld_memory_gate;

    n->d_prev_out                = d_prev_out;

    n->d_update_gate_update_weight = d_update_gate_update_weight;
    n->d_update_gate_weight        = d_update_gate_weight;
    n->d_update_gate_bias          = d_update_gate_bias;

    n->d_reset_gate_update_weight  = d_reset_gate_update_weight;
    n->d_reset_gate_weight         = d_reset_gate_weight;
    n->d_reset_gate_bias           = d_reset_gate_bias;

    n->d_memory_gate_update_weight = d_memory_gate_update_weight;
    n->d_memory_gate_weight        = d_memory_gate_weight;
    n->d_memory_gate_bias          = d_memory_gate_bias;


    //copy RNN_Node_Interface values
    n->series_length = series_length;
    n->input_values = input_values;
    n->output_values = output_values;
    n->error_values = error_values;
    n->d_input      = d_input;

    n->inputs_fired = inputs_fired;
    n->total_inputs = total_inputs;
    n->outputs_fired = outputs_fired;
    n->total_outputs = total_outputs;
    n->enabled = enabled;
    n->forward_reachable = forward_reachable;
    n->backward_reachable = backward_reachable;

    return n;
}

#ifdef GRU_TEST
#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <string>
using std::string;

int main(int argc, char **argv) {

    int number_of_weights = 9 * 2;
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
    parameter_names.push_back("update_gate_update_weight");
    parameter_names.push_back("update_gate_weight");
    parameter_names.push_back("update_gate_bias");

    parameter_names.push_back("reset_gate_update_weight");
    parameter_names.push_back("reset_gate_weight");
    parameter_names.push_back("reset_gate_bias");

    parameter_names.push_back("memory_gate_update_weight");
    parameter_names.push_back("memory_gate_weight");
    parameter_names.push_back("memory_gate_bias");

    GRU_Node *node1 = new GRU_Node(0, RNN_HIDDEN_NODE, 0.0);
    GRU_Node *node2 = new GRU_Node(0, RNN_HIDDEN_NODE, 0.0);
    GRU_Node *n11  = new GRU_Node(0, RNN_HIDDEN_NODE, 0.0);
    GRU_Node *n12  = new GRU_Node(0, RNN_HIDDEN_NODE, 0.0);
    GRU_Node *n21 = new GRU_Node(0, RNN_HIDDEN_NODE, 0.0);
    GRU_Node *n22 = new GRU_Node(0, RNN_HIDDEN_NODE, 0.0);
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

        if (print) cout << "\n\niteration: " << setw(5) << iteration << endl;

        //cout << "firing node1" << endl;
        offset = 0;
        node1->reset(inputs.size());
        node2->reset(inputs.size());
        node1->set_weights(offset, parameters);
        node2->set_weights(offset, parameters);

        for (int time = 0; time < outputs.size(); time++) {
            node1->input_fired(time, inputs[time]);
            node2->input_fired(time, node1->output_values[time]);
        }

        get_mse(node2->output_values, outputs, mse, deltas);

        for (int time = outputs.size() - 1; time >= 0; time--) {
            node2->output_fired(time, deltas[time]);
            node1->output_fired(time, node2->d_input[time]);
        }

        if (print) {
            cout << "mean squared error: " << mse << endl;

            cout << "outputs: " << endl;
            for (uint32_t i = 0; i < inputs.size(); i++) {
                cout << "\toutput[" << i << "]: " << node2->output_values[i] << ", expected[" << i << "]: " << outputs[i] << ", diff: " << node2->output_values[i] - outputs[i] << endl;
            }

            cout << "analytical gradient:" << endl;
            for (uint32_t i = 0; i < parameter_names.size(); i++) {
                node1->print_gradient(parameter_names[i % 9]);
            }
            for (uint32_t i = 0; i < parameter_names.size(); i++) {
                node2->print_gradient(parameter_names[i % 9]);
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
            n12->reset(inputs.size());
            n11->set_weights(offset, parameters);
            n12->set_weights(offset, parameters);

            for (int time = 0; time < outputs.size(); time++) {
                n11->input_fired(time, inputs[time]);
                n12->input_fired(time, n11->output_values[time]);
            }

            get_mse(n12->output_values, outputs, mse1, deltas);

            parameters[i] = save + diff;

            offset = 0;
            n21->reset(inputs.size());
            n22->reset(inputs.size());
            n21->set_weights(offset, parameters);
            n22->set_weights(offset, parameters);

            for (int time = 0; time < outputs.size(); time++) {
                n21->input_fired(time, inputs[time]);
                n22->input_fired(time, n21->output_values[time]);
            }
            get_mse(n22->output_values, outputs, mse2, deltas);

            double gradient = (mse2 - mse1) / (2.0 * diff);

            gradient *= mse;
            if (print) cout << "\tgradient['" << parameter_names[i % 9] << "']: " << gradient << endl;

            parameters[i] = save;
        }

        for (int32_t j = 0; j < parameters.size(); j++) {
            if (j < 9) {
                next_parameters[j] -= learning_rate * node1->get_gradient(parameter_names[j % 9]);
            } else {
                next_parameters[j] -= learning_rate * node2->get_gradient(parameter_names[j % 9]);
            }
        }
        parameters = next_parameters;
    }
}

void GRU_Node::write_to_stream(ostream &out) {
    RNN_Node_Interface::write_to_stream(out);
}

#endif
