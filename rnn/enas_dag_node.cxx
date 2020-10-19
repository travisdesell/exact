
#include <cmath>

#include <fstream>
using std::ostream;
using std::getline;
using std::ifstream;

#include<iostream>
using std::cout;
using std::endl;


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
#include "enas_dag_node.hxx"

#define NUMBER_ENAS_DAG_WEIGHTS 10


ENAS_DAG_Node::ENAS_DAG_Node(int _innovation_number, int _type, double _depth) : RNN_Node_Interface(_innovation_number, _type, _depth) {
  node_type = ENAS_DAG_NODE;
}

ENAS_DAG_Node::~ENAS_DAG_Node(){

}

void ENAS_DAG_Node::initialize_lamarckian(minstd_rand0 &generator, NormalDistribution &normal_distribution, double mu, double sigma) {

    zw = bound(normal_distribution.random(generator, mu, sigma));
    rw = bound(normal_distribution.random(generator, mu, sigma));

    int assigned_node_weights = 2; // 2 weights for the starting node assigned above

    for (int new_node_weight = 0; new_node_weight < NUMBER_ENAS_DAG_WEIGHTS - assigned_node_weights; ++new_node_weight){
        weights.at(new_node_weight)= bound(normal_distribution.random(generator, mu, sigma));
    }

}

void ENAS_DAG_Node::initialize_xavier(minstd_rand0 &generator, uniform_real_distribution<double> &rng_1_1, double range) {
    zw = range * (rng_1_1(generator));
    rw = range * (rng_1_1(generator));

    int assigned_node_weights = 2; // 2 weights for the starting node assigned above

    for (int new_node_weight = 0; new_node_weight < NUMBER_ENAS_DAG_WEIGHTS - assigned_node_weights; ++new_node_weight){
        weights.at(new_node_weight) = range * (rng_1_1(generator));
    }

}

void ENAS_DAG_Node::initialize_kaiming(minstd_rand0 &generator, NormalDistribution &normal_distribution, double range) {
    zw = range * normal_distribution.random(generator, 0, 1);
    rw = range * normal_distribution.random(generator, 0, 1);

    int assigned_node_weights = 2; // 2 weights for the starting node assigned above

    for (int new_node_weight = 0; new_node_weight < NUMBER_ENAS_DAG_WEIGHTS - assigned_node_weights; ++new_node_weight){
        weights.at(new_node_weight) = range * normal_distribution.random(generator, 0, 1);
    }

}

void ENAS_DAG_Node::initialize_uniform_random(minstd_rand0 &generator, uniform_real_distribution<double> &rng) {
    zw = rng(generator);
    rw = rng(generator);

    int assigned_node_weights = 2; // 2 weights for the starting node assigned above

    for (int new_node_weight = 0; new_node_weight < NUMBER_ENAS_DAG_WEIGHTS - assigned_node_weights; ++new_node_weight){
        weights.at(new_node_weight) = rng(generator);
    }
}


double ENAS_DAG_Node::get_gradient(string gradient_name) {
    double gradient_sum = 0.0;
    for (uint32_t i = 0; i < series_length; i++ ) {
        if (gradient_name == "zw") {
            gradient_sum += d_zw[i];
        } else if (gradient_name == "rw") {
            gradient_sum += d_rw[i];
        } else if (gradient_name == "w1") {
            gradient_sum += d_weights[0][i];

        } else if (gradient_name == "w2") {
            gradient_sum += d_weights[1][i];
        } else if (gradient_name == "w3") {
            gradient_sum += d_weights[2][i];
        } else if (gradient_name == "w4") {
            gradient_sum += d_weights[3][i];
        } else if (gradient_name == "w5") {
            gradient_sum +=  d_weights[4][i];
        } else if (gradient_name == "w6") {
            gradient_sum += d_weights[5][i];
        } else if (gradient_name == "w7") {
            gradient_sum +=  d_weights[6][i];
        } else if (gradient_name == "w8") {
            gradient_sum +=  d_weights[7][i];
        } else {
            Log::fatal("ERROR: tried to get unknown gradient: '%s'\n", gradient_name.c_str()); 
            exit(1);
        }
    }
    return gradient_sum;
}

void ENAS_DAG_Node::print_gradient(string gradient_name) {
    Log::info("\tgradient['%s']: %lf\n", gradient_name.c_str(), get_gradient(gradient_name));
}

double ENAS_DAG_Node::activation(double value, int act_operator) {
    if (act_operator == 0) return sigmoid(value);
    if (act_operator == 1) return tanh(value);
    if (act_operator == 2) return swish(value);
    if (act_operator == 3) return leakyReLU(value);
    if (act_operator == 4) return identity(value);

    Log::fatal("ERROR: invalid act_operator: %d\n", act_operator); 
    exit(1);
}

double ENAS_DAG_Node::activation_derivative(double value, double input, int act_operator) {
    if (act_operator == 0) return sigmoid_derivative(input);
    if (act_operator == 1) return tanh_derivative(input);
    if (act_operator == 2) return swish_derivative(value,input);
    if (act_operator == 3) return leakyReLU_derivative(input);
    if (act_operator == 4) return identity_derivative();

    Log::fatal("ERROR: invalid act_operator: %d\n", act_operator); 
    exit(1);
}

void ENAS_DAG_Node::input_fired(int time, double incoming_output) {

    vector<int> connections {0,1,1,1,2,5,3,5,4};
    vector<int> operations {1,1,1,3,3,0,2,1,2};
    vector<int> node_output(connections.size(),1);
    inputs_fired[time]++;
    input_values[time] += incoming_output;

    if (inputs_fired[time] < total_inputs) return;
    else if (inputs_fired[time] > total_inputs) {
        Log::fatal("ERROR: inputs_fired on ENAS_DAG_Node %d at time %d is %d and total_inputs is %d\n", innovation_number, time, inputs_fired[time], total_inputs);
        exit(1);
    }

    //update the reset gate bias so its centered around 1
    //r_bias += 1;
    int no_of_nodes = connections.size();
    Log::debug("ERROR: inputs_fired on ENAS_DAG_Node %d at time %d is %d and no_of_nodes is %d\n", innovation_number, time, inputs_fired[time], no_of_nodes);

    double x = input_values[time];

    double h_prev = 0.0;
    if (time > 0) h_prev = output_values[time - 1];

    double xzw = x*zw;
    double hrw = h_prev*rw;
    double node0_sum = hrw+xzw;

    Nodes[0][time] = activation(node0_sum,operations[0]);
    l_Nodes[0][time] = activation_derivative(node0_sum,Nodes[0][time],operations[0]);
    node_output[0] = 0;
    for(int i = 1;i < connections.size();i++){
        int incoming_node = connections[i] - 1;
        double node_mul = weights[i-1]*Nodes[incoming_node][time];
        Nodes[i][time] = activation(node_mul,operations[i]);
        l_Nodes[i][time] = activation_derivative(node_mul,Nodes[i][time],operations[i]);
        node_output[incoming_node] = 0;
    }

    //int fan_out = 0; 
    for (int i = 0; i < node_output.size(); ++i)
    {
        if(node_output[i]){
           // fan_out ++;
            output_values[time] += Nodes[i][time];  
        } 
    }

    // output_values[time] += Nodes[0][time];

    // output_values[time] /= fan_out;

    Log::debug("DEBUG: input_fired on ENAS_DAG_Node %d at time %d is %d and total_outputs is %d\n", innovation_number, time, outputs_fired[time], total_outputs);


}

void ENAS_DAG_Node::try_update_deltas(int time){
  if (outputs_fired[time] < total_outputs) return;
    else if (outputs_fired[time] > total_outputs) {
        Log::fatal("ERROR: outputs_fired on ENAS_DAG_Node %d at time %d is %d and total_outputs is %d\n", innovation_number, time, outputs_fired[time], total_outputs);
        exit(1);
    }

    double error = error_values[time];
    double x = input_values[time];

    double h_prev = 0.0;
    if (time > 0) h_prev = output_values[time - 1];

    double d_h = error;
    if (time < (series_length - 1)) d_h += d_h_prev[time + 1];

    //d_h *= fan_out;

    vector<int> connections {0,1,1,1,2,5,3,5,4};
    int no_of_nodes = connections.size();
    vector<int> node_output(no_of_nodes,1);
    vector<double> d_node_h(no_of_nodes,0.0);

    node_output[0] = 0;
    for(int i = 1;i < connections.size();i++){
        int incoming_node = connections[i] - 1;
        node_output[incoming_node] = 0;
    }

    for (int i = 0; i < no_of_nodes; ++i)
    {
        if(node_output[i]) d_node_h[i] = d_h;
    }

    for (int i = no_of_nodes - 1; i >=  1; i--)
    {
        int incoming_node = connections[i] - 1;
        d_weights[i-1][time] = d_node_h[i]*l_Nodes[i][time]*Nodes[incoming_node][time];
        d_node_h[incoming_node] += d_node_h[i]*l_Nodes[i][time]*weights[i-1];
    }

    d_h_prev[time] += d_node_h[0]*l_Nodes[0][time]*rw;
    d_rw[time] =  d_node_h[0]*l_Nodes[0][time]*h_prev;

    d_input[time] +=  d_node_h[0]*l_Nodes[0][time]*zw;
    d_zw[time] = d_node_h[0]*l_Nodes[0][time]*x;

    // d_h_prev[time] += d_h*l_Nodes[0][time]*rw;
    // d_rw[time] =  d_h*l_Nodes[0][time]*h_prev;

    // d_input[time] +=  d_h*l_Nodes[0][time]*zw;
    // d_zw[time] = d_h*l_Nodes[0][time]*x;


    Log::debug("DEBUG: output_fired on ENAS_DAG_Node %d at time %d is %d and total_outputs is %d\n", innovation_number, time, outputs_fired[time], total_outputs);


}

void ENAS_DAG_Node::error_fired(int time, double error) {
    outputs_fired[time]++;

    error_values[time] *= error;

    try_update_deltas(time);
}

void ENAS_DAG_Node::output_fired(int time, double delta) {
    outputs_fired[time]++;

    error_values[time] += delta;

    try_update_deltas(time);
}


uint32_t ENAS_DAG_Node::get_number_weights() const {
    return NUMBER_ENAS_DAG_WEIGHTS;
}

void ENAS_DAG_Node::get_weights(vector<double> &parameters) const {
    parameters.resize(get_number_weights());
    uint32_t offset = 0;
    get_weights(offset, parameters);
}

void ENAS_DAG_Node::set_weights(const vector<double> &parameters) {
    uint32_t offset = 0;
    set_weights(offset, parameters);
}

void ENAS_DAG_Node::set_weights(uint32_t &offset, const vector<double> &parameters) {
    //uint32_t start_offset = offset;

    int assigned_node_weights = 2; // 2 weights for the starting node assigned above

    zw = bound(parameters[offset++]);
    rw = bound(parameters[offset++]);

    for (int new_node_weight = 0; new_node_weight < NUMBER_ENAS_DAG_WEIGHTS - assigned_node_weights; ++new_node_weight){
        if(weights.size() < NUMBER_ENAS_DAG_WEIGHTS - assigned_node_weights){
            weights.push_back(bound(parameters[offset++]));            
        }
        else
            weights.at(new_node_weight) = bound(parameters[offset++]);
    }
    Log::debug("DEBUG: no of  weights  on ENAS_DAG_Node %d at time %d is %d \n", innovation_number, time, weights.size());


}

void ENAS_DAG_Node::get_weights(uint32_t &offset, vector<double> &parameters) const {
    //uint32_t start_offset = offset;



    int assigned_node_weights = 2; // 2 weights for the starting node assigned above

    parameters[offset++] = zw;
    parameters[offset++] = rw;

    for (int new_node_weight = 0; new_node_weight < NUMBER_ENAS_DAG_WEIGHTS - assigned_node_weights; ++new_node_weight)
      parameters[offset++] = weights.at(new_node_weight); 

}

void ENAS_DAG_Node::get_gradients(vector<double> &gradients) {
    gradients.assign(NUMBER_ENAS_DAG_WEIGHTS, 0.0);

    for (uint32_t i = 0; i < NUMBER_ENAS_DAG_WEIGHTS; i++) {
        gradients[i] = 0.0;
    }

    for (uint32_t i = 0; i < series_length; i++) {
        gradients[0] += d_zw[i];
        gradients[1] += d_rw[i];

        gradients[2] += d_weights[0][i];

        gradients[3] += d_weights[1][i];
        gradients[4] += d_weights[2][i];
        gradients[5] += d_weights[3][i];

        gradients[6] += d_weights[4][i];
        gradients[7] += d_weights[5][i];
        gradients[8] += d_weights[6][i];
        gradients[9] += d_weights[7][i];

    }
}

void ENAS_DAG_Node::reset(int _series_length) {
    series_length = _series_length;

    d_zw.assign(series_length, 0.0);
    d_rw.assign(series_length, 0.0);

    d_weights.assign(NUMBER_ENAS_DAG_WEIGHTS, vector<double>(series_length,0.0)); 
    d_h_prev.assign(series_length, 0.0);
    Nodes.assign(NUMBER_ENAS_DAG_WEIGHTS, vector<double>(series_length,0.0));
    l_Nodes.assign(NUMBER_ENAS_DAG_WEIGHTS, vector<double>(series_length,0.0));



    //reset values from rnn_node_interface
    d_input.assign(series_length, 0.0);
    error_values.assign(series_length, 0.0);

    input_values.assign(series_length, 0.0);
    output_values.assign(series_length, 0.0);

    inputs_fired.assign(series_length, 0);
    outputs_fired.assign(series_length, 0);
}

RNN_Node_Interface* ENAS_DAG_Node::copy() const {
    ENAS_DAG_Node* n = new ENAS_DAG_Node(innovation_number, layer_type, depth);

    //copy ENAS_DAG_Node values
    n->rw = rw;
    n->zw = zw;

    n->d_zw = d_zw;
    n->d_rw = d_rw;


     for (int i = 0; i < weights.size(); ++i)
    {
        n->weights[i] = weights[i];
        n->d_weights[i] = d_weights[i];
    }


   
    n->d_h_prev = d_h_prev;

    for (int i = 0; i < Nodes.size(); ++i)
    {
        n->Nodes[i] = Nodes[i];
        n->l_Nodes[i] = l_Nodes[i];
    }


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

void ENAS_DAG_Node::write_to_stream(ostream &out) {
    RNN_Node_Interface::write_to_stream(out);
}

