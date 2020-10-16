#include <cmath>

#include <fstream>
using std::ostream;

#include <string>
using std::string;

#include<cmath>
using std::max;

#include "rnn_node_interface.hxx"
#include "rnn_genome.hxx"

#include "common/log.hxx"


extern const int32_t NUMBER_NODE_TYPES = 9;
extern const string NODE_TYPES[] = { "simple" , "jordan", "elman", "UGRNN", "MGU", "GRU", "delta", "LSTM" , "ENARC" , "ENAS_DAG" };

double bound(double value) {
    if (value < -10.0) value = -10.0;
    else if (value > 10.0) value = 10.0;
    return value;
}

double sigmoid(double value) {
    double exp_value = exp(-value);
    return 1.0 / (1.0 + exp_value);
}

double sigmoid_derivative(double input) {
    return input * (1 - input);
}

double identity(double value) {
    return value;
}

double identity_derivative() {
    return 1.0;
}

double tanh_derivative(double input) {
    return 1 - (input * input);
    //return 1 - (tanh(input) * tanh(input));
}

double swish(double value) {
    double exp_value = exp(-value);
    return value*(1.0 / (1.0 + exp_value));
}

double swish_derivative(double value, double input) {
    double sigmoid_value = sigmoid(value);
    return sigmoid_value + (input * (1 - sigmoid_value));
}

double leakyReLU(double value) {
    double alpha = 0.01;
    return fmax(alpha*value, value);
}

double leakyReLU_derivative(double input) {
    double alpha = 0.01;
    if (input > 0) return 1;
    return alpha;
}




RNN_Node_Interface::RNN_Node_Interface(int32_t _innovation_number, int32_t _layer_type, double _depth) : innovation_number(_innovation_number), layer_type(_layer_type), depth(_depth) {
    total_inputs = 0;

    enabled = true;
    forward_reachable = false;
    backward_reachable = false;

    //outputs don't have an official output node but
    //deltas are passed in via the output_fired method
    if (layer_type != HIDDEN_LAYER) {
        Log::fatal("ERROR: Attempted to create a new RNN_Node that was an input or output node without using the constructor which specifies it's parameter name");
        exit(1);
    }
}

RNN_Node_Interface::RNN_Node_Interface(int32_t _innovation_number, int32_t _layer_type, double _depth, string _parameter_name) : innovation_number(_innovation_number), layer_type(_layer_type), depth(_depth), parameter_name(_parameter_name) {
    total_inputs = 0;

    enabled = true;
    forward_reachable = false;
    backward_reachable = false;
    
    if (layer_type == HIDDEN_LAYER) {
        Log::fatal("ERROR: assigned a parameter name '%s' to a hidden node! This should never happen.", parameter_name.c_str());
        exit(1);
    }

    //outputs don't have an official output node but
    //deltas are passed in via the output_fired method
    if (layer_type == OUTPUT_LAYER) {
        total_outputs = 1;
    } else {
        //this is an input node
        total_outputs = 0;
        total_inputs = 1;
    }
}


RNN_Node_Interface::~RNN_Node_Interface() {
}

int32_t RNN_Node_Interface::get_node_type() const {
    return node_type;
}

int32_t RNN_Node_Interface::get_layer_type() const {
    return layer_type;
}

int32_t RNN_Node_Interface::get_innovation_number() const {
    return innovation_number;
}

int32_t RNN_Node_Interface::get_total_inputs() const {
    return total_inputs;
}

int32_t RNN_Node_Interface::get_total_outputs() const {
    return total_outputs;
}



double RNN_Node_Interface::get_depth() const {
    return depth;
}

bool RNN_Node_Interface::is_reachable() const {
    return forward_reachable && backward_reachable;
}

bool RNN_Node_Interface::is_enabled() const {
    return enabled;
}


bool RNN_Node_Interface::equals(RNN_Node_Interface *other) const {
    if (innovation_number == other->innovation_number && enabled == other->enabled) return true;
    return false;
}

void RNN_Node_Interface::write_to_stream(ostream &out) {
    out.write((char*)&innovation_number, sizeof(int32_t));
    out.write((char*)&layer_type, sizeof(int32_t));
    out.write((char*)&node_type, sizeof(int32_t));
    out.write((char*)&depth, sizeof(double));
    out.write((char*)&enabled, sizeof(bool));

    write_binary_string(out, parameter_name, "parameter_name");
}
