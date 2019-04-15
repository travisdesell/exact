#include <cmath>

#include <fstream>
using std::ostream;

#include <string>
using std::string;

#include "rnn_node_interface.hxx"


extern const int32_t NUMBER_NODE_TYPES = 8;
extern const string NODE_TYPES[] = { "simple" , "jordan", "elman", "UGRNN", "MGU", "GRU", "delta", "LSTM" };

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

double tanh_derivative(double input) {
    return 1 - (input * input);
    //return 1 - (tanh(input) * tanh(input));
}

RNN_Node_Interface::RNN_Node_Interface(int32_t _innovation_number, int32_t _layer_type, double _depth) : innovation_number(_innovation_number), layer_type(_layer_type), depth(_depth) {
    total_inputs = 0;

    enabled = true;
    forward_reachable = false;
    backward_reachable = false;

    //outputs don't have an official output node but
    //deltas are passed in via the output_fired method
    if (layer_type == OUTPUT_LAYER) {
        total_outputs = 1;
    } else {
        total_outputs = 0;
    }
}

RNN_Node_Interface::~RNN_Node_Interface() {
}

int32_t RNN_Node_Interface::get_layer_type() const {
    return layer_type;
}

int32_t RNN_Node_Interface::get_innovation_number() const {
    return innovation_number;
}

double RNN_Node_Interface::get_depth() const {
    return depth;
}

bool RNN_Node_Interface::is_reachable() const {
    return forward_reachable && backward_reachable;
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
}
