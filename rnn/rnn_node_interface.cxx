#include <cmath>

#include "rnn_node_interface.hxx"

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

void bound_value(double &value) {
    bound_value(-20.0, 20.0, value);
}

void bound_value(double min, double max, double &value) {
    if (value < min) value = min;
    else if (value > max) value = max;
}

RNN_Node_Interface::RNN_Node_Interface(int _innovation_number, int _type, double _depth) : innovation_number(_innovation_number), type(_type), depth(_depth) {
    total_inputs = 0;

    //outputs don't have an official output node but
    //deltas are passed in via the output_fired method
    if (type == RNN_OUTPUT_NODE) {
        total_outputs = 1;
    } else {
        total_outputs = 0;
    }
}

int RNN_Node_Interface::get_type() const {
    return type;
}

int RNN_Node_Interface::get_innovation_number() const {
    return innovation_number;
}

double RNN_Node_Interface::get_depth() const {
    return depth;
}
