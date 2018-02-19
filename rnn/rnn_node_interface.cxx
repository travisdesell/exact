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

RNN_Node_Interface::RNN_Node_Interface(int _innovation_number, int _type) : innovation_number(_innovation_number), type(_type) {
    inputs_fired = 0;
    outputs_fired = 0;
    total_inputs = 0;
    total_outputs = 0;
}

int RNN_Node_Interface::get_type() {
    return type;
}

int RNN_Node_Interface::get_innovation_number() {
    return innovation_number;
}

