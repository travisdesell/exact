#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include "rnn_edge.hxx"

RNN_Edge::RNN_Edge(int _innovation_number, RNN_Node_Interface *_input_node, RNN_Node_Interface *_output_node) {
    innovation_number = _innovation_number;
    input_node = _input_node;
    output_node = _output_node;

    input_innovation_number = input_node->get_innovation_number();
    output_innovation_number = output_node->get_innovation_number();

    input_node->total_outputs++;
    output_node->total_inputs++;

    //cout << "created edge " << innovation_number << " from " << input_innovation_number << ", to " << output_innovation_number << endl;
}

RNN_Edge::RNN_Edge(int _innovation_number, int _input_innovation_number, int _output_innovation_number, const vector<RNN_Node_Interface*> &nodes) {
    innovation_number = _innovation_number;

    input_innovation_number = _input_innovation_number;
    output_innovation_number = _output_innovation_number;

    input_node = NULL;
    output_node = NULL;
    for (int i = 0; i < nodes.size(); i++) {
        if (nodes[i]->innovation_number == _input_innovation_number) {
            if (input_node != NULL) {
                cerr << "ERROR in copying RNN_Edge, list of nodes has multiple nodes with same input_innovation_number -- this should never happen." << endl;
                exit(1);
            }

            input_node = nodes[i];
        }

        if (nodes[i]->innovation_number == _output_innovation_number) {
            if (output_node != NULL) {
                cerr << "ERROR in copying RNN_Edge, list of nodes has multiple nodes with same output_innovation_number -- this should never happen." << endl;
                exit(1);
            }

            output_node = nodes[i];
        }
    }
}



void RNN_Edge::propagate_forward(int time) {
    if (input_node->inputs_fired[time] != input_node->total_inputs) {
        cerr << "ERROR! propagate forward called on edge " << innovation_number << " where input_node->inputs_fired[" << time << "] (" << input_node->inputs_fired[time] << ") != total_inputs (" << input_node->total_inputs << ")" << endl;
        cerr << "input innovation number: " << input_node->innovation_number << ", output innovation number: " << output_node->innovation_number << endl;
        exit(1);
    }

    double output = input_node->output_values[time] * weight;

    //cout << "propagating forward at time " << time << " from " << input_node->innovation_number << " to " << output_node->innovation_number << ", value: " << output << ", input: " << input_node->output_values[time] << ", weight: " << weight << endl;

    outputs[time] = output;
    output_node->input_fired(time, output);
}


void RNN_Edge::propagate_forward(int time, bool training, double dropout_probability) {
    if (input_node->inputs_fired[time] != input_node->total_inputs) {
        cerr << "ERROR! propagate forward called on edge " << innovation_number << " where input_node->inputs_fired[" << time << "] (" << input_node->inputs_fired[time] << ") != total_inputs (" << input_node->total_inputs << ")" << endl;
        exit(1);
    }

    double output = input_node->output_values[time] * weight;

    //cout << "propagating forward at time " << time << " from " << input_node->innovation_number << " to " << output_node->innovation_number << ", value: " << output << ", input: " << input_node->output_values[time] << ", weight: " << weight << endl;

    if (training) {
        if (drand48() < dropout_probability) {
            dropped_out[time] = true;
            output = 0.0;
        } else {
            dropped_out[time] = false;
        }
    } else {
        output *= (1.0 - dropout_probability);
    }

    outputs[time] = output;
    output_node->input_fired(time, output);
}

void RNN_Edge::propagate_backward(int time) {
    if (output_node->outputs_fired[time] != output_node->total_outputs) {
        cerr << "ERROR! propagate backward called on edge " << innovation_number << " where output_node->outputs_fired[" << time << "] (" << output_node->outputs_fired[time] << ") != total_outputs (" << output_node->total_outputs << ")" << endl;
        exit(1);
    }

    double delta = output_node->d_input[time];

    d_weight += delta * input_node->output_values[time];
    deltas[time] = delta * weight;
    input_node->output_fired(time, deltas[time]);
}

void RNN_Edge::propagate_backward(int time, bool training, double dropout_probability) {
    if (output_node->outputs_fired[time] != output_node->total_outputs) {
        cerr << "ERROR! propagate backward called on edge " << innovation_number << " where output_node->outputs_fired[" << time << "] (" << output_node->outputs_fired[time] << ") != total_outputs (" << output_node->total_outputs << ")" << endl;
        exit(1);
    }

    /*
    cout << "edge " << innovation_number << " propagating backwards, input_node->series_length: " << input_node->series_length << endl;
    cout << "input_innovation_number: " << input_innovation_number << endl;
    cout << "output_innovation_number: " << output_innovation_number << endl;
    cout << "input_node->output_values.size(): " << input_node->output_values.size() << endl;
    cout << "output_node->d_input.size(): " << output_node->d_input.size() << endl;
    */

    double delta = output_node->d_input[time];

    if (training) {
        if (dropped_out[time]) delta = 0.0;
    }

    d_weight += delta * input_node->output_values[time];
    deltas[time] = delta * weight;
    input_node->output_fired(time, deltas[time]);
}

void RNN_Edge::reset(int series_length) {
    d_weight = 0.0;
    outputs.resize(series_length);
    deltas.resize(series_length);
    dropped_out.resize(series_length);
}

double RNN_Edge::get_gradient() const {
    return d_weight;
}

int RNN_Edge::get_innovation_number() const {
    return innovation_number;
}

const RNN_Node_Interface* RNN_Edge::get_input_node() const {
    return input_node;
}

const RNN_Node_Interface* RNN_Edge::get_output_node() const {
    return output_node;
}



RNN_Edge* RNN_Edge::copy(const vector<RNN_Node_Interface*> new_nodes) {
    RNN_Edge* e = new RNN_Edge(innovation_number, input_innovation_number, output_innovation_number, new_nodes);

    e->weight = weight;
    e->d_weight = d_weight;

    e->outputs = outputs;
    e->deltas = deltas;

    return e;
}
