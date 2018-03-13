#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include "rnn_edge.hxx"

RNN_Recurrent_Edge::RNN_Recurrent_Edge(int _innovation_number, RNN_Node_Interface *_input_node, RNN_Node_Interface *_output_node) {
    innovation_number = _innovation_number;
    input_node = _input_node;
    output_node = _output_node;

    input_innovation_number = input_node->get_innovation_number();
    output_innovation_number = output_node->get_innovation_number();

    input_node->total_outputs++;
    output_node->total_inputs++;

    //cout << "created recurrent edge " << innovation_number << ", from " << input_innovation_number << ", to " << output_innovation_number << endl;
}

RNN_Recurrent_Edge::RNN_Recurrent_Edge(int _innovation_number, int _input_innovation_number, int _output_innovation_number, const vector<RNN_Node_Interface*> &nodes) {
    innovation_number = _innovation_number;

    input_innovation_number = _input_innovation_number;
    output_innovation_number = _output_innovation_number;

    input_node = NULL;
    output_node = NULL;
    for (int i = 0; i < nodes.size(); i++) {
        if (nodes[i]->innovation_number == _input_innovation_number) {
            if (input_node != NULL) {
                cerr << "ERROR in copying RNN_Recurrent_Edge, list of nodes has multiple nodes with same input_innovation_number -- this should never happen." << endl;
                exit(1);
            }

            input_node = nodes[i];
        }

        if (nodes[i]->innovation_number == _output_innovation_number) {
            if (output_node != NULL) {
                cerr << "ERROR in copying RNN_Recurrent_Edge, list of nodes has multiple nodes with same output_innovation_number -- this should never happen." << endl;
                exit(1);
            }

            output_node = nodes[i];
        }
    }
}


//do a propagate to the network at time 0 so that the
//input fireds are correct
void RNN_Recurrent_Edge::first_propagate_forward() {
    output_node->input_fired(0, 0.0);
}

void RNN_Recurrent_Edge::propagate_forward(int time) {
    double output = input_node->output_values[time] * weight;
    if (time < series_length - 1) {
        //cout << "propagating recurrent at time " << time << " from " << input_node->innovation_number << " to " << output_node->innovation_number << ", value: " << output << ", input: " << input_node->output_values[time] << ", weight: " << weight << endl;
        outputs[time + 1] = output;
        output_node->input_fired(time + 1, output);
    }
}

//do a propagate to the network at time (series_length - 1) so that the
//output fireds are correct
void RNN_Recurrent_Edge::first_propagate_backward() {
    input_node->output_fired(series_length - 1, 0.0);
}

void RNN_Recurrent_Edge::propagate_backward(int time) {
    /*
    cout << "edge " << innovation_number << " propagating backwards, input_node->series_length: " << input_node->series_length << endl;
    cout << "input_innovation_number: " << input_innovation_number << endl;
    cout << "output_innovation_number: " << output_innovation_number << endl;
    cout << "input_node->output_values.size(): " << input_node->output_values.size() << endl;
    cout << "output_node->d_input.size(): " << output_node->d_input.size() << endl;
    */

    double delta = output_node->d_input[time];

    if (time > 0) {
        d_weight += delta * input_node->output_values[time - 1];
        deltas[time] = delta * weight;
        input_node->output_fired(time - 1, deltas[time]);
    }
}

void RNN_Recurrent_Edge::reset(int _series_length) {
    series_length = _series_length;
    d_weight = 0.0;
    outputs.resize(series_length);
    deltas.resize(series_length);
}

double RNN_Recurrent_Edge::get_gradient() {
    return d_weight;
}

RNN_Recurrent_Edge* RNN_Recurrent_Edge::copy(const vector<RNN_Node_Interface*> new_nodes) {
    RNN_Recurrent_Edge* e = new RNN_Recurrent_Edge(innovation_number, input_innovation_number, output_innovation_number, new_nodes);

    e->weight = weight;
    e->d_weight = d_weight;

    e->outputs = outputs;
    e->deltas = deltas;

    return e;
}
