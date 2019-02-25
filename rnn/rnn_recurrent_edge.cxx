#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include "rnn_recurrent_edge.hxx"

RNN_Recurrent_Edge::RNN_Recurrent_Edge(int32_t _innovation_number, int32_t _recurrent_depth, RNN_Node_Interface *_input_node, RNN_Node_Interface *_output_node) {
    innovation_number = _innovation_number;
    recurrent_depth = _recurrent_depth;

    if (recurrent_depth <= 0) {
        cerr << "ERROR, trying to create a recurrent edge with recurrent depth <= 0" << endl;
        cerr << "innovation number: " << innovation_number << endl;
        cerr << "input_node->innovation_number: " << input_node->get_innovation_number() << endl;
        cerr << "output_node->innovation_number: " << output_node->get_innovation_number() << endl;
        exit(1);
    }

    input_node = _input_node;
    output_node = _output_node;

    input_innovation_number = input_node->get_innovation_number();
    output_innovation_number = output_node->get_innovation_number();

    input_node->total_outputs++;
    output_node->total_inputs++;

    enabled = true;
    forward_reachable = true;
    backward_reachable = true;

    //cout << "created recurrent edge " << innovation_number << ", from " << input_innovation_number << ", to " << output_innovation_number << endl;
}

RNN_Recurrent_Edge::RNN_Recurrent_Edge(int32_t _innovation_number, int32_t _recurrent_depth, int32_t _input_innovation_number, int32_t _output_innovation_number, const vector<RNN_Node_Interface*> &nodes) {
    innovation_number = _innovation_number;
    recurrent_depth = _recurrent_depth;

    input_innovation_number = _input_innovation_number;
    output_innovation_number = _output_innovation_number;

    if (recurrent_depth <= 0) {
        cerr << "ERROR, trying to create a recurrent edge with recurrent depth <= 0" << endl;
        cerr << "innovation number: " << innovation_number << endl;
        cerr << "input_innovation_number: " << input_innovation_number << endl;
        cerr << "output_innovation_number: " << output_innovation_number << endl;
        exit(1);
    }


    input_node = NULL;
    output_node = NULL;
    for (int32_t i = 0; i < nodes.size(); i++) {
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

    if (input_node == NULL) {
        cerr << "ERROR initializing RNN_Edge, input node with innovation number; " << input_innovation_number << " was not found!" << endl;
        exit(1);
    }

    if (output_node == NULL) {
        cerr << "ERROR initializing RNN_Edge, output node with innovation number; " << output_innovation_number << " was not found!" << endl;
        exit(1);
    }
}

RNN_Recurrent_Edge* RNN_Recurrent_Edge::copy(const vector<RNN_Node_Interface*> new_nodes) {
    RNN_Recurrent_Edge* e = new RNN_Recurrent_Edge(innovation_number, recurrent_depth, input_innovation_number, output_innovation_number, new_nodes);

    e->recurrent_depth = recurrent_depth;

    e->weight = weight;
    e->d_weight = d_weight;

    e->outputs = outputs;
    e->deltas = deltas;

    e->enabled = enabled;
    e->forward_reachable = forward_reachable;
    e->backward_reachable = backward_reachable;

    return e;
}


int32_t RNN_Recurrent_Edge::get_innovation_number() const {
    return innovation_number;
}

int32_t RNN_Recurrent_Edge::get_input_innovation_number() const {
    return input_innovation_number;
}

int32_t RNN_Recurrent_Edge::get_output_innovation_number() const {
    return output_innovation_number;
}


const RNN_Node_Interface* RNN_Recurrent_Edge::get_input_node() const {
    return input_node;
}

const RNN_Node_Interface* RNN_Recurrent_Edge::get_output_node() const {
    return output_node;
}



//do a propagate to the network at time 0 so that the
//input fireds are correct
void RNN_Recurrent_Edge::first_propagate_forward() {
    for (uint32_t i = 0; i < recurrent_depth; i++) {
        cout << "FIRST propagating forward on RecEdge " << innovation_number << " from node " << input_innovation_number << " to node " << output_innovation_number << " with recurrent depth: " << recurrent_depth << endl;
        output_node->input_fired(i, 0.0);
    }
}

void RNN_Recurrent_Edge::propagate_forward(int32_t time) {
    if (input_node->inputs_fired[time] != input_node->total_inputs) {
        cerr << "ERROR! propagate forward called on recurrent edge " << innovation_number << " where input_node->inputs_fired[" << time << "] (" << input_node->inputs_fired[time] << ") != total_inputs (" << input_node->total_inputs << ")" << endl;
        exit(1);
    }


    double output = input_node->output_values[time] * weight;
    if (time < series_length - recurrent_depth) {
        cout << "propagating forward on RecEdge " << innovation_number << " from time " << time << " to time " << time + recurrent_depth << " from node " << input_innovation_number << " to node " << output_innovation_number << ", value: " << output << ", input: " << input_node->output_values[time] << ", weight: " << weight << endl;

        //cout << "propagating recurrent at time " << time << " from " << input_node->innovation_number << " to " << output_node->innovation_number << ", value: " << output << ", input: " << input_node->output_values[time] << ", weight: " << weight << endl;
        outputs[time + recurrent_depth] = output;
        output_node->input_fired(time + recurrent_depth, output);
    }
}

//do a propagate to the network at time (series_length - 1) so that the
//output fireds are correct
void RNN_Recurrent_Edge::first_propagate_backward() {
    for (uint32_t i = 0; i < recurrent_depth; i++) {
        //cout << "FIRST propagating backward on recurrent edge " << innovation_number << " to time " << series_length - 1 - i << " from node " << output_innovation_number << " to node " << input_innovation_number << endl;
        input_node->output_fired(series_length - 1 - i, 0.0);
    }
}

void RNN_Recurrent_Edge::propagate_backward(int32_t time) {
    if (output_node->outputs_fired[time] != output_node->total_outputs) {
        //if (output_node->innovation_number == input_node->innovation_number) {
            //circular recurrent edge
            /*
            if (output_node->outputs_fired[time] != (output_node->total_outputs - 1)) {
                cerr << "ERROR! propagate backward called on recurrent edge " << innovation_number << " where output_node->outputs_fired[" << time << "] (" << output_node->outputs_fired[time] << ") != total_outputs (" << output_node->total_outputs << ")" << endl;
                cerr << "input innovation number: " << input_node->innovation_number << ", output innovation number: " << output_node->innovation_number << endl;
                exit(1);
            }
            */
        //} else {
            cerr << "ERROR! propagate backward called on recurrent edge " << innovation_number << " where output_node->outputs_fired[" << time << "] (" << output_node->outputs_fired[time] << ") != total_outputs (" << output_node->total_outputs << ")" << endl;
            cerr << "input innovation number: " << input_node->innovation_number << ", output innovation number: " << output_node->innovation_number << endl;
            exit(1);
        //}
    }

    /*
    cout << "edge " << innovation_number << " propagating backwards, input_node->series_length: " << input_node->series_length << endl;
    cout << "input_innovation_number: " << input_innovation_number << endl;
    cout << "output_innovation_number: " << output_innovation_number << endl;
    cout << "input_node->output_values.size(): " << input_node->output_values.size() << endl;
    cout << "output_node->d_input.size(): " << output_node->d_input.size() << endl;
    */

    double delta = output_node->d_input[time];

    if (time - recurrent_depth >= 0) {
        //cout << "propagating backward on recurrent edge " << innovation_number << " from time " << time << " to time " << time - recurrent_depth << " from node " << output_innovation_number << " to node " << input_innovation_number << endl;

        d_weight += delta * input_node->output_values[time - recurrent_depth];
        deltas[time] = delta * weight;
        input_node->output_fired(time - recurrent_depth, deltas[time]);
    }
}

void RNN_Recurrent_Edge::reset(int32_t _series_length) {
    series_length = _series_length;
    d_weight = 0.0;
    outputs.resize(series_length);
    deltas.resize(series_length);
}

double RNN_Recurrent_Edge::get_gradient() {
    return d_weight;
}

bool RNN_Recurrent_Edge::is_reachable() const {
    return forward_reachable && backward_reachable;
}

bool RNN_Recurrent_Edge::equals(RNN_Recurrent_Edge *other) const {
    if (innovation_number == other->innovation_number && enabled == other->enabled) return true;
    return false;
}

void RNN_Recurrent_Edge::write_to_stream(ostream &out) {
    out.write((char*)&innovation_number, sizeof(int32_t));
    out.write((char*)&recurrent_depth, sizeof(int32_t));
    out.write((char*)&input_innovation_number, sizeof(int32_t));
    out.write((char*)&output_innovation_number, sizeof(int32_t));
    out.write((char*)&enabled, sizeof(bool));
}
