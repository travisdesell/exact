#include "rnn_edge.hxx"

#include "common/log.hxx"

RNN_Edge::RNN_Edge(int _innovation_number, RNN_Node_Interface *_input_node, RNN_Node_Interface *_output_node) {
    innovation_number = _innovation_number;
    input_node = _input_node;
    output_node = _output_node;

    enabled = true;
    forward_reachable = false;
    backward_reachable = false;

    input_innovation_number = input_node->get_innovation_number();
    output_innovation_number = output_node->get_innovation_number();

    input_node->total_outputs++;
    output_node->total_inputs++;

    Log::debug("\t\tcreated edge %d from %d to %d\n", innovation_number, input_innovation_number, output_innovation_number);
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
                Log::fatal("ERROR in copying RNN_Edge, list of nodes has multiple nodes with same input_innovation_number -- this should never happen.\n");
                exit(1);
            }

            input_node = nodes[i];
        }

        if (nodes[i]->innovation_number == _output_innovation_number) {
            if (output_node != NULL) {
                Log::fatal("ERROR in copying RNN_Edge, list of nodes has multiple nodes with same output_innovation_number -- this should never happen.\n");
                exit(1);
            }

            output_node = nodes[i];
        }
    }

    if (input_node == NULL) {
        Log::fatal("ERROR initializing RNN_Edge, input node with innovation number; %d was not found!\n", input_innovation_number);
        exit(1);
    }

    if (output_node == NULL) {
        Log::fatal("ERROR initializing RNN_Edge, output node with innovation number; %d was not found!\n", output_innovation_number);
        exit(1);
    }
}

RNN_Edge* RNN_Edge::copy(const vector<RNN_Node_Interface*> new_nodes) {
    RNN_Edge* e = new RNN_Edge(innovation_number, input_innovation_number, output_innovation_number, new_nodes);

    e->weight = weight;
    e->d_weight = d_weight;

    e->outputs = outputs;
    e->deltas = deltas;
    e->enabled = enabled;
    e->forward_reachable = forward_reachable;
    e->backward_reachable = backward_reachable;

    return e;
}




void RNN_Edge::propagate_forward(int time) {
    if (input_node->inputs_fired[time] != input_node->total_inputs) {
        Log::fatal("ERROR! propagate forward called on edge %d where input_node->inputs_fired[%d] (%d) != total_inputs (%d)\n", innovation_number, time, input_node->inputs_fired[time], input_node->total_inputs);
        Log::fatal("input innovation number: %d, output innovation number: %d\n", input_node->innovation_number, output_node->innovation_number);
        exit(1);
    }

    double output = input_node->output_values[time] * weight;

    //Log::trace("propagating forward at time %d from %d to %d, value: %lf, input: %lf, weight: %lf\n", time, input_node->innovation_number, output_node->innovation_number, output, input_node->output_values[time], weight);

    outputs[time] = output;
    output_node->input_fired(time, output);
}


void RNN_Edge::propagate_forward(int time, bool training, double dropout_probability) {
    if (input_node->inputs_fired[time] != input_node->total_inputs) {
        Log::fatal("ERROR! propagate forward called on edge %d where input_node->inputs_fired[%d] (%d) != total_inputs (%d)\n", innovation_number, time, input_node->inputs_fired[time], input_node->total_inputs);
        exit(1);
    }

    double output = input_node->output_values[time] * weight;

    //Log::trace("propagating forward at time %d from %d to %d, value: %lf, input: %lf, weight: %lf\n", time, input_node->innovation_number, output_node->innovation_number, output, input_node->output_values[time], weight);

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
        Log::fatal("ERROR! propagate backward called on edge %d where output_node->outputs_fired[%d] (%d) != total_outputs (%d)\n", innovation_number, time, output_node->outputs_fired[time], output_node->total_outputs);
        Log::fatal("input innovation number: %d, output innovation number: %d\n", input_node->innovation_number, output_node->innovation_number);
        Log::fatal("series_length: %d\n", input_node->series_length);
        exit(1);
    }

    //Log::trace("propgating backward on edge %d at time %d from node %d to node %d\n", innovation_number, time, output_innovation_number, input_innovation_number);

    double delta = output_node->d_input[time];

    d_weight += delta * input_node->output_values[time];
    deltas[time] = delta * weight;
    input_node->output_fired(time, deltas[time]);
}

void RNN_Edge::propagate_backward(int time, bool training, double dropout_probability) {
    if (output_node->outputs_fired[time] != output_node->total_outputs) {
        Log::fatal("ERROR! propagate backward called on edge %d where output_node->outputs_fired[%d] (%d) != total_outputs (%d)\n", innovation_number, time, output_node->outputs_fired[time], output_node->total_outputs);
        Log::fatal("input innovation number: %d, output innovation number: %d\n", input_node->innovation_number, output_node->innovation_number);
        Log::fatal("series_length: %d\n", input_node->series_length);
        exit(1);
    }

    //Log::trace("propgating backward on edge %d at time %d from node %d to node %d\n", innovation_number, time, output_innovation_number, input_innovation_number);

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

int32_t RNN_Edge::get_innovation_number() const {
    return innovation_number;
}

int32_t RNN_Edge::get_input_innovation_number() const {
    return input_innovation_number;
}


int32_t RNN_Edge::get_output_innovation_number() const {
    return output_innovation_number;
}


const RNN_Node_Interface* RNN_Edge::get_input_node() const {
    return input_node;
}

const RNN_Node_Interface* RNN_Edge::get_output_node() const {
    return output_node;
}

bool RNN_Edge::is_enabled() const {
    return enabled;
}

bool RNN_Edge::is_reachable() const {
    return forward_reachable && backward_reachable;
}


bool RNN_Edge::equals(RNN_Edge *other) const {
    if (innovation_number == other->innovation_number && enabled == other->enabled) return true;
    return false;
}

void RNN_Edge::write_to_stream(ostream &out) {
    out.write((char*)&innovation_number, sizeof(int32_t));
    out.write((char*)&input_innovation_number, sizeof(int32_t));
    out.write((char*)&output_innovation_number, sizeof(int32_t));
    out.write((char*)&enabled, sizeof(bool));
}
