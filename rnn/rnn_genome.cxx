#include <limits>
using std::numeric_limits;

#include <iostream>
using std::cerr;
using std::endl;

#include <vector>
using std::vector;


#include "rnn_edge.hxx"
#include "rnn_genome.hxx"
#include "rnn_node_interface.hxx"
#include "rnn_node.hxx"
#include "lstm_node.hxx"


RNN_Genome::RNN_Genome(vector<RNN_Node_Interface*> &_nodes, vector<RNN_Edge*> &_edges) {
    nodes = _nodes;
    edges = _edges;

    //sort nodes by depth
    //sort edges by depth

    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->type == RNN_INPUT_NODE) {
            input_nodes.push_back(nodes[i]);
        } else if (nodes[i]->type == RNN_OUTPUT_NODE) {
            output_nodes.push_back(nodes[i]);
        }
    }

}

double RNN_Genome::predict(const vector< vector<double> > &series_data, double expected_class) {
    //reset the values and recurrent values in each node to 0
    for (uint32_t i = 0; i < nodes.size(); i++) {
        nodes[i]->full_reset();
    }

    int count = 0;
    double max_prediction = -numeric_limits<double>::max();
    double output_prediction = 0;
    double sum = 0.0;
    for (uint32_t i = 0; i < series_data.size(); i++) {
        //reset the values in each node to 0
        for (uint32_t i = 0; i < nodes.size(); i++) {
            //reset the non-recurrent values
            nodes[i]->reset();
        }

        //set the input node values
        //TODO: check the number of input nodes == the series data width
        for (uint32_t j = 0; j < input_nodes.size(); j++) {
            input_nodes[j]->input_value = series_data[i][j];
            input_nodes[j]->input_fired();
        }

        //feed forward
        for (uint32_t j = 0; j < edges.size(); j++) {
            edges[j]->propagate_forward();
        }

        //calculate max prediction
        output_prediction = output_nodes[0]->output_value;
        if (output_prediction > max_prediction) max_prediction = output_prediction;
        sum += output_prediction;
        count++;
    }

    //check to see error between output and expected class
    //return max_prediction;
    //return output_prediction;
    return sum / count;
}

void RNN_Genome::set_weights(const vector<double> &parameters) {
    if (parameters.size() != get_number_weights()) {
        cerr << "ERROR! Trying to set weights where the RNN has " << get_number_weights() << " weights, and the parameters vector has << " << parameters.size() << " weights!" << endl;
        exit(1);
    }

    uint32_t current = 0;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        nodes[i]->set_weights(current, parameters);
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        //cerr << "setting edge " << edges[i]->innovation_number << " weight to parameters[" << current << "]: " << parameters[current] << endl;

        edges[i]->weight = parameters[current++];
    }

}

uint32_t RNN_Genome::get_number_weights() {
    uint32_t number_weights = 0;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        number_weights += nodes[i]->get_number_weights();
    }

    number_weights += edges.size();

    return number_weights;
}
