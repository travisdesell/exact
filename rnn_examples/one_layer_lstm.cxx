#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <vector>
using std::vector;

#include "rnn/lstm_node.hxx"
#include "rnn/rnn_edge.hxx"
#include "rnn/rnn_genome.hxx"
#include "rnn/rnn_node.hxx"
#include "rnn/rnn_node_interface.hxx"


RNN_Genome* create_one_layer_lstm(int number_inputs, int number_outputs) {
    vector<RNN_Node_Interface*> rnn_nodes;
    vector<RNN_Node_Interface*> layer1_nodes;
    vector<RNN_Node_Interface*> layer2_nodes;
    vector<RNN_Edge*> rnn_edges;

    int node_innovation_count = 0;
    int edge_innovation_count = 0;

    for (int32_t i = 0; i < number_inputs; i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, RNN_INPUT_NODE);
        rnn_nodes.push_back(node);
        layer1_nodes.push_back(node);
    }

    for (int32_t i = 0; i < number_inputs; i++) {
        LSTM_Node *node = new LSTM_Node(++node_innovation_count, RNN_HIDDEN_NODE);
        rnn_nodes.push_back(node);
        layer2_nodes.push_back(node);

        for (int32_t j = 0; j < number_inputs; j++) {
            rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer1_nodes[j], layer2_nodes[i]));
        }
    }

    LSTM_Node *output_node = new LSTM_Node(++node_innovation_count, RNN_OUTPUT_NODE);
    rnn_nodes.push_back(output_node);
    for (int32_t i = 0; i < number_inputs; i++) {
        rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer2_nodes[i], output_node));
    }

    return new RNN_Genome(rnn_nodes, rnn_edges);
}

