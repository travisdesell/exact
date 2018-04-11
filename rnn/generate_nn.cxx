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

RNN_Genome* create_jordan(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs) {
    vector<RNN_Node_Interface*> rnn_nodes;
    vector<RNN_Node_Interface*> output_layer;
    vector< vector<RNN_Node_Interface*> > layer_nodes(2 + number_hidden_layers);
    vector<RNN_Edge*> rnn_edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    int node_innovation_count = 0;
    int edge_innovation_count = 0;
    int current_layer = 0;

    for (int32_t i = 0; i < number_inputs; i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, RNN_INPUT_NODE, current_layer);
        rnn_nodes.push_back(node);
        layer_nodes[current_layer].push_back(node);
    }
    current_layer++;

    for (int32_t i = 0; i < number_hidden_layers; i++) {
        for (uint32_t j = 0; j < number_hidden_nodes; j++) {
            RNN_Node *node = new RNN_Node(++node_innovation_count, RNN_HIDDEN_NODE, current_layer);
            rnn_nodes.push_back(node);
            layer_nodes[current_layer].push_back(node);

            for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
                rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], node));
            }
        }
        current_layer++;
    }

    for (int32_t i = 0; i < number_outputs; i++) {
        RNN_Node *output_node = new RNN_Node(++node_innovation_count, RNN_OUTPUT_NODE, current_layer);
        output_layer.push_back(output_node);

        rnn_nodes.push_back(output_node);

        for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
            rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], output_node));
        }
    }

    //connect the output node with recurrent edges to each hidden node
    for (int32_t k = 0; k < output_layer.size(); k++) {
        for (int32_t i = 0; i < number_hidden_layers; i++) {
            for (int32_t j = 0; j < number_hidden_nodes; j++) {
                recurrent_edges.push_back(new RNN_Recurrent_Edge(++edge_innovation_count, output_layer[k], layer_nodes[1 + i][j]));
            }
        }
    }


    return new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges);
}

RNN_Genome* create_elman(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs) {
    vector<RNN_Node_Interface*> rnn_nodes;
    vector<RNN_Node_Interface*> output_layer;
    vector< vector<RNN_Node_Interface*> > layer_nodes(2 + number_hidden_layers);
    vector<RNN_Edge*> rnn_edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    int node_innovation_count = 0;
    int edge_innovation_count = 0;
    int current_layer = 0;

    for (int32_t i = 0; i < number_inputs; i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, RNN_INPUT_NODE, current_layer);
        rnn_nodes.push_back(node);
        layer_nodes[current_layer].push_back(node);
    }
    current_layer++;

    for (int32_t i = 0; i < number_hidden_layers; i++) {
        for (uint32_t j = 0; j < number_hidden_nodes; j++) {
            RNN_Node *node = new RNN_Node(++node_innovation_count, RNN_HIDDEN_NODE, current_layer);
            rnn_nodes.push_back(node);
            layer_nodes[current_layer].push_back(node);

            for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
                rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], node));
            }
        }
        current_layer++;
    }

    for (int32_t i = 0; i < number_outputs; i++) {
        RNN_Node *output_node = new RNN_Node(++node_innovation_count, RNN_OUTPUT_NODE, current_layer);
        output_layer.push_back(output_node);

        rnn_nodes.push_back(output_node);

        for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
            rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], output_node));
        }
    }

    //connect the hidden nodes back to their hidden layer
    for (int32_t i = 1; i < layer_nodes.size(); i++) {
        for (int32_t j = 0; j < layer_nodes[i].size(); j++) {
            for (int32_t k = 0; k < layer_nodes[k].size(); k++) {
                recurrent_edges.push_back(new RNN_Recurrent_Edge(++edge_innovation_count, layer_nodes[i][j], layer_nodes[i][k]));
            }
        }
    }

    return new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges);
}

RNN_Genome* create_ff(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs) {
    cout << "creating ff with inputs: " << number_inputs << ", hidden: " << number_hidden_layers << "x" << number_hidden_nodes << ", outputs: " << number_outputs << endl;
    vector<RNN_Node_Interface*> rnn_nodes;
    vector< vector<RNN_Node_Interface*> > layer_nodes(2 + number_hidden_layers);
    vector<RNN_Edge*> rnn_edges;

    int node_innovation_count = 0;
    int edge_innovation_count = 0;
    int current_layer = 0;

    for (int32_t i = 0; i < number_inputs; i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, RNN_INPUT_NODE, current_layer);
        rnn_nodes.push_back(node);
        layer_nodes[current_layer].push_back(node);
    }
    current_layer++;

    for (int32_t i = 0; i < number_hidden_layers; i++) {
        for (uint32_t j = 0; j < number_hidden_nodes; j++) {
            RNN_Node *node = new RNN_Node(++node_innovation_count, RNN_HIDDEN_NODE, current_layer);
            rnn_nodes.push_back(node);
            layer_nodes[current_layer].push_back(node);

            for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
                rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], node));
            }
        }
        current_layer++;
    }

    for (int32_t i = 0; i < number_outputs; i++) {
        RNN_Node *output_node = new RNN_Node(++node_innovation_count, RNN_OUTPUT_NODE, current_layer);
        rnn_nodes.push_back(output_node);

        for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
            rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], output_node));
        }
    }

    return new RNN_Genome(rnn_nodes, rnn_edges);
}

RNN_Genome* create_lstm(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs) {
    vector<RNN_Node_Interface*> rnn_nodes;
    vector< vector<RNN_Node_Interface*> > layer_nodes(2 + number_hidden_layers);
    vector<RNN_Edge*> rnn_edges;

    int node_innovation_count = 0;
    int edge_innovation_count = 0;
    int current_layer = 0;

    for (int32_t i = 0; i < number_inputs; i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, RNN_INPUT_NODE, current_layer);
        rnn_nodes.push_back(node);
        layer_nodes[current_layer].push_back(node);
    }
    current_layer++;

    for (int32_t i = 0; i < number_hidden_layers; i++) {
        for (uint32_t j = 0; j < number_hidden_nodes; j++) {
            LSTM_Node *node = new LSTM_Node(++node_innovation_count, RNN_HIDDEN_NODE, current_layer);
            rnn_nodes.push_back(node);
            layer_nodes[current_layer].push_back(node);

            for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
                rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], node));
            }
        }
        current_layer++;
    }

    for (int32_t i = 0; i < number_outputs; i++) {
        LSTM_Node *output_node = new LSTM_Node(++node_innovation_count, RNN_OUTPUT_NODE, current_layer);
        rnn_nodes.push_back(output_node);

        for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
            rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], output_node));
        }
    }

    return new RNN_Genome(rnn_nodes, rnn_edges);
}

