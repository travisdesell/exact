#ifndef RNN_GENERATE_NN_HXX
#define RNN_GENERATE_NN_HXX

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "rnn/delta_node.hxx"
#include "rnn/ugrnn_node.hxx"
#include "rnn/gru_node.hxx"
#include "rnn/mgu_node.hxx"
#include "rnn/lstm_node.hxx"
#include "rnn/enarc_node.hxx"
#include "rnn/enas_dag_node.hxx"
#include "rnn/random_dag_node.hxx"
#include "rnn/rnn_edge.hxx"
#include "rnn/rnn_node.hxx"
#include "rnn/rnn_node_interface.hxx"
#include "rnn/rnn_genome.hxx"

#include "common/log.hxx"
#include "common/weight_initialize.hxx"

template <class NodeT>
NodeT *create_hidden_node(int32_t &innovation_counter, double depth) {
    return new NodeT(++innovation_counter, HIDDEN_LAYER, depth);
}
RNN_Node_Interface *create_hidden_memory_cell(int32_t node_kind, int32_t &innovation_counter, double depth);
RNN_Node *create_hidden_node(int32_t node_kind, int32_t &innovation_counter, double depth);

template <unsigned int Kind>
RNN_Genome* create_simple_nn(const vector<string> &input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes, const vector<string> &output_parameter_names, int32_t max_recurrent_depth, WeightType weight_initialize, WeightType weight_inheritance = WeightType::RANDOM, WeightType mutated_component_weight = WeightType::RANDOM) {
    Log::debug("creating feed forward network with inputs: %d, hidden: %dx%d, outputs: %d, max recurrent depth: %d\n", input_parameter_names.size(), number_hidden_layers, number_hidden_nodes, output_parameter_names.size(), max_recurrent_depth);
    vector<RNN_Node_Interface*> rnn_nodes;
    vector< vector<RNN_Node_Interface*> > layer_nodes(2 + number_hidden_layers);
    vector<RNN_Edge*> rnn_edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    int32_t node_innovation_count = 0;
    int32_t edge_innovation_count = 0;
    int32_t current_layer = 0;

    for (int32_t i = 0; i < (int32_t)input_parameter_names.size(); i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, INPUT_LAYER, current_layer, SIMPLE_NODE, input_parameter_names[i]);
        rnn_nodes.push_back(node);
        layer_nodes[current_layer].push_back(node);
    }
    current_layer++;

    for (int32_t i = 0; i < (int32_t)number_hidden_layers; i++) {
        for (int32_t j = 0; j < (int32_t)number_hidden_nodes; j++) {
            RNN_Node *node = new RNN_Node(++node_innovation_count, HIDDEN_LAYER, current_layer, Kind);
            rnn_nodes.push_back(node);
            layer_nodes[current_layer].push_back(node);

            for (int32_t k = 0; k < (int32_t)layer_nodes[current_layer - 1].size(); k++) {
                rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], node));

                for (int32_t d = 1; d <= max_recurrent_depth; d++) {
                    recurrent_edges.push_back(new RNN_Recurrent_Edge(++edge_innovation_count, d, layer_nodes[current_layer - 1][k], node));
                }
            }

        }
        current_layer++;
    }

    for (int32_t i = 0; i < (int32_t)output_parameter_names.size(); i++) {
        RNN_Node *output_node = new RNN_Node(++node_innovation_count, OUTPUT_LAYER, current_layer, SIMPLE_NODE, output_parameter_names[i]);
        rnn_nodes.push_back(output_node);

        for (int32_t k = 0; k < (int32_t)layer_nodes[current_layer - 1].size(); k++) {
            rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], output_node));

            for (int32_t d = 1; d <= max_recurrent_depth; d++) {
                recurrent_edges.push_back(new RNN_Recurrent_Edge(++edge_innovation_count, d, layer_nodes[current_layer - 1][k], output_node));
            }
        }
    }

    RNN_Genome *genome = new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges, weight_initialize, weight_inheritance, mutated_component_weight);
    genome->set_parameter_names(input_parameter_names, output_parameter_names);
    return genome;
}

#define create_ff(...) create_simple_nn<SIMPLE_NODE>(__VA_ARGS__)
#define create_elman(...) create_simple_nn<ELMAN_NODE>(__VA_ARGS__)
#define create_jordan(...) create_simple_nn<JORDAN_NODE>(__VA_ARGS__)

template <typename NodeT>
RNN_Genome *create_nn(const vector<string> &input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes, const vector<string> &output_parameter_names, int32_t max_recurrent_depth, WeightType weight_initialize) {
     Log::debug("creating network with inputs: %d, hidden: %dx%d, outputs: %d, max recurrent depth: %d\n", input_parameter_names.size(), number_hidden_layers, number_hidden_nodes, output_parameter_names.size(), max_recurrent_depth);
    vector<RNN_Node_Interface*> rnn_nodes;
    vector< vector<RNN_Node_Interface*> > layer_nodes(2 + number_hidden_layers);
    vector<RNN_Edge*> rnn_edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    int32_t node_innovation_count = 0;
    int32_t edge_innovation_count = 0;
    int32_t current_layer = 0;

    for (int32_t i = 0; i < (int32_t)input_parameter_names.size(); i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, INPUT_LAYER, current_layer, SIMPLE_NODE, input_parameter_names[i]);
        rnn_nodes.push_back(node);
        layer_nodes[current_layer].push_back(node);
    }
    current_layer++;

    for (int32_t i = 0; i < number_hidden_layers; i++) {
        for (int32_t j = 0; j < number_hidden_nodes; j++) {
            NodeT *node = create_hidden_node<NodeT>(node_innovation_count, current_layer);
            rnn_nodes.push_back(node);
            layer_nodes[current_layer].push_back(node);

            for (int32_t k = 0; k < (int32_t)layer_nodes[current_layer - 1].size(); k++) {
                rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], node));
            }
        }
        current_layer++;
    }

    for (int32_t i = 0; i < (int32_t)output_parameter_names.size(); i++) {
        RNN_Node *output_node = new RNN_Node(++node_innovation_count, OUTPUT_LAYER, current_layer, SIMPLE_NODE, output_parameter_names[i]);
        rnn_nodes.push_back(output_node);

        for (int32_t k = 0; k < (int32_t)layer_nodes[current_layer - 1].size(); k++) {
            rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], output_node));
        }
    }

    RNN_Genome *genome = new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges, weight_initialize, WeightType::NONE, WeightType::NONE);
    genome->set_parameter_names(input_parameter_names, output_parameter_names);
    return genome;
}

#define create_mgu(...) create_nn<MGU_Node>(__VA_ARGS__)
#define create_gru(...) create_nn<GRU_Node>(__VA_ARGS__)
#define create_delta(...) create_nn<Delta_Node>(__VA_ARGS__)
#define create_lstm(...) create_nn<LSTM_Node>(__VA_ARGS__)
#define create_enarc(...) create_nn<ENARC_Node>(__VA_ARGS__)
#define create_enas_dag(...) create_nn<ENAS_DAG_Node>(__VA_ARGS__)
#define create_random_dag(...) create_nn<RANDOM_DAG_Node>(__VA_ARGS__)
#define create_ugrnn(...) create_nn<UGRNN_Node>(__VA_ARGS__)

RNN_Genome* create_dnas(const vector<string> &input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes, const vector<string> &output_parameter_names, int32_t max_recurrent_depth, WeightType weight_initialize, vector<RNN_Node_Interface *> &nodes);

#endif
