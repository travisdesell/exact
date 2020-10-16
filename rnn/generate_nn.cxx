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
#include "rnn/rnn_genome.hxx"
#include "rnn/rnn_node.hxx"
#include "rnn/rnn_node_interface.hxx"

#include "common/log.hxx"

RNN_Genome* create_ff(const vector<string> &input_parameter_names, int number_hidden_layers, int number_hidden_nodes, const vector<string> &output_parameter_names, int max_recurrent_depth, WeightType weight_initialize, WeightType weight_inheritance, WeightType mutated_component_weight) {
    Log::debug("creating feed forward network with inputs: %d, hidden: %dx%d, outputs: %d, max recurrent depth: %d\n", input_parameter_names.size(), number_hidden_layers, number_hidden_nodes, output_parameter_names.size(), max_recurrent_depth);
    vector<RNN_Node_Interface*> rnn_nodes;
    vector< vector<RNN_Node_Interface*> > layer_nodes(2 + number_hidden_layers);
    vector<RNN_Edge*> rnn_edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    int node_innovation_count = 0;
    int edge_innovation_count = 0;
    int current_layer = 0;

    for (int32_t i = 0; i < input_parameter_names.size(); i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, INPUT_LAYER, current_layer, SIMPLE_NODE, input_parameter_names[i]);
        rnn_nodes.push_back(node);
        layer_nodes[current_layer].push_back(node);
    }
    current_layer++;

    for (int32_t i = 0; i < number_hidden_layers; i++) {
        for (int32_t j = 0; j < number_hidden_nodes; j++) {
            RNN_Node *node = new RNN_Node(++node_innovation_count, HIDDEN_LAYER, current_layer, SIMPLE_NODE);
            rnn_nodes.push_back(node);
            layer_nodes[current_layer].push_back(node);

            for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
                rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], node));

                for (int32_t d = 1; d <= max_recurrent_depth; d++) {
                    recurrent_edges.push_back(new RNN_Recurrent_Edge(++edge_innovation_count, d, layer_nodes[current_layer - 1][k], node));
                }
            }

        }
        current_layer++;
    }

    for (int32_t i = 0; i < output_parameter_names.size(); i++) {
        RNN_Node *output_node = new RNN_Node(++node_innovation_count, OUTPUT_LAYER, current_layer, SIMPLE_NODE, output_parameter_names[i]);
        rnn_nodes.push_back(output_node);

        for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
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


RNN_Genome* create_jordan(const vector<string> &input_parameter_names, int number_hidden_layers, int number_hidden_nodes, const vector<string> &output_parameter_names, int max_recurrent_depth, WeightType weight_initialize) {
    Log::debug("creating jordan neural network with inputs: %d, hidden: %dx%d, outputs: %d, max recurrent depth: %d\n", input_parameter_names.size(), number_hidden_layers, number_hidden_nodes, output_parameter_names.size(), max_recurrent_depth);
    vector<RNN_Node_Interface*> rnn_nodes;
    vector<RNN_Node_Interface*> output_layer;
    vector< vector<RNN_Node_Interface*> > layer_nodes(2 + number_hidden_layers);
    vector<RNN_Edge*> rnn_edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    int node_innovation_count = 0;
    int edge_innovation_count = 0;
    int current_layer = 0;

    for (int32_t i = 0; i < input_parameter_names.size(); i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, INPUT_LAYER, current_layer, SIMPLE_NODE, input_parameter_names[i]);
        rnn_nodes.push_back(node);
        layer_nodes[current_layer].push_back(node);
    }
    current_layer++;

    for (int32_t i = 0; i < number_hidden_layers; i++) {
        for (int32_t j = 0; j < number_hidden_nodes; j++) {
            RNN_Node *node = new RNN_Node(++node_innovation_count, HIDDEN_LAYER, current_layer, JORDAN_NODE);
            rnn_nodes.push_back(node);
            layer_nodes[current_layer].push_back(node);

            for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
                rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], node));
            }
        }
        current_layer++;
    }

    for (int32_t i = 0; i < output_parameter_names.size(); i++) {
        RNN_Node *output_node = new RNN_Node(++node_innovation_count, OUTPUT_LAYER, current_layer, SIMPLE_NODE, output_parameter_names[i]);
        output_layer.push_back(output_node);

        rnn_nodes.push_back(output_node);

        for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
            rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], output_node));
        }
    }

    //connect the output node with recurrent edges to each hidden node
    for (uint32_t k = 0; k < output_layer.size(); k++) {
        for (int32_t i = 0; i < number_hidden_layers; i++) {
            for (int32_t j = 0; j < number_hidden_nodes; j++) {
                for (int32_t d = 1; d <= max_recurrent_depth; d++) {
                    recurrent_edges.push_back(new RNN_Recurrent_Edge(++edge_innovation_count, d, output_layer[k], layer_nodes[i + 1][j]));
                }
            }
        }
    }

    RNN_Genome *genome = new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges, weight_initialize, WeightType::NONE, WeightType::NONE);
    genome->set_parameter_names(input_parameter_names, output_parameter_names);
    return genome;
}

RNN_Genome* create_elman(const vector<string> &input_parameter_names, int number_hidden_layers, int number_hidden_nodes, const vector<string> &output_parameter_names, int max_recurrent_depth, WeightType weight_initialize) {
    Log::debug("creating elman network with inputs: %d, hidden: %dx%d, outputs: %d, max recurrent depth: %d\n", input_parameter_names.size(), number_hidden_layers, number_hidden_nodes, output_parameter_names.size(), max_recurrent_depth);
    vector<RNN_Node_Interface*> rnn_nodes;
    vector<RNN_Node_Interface*> output_layer;
    vector< vector<RNN_Node_Interface*> > layer_nodes(2 + number_hidden_layers);
    vector<RNN_Edge*> rnn_edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    int node_innovation_count = 0;
    int edge_innovation_count = 0;
    int current_layer = 0;

    for (int32_t i = 0; i < input_parameter_names.size(); i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, INPUT_LAYER, current_layer, SIMPLE_NODE, input_parameter_names[i]);
        rnn_nodes.push_back(node);
        layer_nodes[current_layer].push_back(node);
    }
    current_layer++;

    for (int32_t i = 0; i < number_hidden_layers; i++) {
        for (int32_t j = 0; j < number_hidden_nodes; j++) {
            RNN_Node *node = new RNN_Node(++node_innovation_count, HIDDEN_LAYER, current_layer, ELMAN_NODE);
            rnn_nodes.push_back(node);
            layer_nodes[current_layer].push_back(node);

            for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
                rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], node));
            }
        }
        current_layer++;
    }

    for (int32_t i = 0; i < output_parameter_names.size(); i++) {
        RNN_Node *output_node = new RNN_Node(++node_innovation_count, OUTPUT_LAYER, current_layer, SIMPLE_NODE, output_parameter_names[i]);
        output_layer.push_back(output_node);

        rnn_nodes.push_back(output_node);

        for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
            rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], output_node));
        }
    }

    //recurrently connect every hidden node to every other hidden node
    for (int32_t i = 0; i < number_hidden_layers; i++) {
        for (int32_t j = 0; j < number_hidden_nodes; j++) {

            for (int32_t k = 0; k < number_hidden_layers; k++) {
                for (int32_t l = 0; l < number_hidden_nodes; l++) {

                    for (int32_t d = 1; d <= max_recurrent_depth; d++) {
                        recurrent_edges.push_back(new RNN_Recurrent_Edge(++edge_innovation_count, d, layer_nodes[i+1][j], layer_nodes[k+1][l]));
                    }
                }
            }

        }
    }

    RNN_Genome *genome = new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges, weight_initialize, WeightType::NONE, WeightType::NONE);
    genome->set_parameter_names(input_parameter_names, output_parameter_names);
    return genome;
}

RNN_Genome* create_lstm(const vector<string> &input_parameter_names, int number_hidden_layers, int number_hidden_nodes, const vector<string> &output_parameter_names, int max_recurrent_depth, WeightType weight_initialize) {
    Log::debug("creating LSTM network with inputs: %d, hidden: %dx%d, outputs: %d, max recurrent depth: %d\n", input_parameter_names.size(), number_hidden_layers, number_hidden_nodes, output_parameter_names.size(), max_recurrent_depth);
    vector<RNN_Node_Interface*> rnn_nodes;
    vector< vector<RNN_Node_Interface*> > layer_nodes(2 + number_hidden_layers);
    vector<RNN_Edge*> rnn_edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    int node_innovation_count = 0;
    int edge_innovation_count = 0;
    int current_layer = 0;

    for (int32_t i = 0; i < input_parameter_names.size(); i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, INPUT_LAYER, current_layer, SIMPLE_NODE, input_parameter_names[i]);
        rnn_nodes.push_back(node);
        layer_nodes[current_layer].push_back(node);
    }
    current_layer++;

    for (int32_t i = 0; i < number_hidden_layers; i++) {
        for (int32_t j = 0; j < number_hidden_nodes; j++) {
            LSTM_Node *node = new LSTM_Node(++node_innovation_count, HIDDEN_LAYER, current_layer);
            rnn_nodes.push_back(node);
            layer_nodes[current_layer].push_back(node);

            for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
                rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], node));
            }
        }
        current_layer++;
    }

    for (int32_t i = 0; i < output_parameter_names.size(); i++) {
        RNN_Node *output_node = new RNN_Node(++node_innovation_count, OUTPUT_LAYER, current_layer, SIMPLE_NODE, output_parameter_names[i]);
        rnn_nodes.push_back(output_node);

        for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
            rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], output_node));
        }
    }

    RNN_Genome *genome = new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges, weight_initialize, WeightType::NONE, WeightType::NONE);
    genome->set_parameter_names(input_parameter_names, output_parameter_names);
    return genome;
}


RNN_Genome* create_ugrnn(const vector<string> &input_parameter_names, int number_hidden_layers, int number_hidden_nodes, const vector<string> &output_parameter_names, int max_recurrent_depth, WeightType weight_initialize) {
    Log::debug("creating UGRNN network with inputs: %d, hidden: %dx%d, outputs: %d, max recurrent depth: %d\n", input_parameter_names.size(), number_hidden_layers, number_hidden_nodes, output_parameter_names.size(), max_recurrent_depth);
    vector<RNN_Node_Interface*> rnn_nodes;
    vector< vector<RNN_Node_Interface*> > layer_nodes(2 + number_hidden_layers);
    vector<RNN_Edge*> rnn_edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    int node_innovation_count = 0;
    int edge_innovation_count = 0;
    int current_layer = 0;

    for (int32_t i = 0; i < input_parameter_names.size(); i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, INPUT_LAYER, current_layer, SIMPLE_NODE, input_parameter_names[i]);
        rnn_nodes.push_back(node);
        layer_nodes[current_layer].push_back(node);
    }
    current_layer++;

    for (int32_t i = 0; i < number_hidden_layers; i++) {
        for (int32_t j = 0; j < number_hidden_nodes; j++) {
            UGRNN_Node *node = new UGRNN_Node(++node_innovation_count, HIDDEN_LAYER, current_layer);
            rnn_nodes.push_back(node);
            layer_nodes[current_layer].push_back(node);

            for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
                rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], node));
            }
        }
        current_layer++;
    }

    for (int32_t i = 0; i < output_parameter_names.size(); i++) {
        RNN_Node *output_node = new RNN_Node(++node_innovation_count, OUTPUT_LAYER, current_layer, SIMPLE_NODE, output_parameter_names[i]);
        rnn_nodes.push_back(output_node);

        for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
            rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], output_node));
        }
    }

    RNN_Genome *genome = new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges, weight_initialize, WeightType::NONE, WeightType::NONE);
    genome->set_parameter_names(input_parameter_names, output_parameter_names);
    return genome;
}



RNN_Genome* create_gru(const vector<string> &input_parameter_names, int number_hidden_layers, int number_hidden_nodes, const vector<string> &output_parameter_names, int max_recurrent_depth, WeightType weight_initialize) {
    Log::debug("creating GRU network with inputs: %d, hidden: %dx%d, outputs: %d, max recurrent depth: %d\n", input_parameter_names.size(), number_hidden_layers, number_hidden_nodes, output_parameter_names.size(), max_recurrent_depth);
    vector<RNN_Node_Interface*> rnn_nodes;
    vector< vector<RNN_Node_Interface*> > layer_nodes(2 + number_hidden_layers);
    vector<RNN_Edge*> rnn_edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    int node_innovation_count = 0;
    int edge_innovation_count = 0;
    int current_layer = 0;

    for (int32_t i = 0; i < input_parameter_names.size(); i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, INPUT_LAYER, current_layer, SIMPLE_NODE, input_parameter_names[i]);
        rnn_nodes.push_back(node);
        layer_nodes[current_layer].push_back(node);
    }
    current_layer++;

    for (int32_t i = 0; i < number_hidden_layers; i++) {
        for (int32_t j = 0; j < number_hidden_nodes; j++) {
            GRU_Node *node = new GRU_Node(++node_innovation_count, HIDDEN_LAYER, current_layer);
            rnn_nodes.push_back(node);
            layer_nodes[current_layer].push_back(node);

            for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
                rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], node));
            }
        }
        current_layer++;
    }

    for (int32_t i = 0; i < output_parameter_names.size(); i++) {
        RNN_Node *output_node = new RNN_Node(++node_innovation_count, OUTPUT_LAYER, current_layer, SIMPLE_NODE, output_parameter_names[i]);
        rnn_nodes.push_back(output_node);

        for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
            rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], output_node));
        }
    }

    RNN_Genome *genome = new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges, weight_initialize, WeightType::NONE, WeightType::NONE);
    genome->set_parameter_names(input_parameter_names, output_parameter_names);
    return genome;
}

RNN_Genome* create_enarc(const vector<string> &input_parameter_names, int number_hidden_layers, int number_hidden_nodes, const vector<string> &output_parameter_names, int max_recurrent_depth, WeightType weight_initialize) {
    Log::debug("creating ENARC network with inputs: %d, hidden: %dx%d, outputs: %d, max recurrent depth: %d\n", input_parameter_names.size(), number_hidden_layers, number_hidden_nodes, output_parameter_names.size(), max_recurrent_depth);
    vector<RNN_Node_Interface*> rnn_nodes;
    vector< vector<RNN_Node_Interface*> > layer_nodes(2 + number_hidden_layers);
    vector<RNN_Edge*> rnn_edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    int node_innovation_count = 0;
    int edge_innovation_count = 0;
    int current_layer = 0;

    for (int32_t i = 0; i < input_parameter_names.size(); i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, INPUT_LAYER, current_layer, SIMPLE_NODE, input_parameter_names[i]);
        rnn_nodes.push_back(node);
        layer_nodes[current_layer].push_back(node);
    }
    current_layer++;

    for (int32_t i = 0; i < number_hidden_layers; i++) {
        for (int32_t j = 0; j < number_hidden_nodes; j++) {
            ENARC_Node *node = new ENARC_Node(++node_innovation_count, HIDDEN_LAYER, current_layer);
            rnn_nodes.push_back(node);
            layer_nodes[current_layer].push_back(node);

            for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
                rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], node));
            }
        }
        current_layer++;
    }

    for (int32_t i = 0; i < output_parameter_names.size(); i++) {
        RNN_Node *output_node = new RNN_Node(++node_innovation_count, OUTPUT_LAYER, current_layer, SIMPLE_NODE, output_parameter_names[i]);
        rnn_nodes.push_back(output_node);

        for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
            rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], output_node));
        }
    }

    RNN_Genome *genome = new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges, weight_initialize, WeightType::NONE, WeightType::NONE);
    genome->set_parameter_names(input_parameter_names, output_parameter_names);
    return genome;
}

RNN_Genome* create_enas_dag(const vector<string> &input_parameter_names, int number_hidden_layers, int number_hidden_nodes, const vector<string> &output_parameter_names, int max_recurrent_depth, WeightType weight_initialize) {
    Log::debug("creating ENAS_DAG network with inputs: %d, hidden: %dx%d, outputs: %d, max recurrent depth: %d\n", input_parameter_names.size(), number_hidden_layers, number_hidden_nodes, output_parameter_names.size(), max_recurrent_depth);
    vector<RNN_Node_Interface*> rnn_nodes;
    vector< vector<RNN_Node_Interface*> > layer_nodes(2 + number_hidden_layers);
    vector<RNN_Edge*> rnn_edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    int node_innovation_count = 0;
    int edge_innovation_count = 0;
    int current_layer = 0;

    for (int32_t i = 0; i < input_parameter_names.size(); i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, INPUT_LAYER, current_layer, SIMPLE_NODE, input_parameter_names[i]);
        rnn_nodes.push_back(node);
        layer_nodes[current_layer].push_back(node);
    }
    current_layer++;
    Log::debug("creating ENAS_DAG Node\n");
    for (int32_t i = 0; i < number_hidden_layers; i++) {
        for (int32_t j = 0; j < number_hidden_nodes; j++) {
            ENAS_DAG_Node *node = new ENAS_DAG_Node(++node_innovation_count, HIDDEN_LAYER, current_layer);
            rnn_nodes.push_back(node);
            layer_nodes[current_layer].push_back(node);

            for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
                rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], node));
            }
        }
        current_layer++;
    }

    for (int32_t i = 0; i < output_parameter_names.size(); i++) {
        RNN_Node *output_node = new RNN_Node(++node_innovation_count, OUTPUT_LAYER, current_layer, SIMPLE_NODE, output_parameter_names[i]);
        rnn_nodes.push_back(output_node);

        for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
            rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], output_node));
        }
    }

    RNN_Genome *genome = new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges, weight_initialize, WeightType::NONE, WeightType::NONE);
    genome->set_parameter_names(input_parameter_names, output_parameter_names);
    return genome;
}


RNN_Genome* create_random_dag(const vector<string> &input_parameter_names, int number_hidden_layers, int number_hidden_nodes, const vector<string> &output_parameter_names, int max_recurrent_depth, WeightType weight_initialize) {
    Log::debug("creating RANDOM_DAG network with inputs: %d, hidden: %dx%d, outputs: %d, max recurrent depth: %d\n", input_parameter_names.size(), number_hidden_layers, number_hidden_nodes, output_parameter_names.size(), max_recurrent_depth);
    vector<RNN_Node_Interface*> rnn_nodes;
    vector< vector<RNN_Node_Interface*> > layer_nodes(2 + number_hidden_layers);
    vector<RNN_Edge*> rnn_edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    int node_innovation_count = 0;
    int edge_innovation_count = 0;
    int current_layer = 0;

    for (int32_t i = 0; i < input_parameter_names.size(); i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, INPUT_LAYER, current_layer, SIMPLE_NODE, input_parameter_names[i]);
        rnn_nodes.push_back(node);
        layer_nodes[current_layer].push_back(node);
    }
    current_layer++;
    Log::debug("creating RANDOM_DAG Node\n");
    for (int32_t i = 0; i < number_hidden_layers; i++) {
        for (int32_t j = 0; j < number_hidden_nodes; j++) {
            RANDOM_DAG_Node *node = new RANDOM_DAG_Node(++node_innovation_count, HIDDEN_LAYER, current_layer);
            rnn_nodes.push_back(node);
            layer_nodes[current_layer].push_back(node);

            for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
                rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], node));
            }
        }
        current_layer++;
    }

    for (int32_t i = 0; i < output_parameter_names.size(); i++) {
        RNN_Node *output_node = new RNN_Node(++node_innovation_count, OUTPUT_LAYER, current_layer, SIMPLE_NODE, output_parameter_names[i]);
        rnn_nodes.push_back(output_node);

        for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
            rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], output_node));
        }
    }

    RNN_Genome *genome = new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges, weight_initialize, WeightType::NONE, WeightType::NONE);
    genome->set_parameter_names(input_parameter_names, output_parameter_names);
    return genome;
}


RNN_Genome* create_mgu(const vector<string> &input_parameter_names, int number_hidden_layers, int number_hidden_nodes, const vector<string> &output_parameter_names, int max_recurrent_depth, WeightType weight_initialize) {
    Log::debug("creating MGU network with inputs: %d, hidden: %dx%d, outputs: %d, max recurrent depth: %d\n", input_parameter_names.size(), number_hidden_layers, number_hidden_nodes, output_parameter_names.size(), max_recurrent_depth);
    vector<RNN_Node_Interface*> rnn_nodes;
    vector< vector<RNN_Node_Interface*> > layer_nodes(2 + number_hidden_layers);
    vector<RNN_Edge*> rnn_edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    int node_innovation_count = 0;
    int edge_innovation_count = 0;
    int current_layer = 0;

    for (int32_t i = 0; i < input_parameter_names.size(); i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, INPUT_LAYER, current_layer, SIMPLE_NODE, input_parameter_names[i]);
        rnn_nodes.push_back(node);
        layer_nodes[current_layer].push_back(node);
    }
    current_layer++;

    for (int32_t i = 0; i < number_hidden_layers; i++) {
        for (int32_t j = 0; j < number_hidden_nodes; j++) {
            MGU_Node *node = new MGU_Node(++node_innovation_count, HIDDEN_LAYER, current_layer);
            rnn_nodes.push_back(node);
            layer_nodes[current_layer].push_back(node);

            for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
                rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], node));
            }
        }
        current_layer++;
    }

    for (int32_t i = 0; i < output_parameter_names.size(); i++) {
        RNN_Node *output_node = new RNN_Node(++node_innovation_count, OUTPUT_LAYER, current_layer, SIMPLE_NODE, output_parameter_names[i]);
        rnn_nodes.push_back(output_node);

        for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
            rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], output_node));
        }
    }

    RNN_Genome *genome = new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges, weight_initialize, WeightType::NONE, WeightType::NONE);
    genome->set_parameter_names(input_parameter_names, output_parameter_names);
    return genome;
}


RNN_Genome* create_delta(const vector<string> &input_parameter_names, int number_hidden_layers, int number_hidden_nodes, const vector<string> &output_parameter_names, int max_recurrent_depth, WeightType weight_initialize) {
    Log::debug("creating delta network with inputs: %d, hidden: %dx%d, outputs: %d, max recurrent depth: %d\n", input_parameter_names.size(), number_hidden_layers, number_hidden_nodes, output_parameter_names.size(), max_recurrent_depth);
    vector<RNN_Node_Interface*> rnn_nodes;
    vector< vector<RNN_Node_Interface*> > layer_nodes(2 + number_hidden_layers);
    vector<RNN_Edge*> rnn_edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    int node_innovation_count = 0;
    int edge_innovation_count = 0;
    int current_layer = 0;

    for (int32_t i = 0; i < input_parameter_names.size(); i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, INPUT_LAYER, current_layer, SIMPLE_NODE, input_parameter_names[i]);
        rnn_nodes.push_back(node);
        layer_nodes[current_layer].push_back(node);
    }
    current_layer++;

    for (int32_t i = 0; i < number_hidden_layers; i++) {
        for (int32_t j = 0; j < number_hidden_nodes; j++) {
            Delta_Node *node = new Delta_Node(++node_innovation_count, HIDDEN_LAYER, current_layer);
            rnn_nodes.push_back(node);
            layer_nodes[current_layer].push_back(node);

            for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
                rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], node));
            }
        }
        current_layer++;
    }

    for (int32_t i = 0; i < output_parameter_names.size(); i++) {
        RNN_Node *output_node = new RNN_Node(++node_innovation_count, OUTPUT_LAYER, current_layer, SIMPLE_NODE, output_parameter_names[i]);
        rnn_nodes.push_back(output_node);

        for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
            rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], output_node));
        }
    }

    RNN_Genome *genome = new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges, weight_initialize, WeightType::NONE, WeightType::NONE);
    genome->set_parameter_names(input_parameter_names, output_parameter_names);
    return genome;
}
