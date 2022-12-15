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

RNN_Genome* create_ff(const vector<string> &input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes, const vector<string> &output_parameter_names, int32_t max_recurrent_depth, WeightType weight_initialize, WeightType weight_inheritance, WeightType mutated_component_weight) {
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
            RNN_Node *node = new RNN_Node(++node_innovation_count, HIDDEN_LAYER, current_layer, SIMPLE_NODE);
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


RNN_Genome* create_jordan(const vector<string> &input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes, const vector<string> &output_parameter_names, int32_t max_recurrent_depth, WeightType weight_initialize) {
    Log::debug("creating jordan neural network with inputs: %d, hidden: %dx%d, outputs: %d, max recurrent depth: %d\n", input_parameter_names.size(), number_hidden_layers, number_hidden_nodes, output_parameter_names.size(), max_recurrent_depth);
    vector<RNN_Node_Interface*> rnn_nodes;
    vector<RNN_Node_Interface*> output_layer;
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
            RNN_Node *node = new RNN_Node(++node_innovation_count, HIDDEN_LAYER, current_layer, JORDAN_NODE);
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
        output_layer.push_back(output_node);

        rnn_nodes.push_back(output_node);

        for (int32_t k = 0; k < (int32_t)layer_nodes[current_layer - 1].size(); k++) {
            rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], output_node));
        }
    }

    //connect the output node with recurrent edges to each hidden node
    for (int32_t k = 0; k < (int32_t)output_layer.size(); k++) {
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

RNN_Genome* create_elman(const vector<string> &input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes, const vector<string> &output_parameter_names, int32_t max_recurrent_depth, WeightType weight_initialize) {
    Log::debug("creating elman network with inputs: %d, hidden: %dx%d, outputs: %d, max recurrent depth: %d\n", input_parameter_names.size(), number_hidden_layers, number_hidden_nodes, output_parameter_names.size(), max_recurrent_depth);
    vector<RNN_Node_Interface*> rnn_nodes;
    vector<RNN_Node_Interface*> output_layer;
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
            RNN_Node *node = new RNN_Node(++node_innovation_count, HIDDEN_LAYER, current_layer, ELMAN_NODE);
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
        output_layer.push_back(output_node);

        rnn_nodes.push_back(output_node);

        for (int32_t k = 0; k < (int32_t)layer_nodes[current_layer - 1].size(); k++) {
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

RNN_Genome* create_lstm(const vector<string> &input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes, const vector<string> &output_parameter_names, int32_t max_recurrent_depth, WeightType weight_initialize) {
    Log::debug("creating LSTM network with inputs: %d, hidden: %dx%d, outputs: %d, max recurrent depth: %d\n", input_parameter_names.size(), number_hidden_layers, number_hidden_nodes, output_parameter_names.size(), max_recurrent_depth);
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
            LSTM_Node *node = new LSTM_Node(++node_innovation_count, HIDDEN_LAYER, current_layer);
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


RNN_Genome* create_ugrnn(const vector<string> &input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes, const vector<string> &output_parameter_names, int32_t max_recurrent_depth, WeightType weight_initialize) {
    Log::debug("creating UGRNN network with inputs: %d, hidden: %dx%d, outputs: %d, max recurrent depth: %d\n", input_parameter_names.size(), number_hidden_layers, number_hidden_nodes, output_parameter_names.size(), max_recurrent_depth);
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
            UGRNN_Node *node = new UGRNN_Node(++node_innovation_count, HIDDEN_LAYER, current_layer);
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



RNN_Genome* create_gru(const vector<string> &input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes, const vector<string> &output_parameter_names, int32_t max_recurrent_depth, WeightType weight_initialize) {
    Log::debug("creating GRU network with inputs: %d, hidden: %dx%d, outputs: %d, max recurrent depth: %d\n", input_parameter_names.size(), number_hidden_layers, number_hidden_nodes, output_parameter_names.size(), max_recurrent_depth);
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
            GRU_Node *node = new GRU_Node(++node_innovation_count, HIDDEN_LAYER, current_layer);
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

RNN_Genome* create_enarc(const vector<string> &input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes, const vector<string> &output_parameter_names, int32_t max_recurrent_depth, WeightType weight_initialize) {
    Log::debug("creating ENARC network with inputs: %d, hidden: %dx%d, outputs: %d, max recurrent depth: %d\n", input_parameter_names.size(), number_hidden_layers, number_hidden_nodes, output_parameter_names.size(), max_recurrent_depth);
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
            ENARC_Node *node = new ENARC_Node(++node_innovation_count, HIDDEN_LAYER, current_layer);
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

RNN_Genome* create_enas_dag(const vector<string> &input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes, const vector<string> &output_parameter_names, int32_t max_recurrent_depth, WeightType weight_initialize) {
    Log::debug("creating ENAS_DAG network with inputs: %d, hidden: %dx%d, outputs: %d, max recurrent depth: %d\n", input_parameter_names.size(), number_hidden_layers, number_hidden_nodes, output_parameter_names.size(), max_recurrent_depth);
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
    Log::debug("creating ENAS_DAG Node\n");
    for (int32_t i = 0; i < number_hidden_layers; i++) {
        for (int32_t j = 0; j < number_hidden_nodes; j++) {
            ENAS_DAG_Node *node = new ENAS_DAG_Node(++node_innovation_count, HIDDEN_LAYER, current_layer);
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


RNN_Genome* create_random_dag(const vector<string> &input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes, const vector<string> &output_parameter_names, int32_t max_recurrent_depth, WeightType weight_initialize) {
    Log::debug("creating RANDOM_DAG network with inputs: %d, hidden: %dx%d, outputs: %d, max recurrent depth: %d\n", input_parameter_names.size(), number_hidden_layers, number_hidden_nodes, output_parameter_names.size(), max_recurrent_depth);
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
    Log::debug("creating RANDOM_DAG Node\n");
    for (int32_t i = 0; i < number_hidden_layers; i++) {
        for (int32_t j = 0; j < number_hidden_nodes; j++) {
            RANDOM_DAG_Node *node = new RANDOM_DAG_Node(++node_innovation_count, HIDDEN_LAYER, current_layer);
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


RNN_Genome* create_mgu(const vector<string> &input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes, const vector<string> &output_parameter_names, int32_t max_recurrent_depth, WeightType weight_initialize) {
    Log::debug("creating MGU network with inputs: %d, hidden: %dx%d, outputs: %d, max recurrent depth: %d\n", input_parameter_names.size(), number_hidden_layers, number_hidden_nodes, output_parameter_names.size(), max_recurrent_depth);
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
            MGU_Node *node = new MGU_Node(++node_innovation_count, HIDDEN_LAYER, current_layer);
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


RNN_Genome* create_delta(const vector<string> &input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes, const vector<string> &output_parameter_names, int32_t max_recurrent_depth, WeightType weight_initialize) {
    Log::debug("creating delta network with inputs: %d, hidden: %dx%d, outputs: %d, max recurrent depth: %d\n", input_parameter_names.size(), number_hidden_layers, number_hidden_nodes, output_parameter_names.size(), max_recurrent_depth);
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
            Delta_Node *node = new Delta_Node(++node_innovation_count, HIDDEN_LAYER, current_layer);
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

RNN_Genome* get_seed_genome(const vector<string> &arguments, TimeSeriesSets *time_series_sets, WeightType weight_initialize, WeightType weight_inheritance, WeightType mutated_component_weight) {
    int32_t min_recurrent_depth = 1;
    get_argument(arguments, "--min_recurrent_depth", false, min_recurrent_depth);

    int32_t max_recurrent_depth = 10;
    get_argument(arguments, "--max_recurrent_depth", false, max_recurrent_depth);
    
    RNN_Genome *seed_genome = NULL;
    string genome_file_name = "";
    string transfer_learning_version = "";
    if (get_argument(arguments, "--genome_bin", false, genome_file_name)) {
        seed_genome = new RNN_Genome(genome_file_name);
        seed_genome->set_normalize_bounds(time_series_sets->get_normalize_type(), time_series_sets->get_normalize_mins(), time_series_sets->get_normalize_maxs(), time_series_sets->get_normalize_avgs(), time_series_sets->get_normalize_std_devs());

        get_argument(arguments, "--transfer_learning_version", true, transfer_learning_version);

        bool epigenetic_weights = argument_exists(arguments, "--epigenetic_weights");

        seed_genome->transfer_to(time_series_sets->get_input_parameter_names(), time_series_sets->get_output_parameter_names(), transfer_learning_version, epigenetic_weights, min_recurrent_depth, max_recurrent_depth);
        seed_genome->tl_with_epigenetic = epigenetic_weights ;
    } else {
        
        // bool seed_genome_was_minimal = false;
        if (seed_genome == NULL) {
            // seed_genome_was_minimal = true;
            seed_genome = create_ff(time_series_sets->get_input_parameter_names(), 0, 0, time_series_sets->get_output_parameter_names(), 0, weight_initialize, weight_inheritance, mutated_component_weight);
            seed_genome->initialize_randomly();
        } //otherwise the seed genome was passed into EXAMM

        //make sure we don't duplicate node or edge innovation numbers
        // edge_innovation_count = seed_genome->get_max_edge_innovation_count() + 1;
        // node_innovation_count = seed_genome->get_max_node_innovation_count() + 1;

        // seed_genome->set_generated_by("initial");

        // //insert a copy of it into the population so
        // //additional requests can mutate it

        // seed_genome->best_validation_mse = EXAMM_MAX_DOUBLE;

        // seed_genome->best_validation_mse = EXAMM_MAX_DOUBLE;
        // seed_genome->best_validation_mae = EXAMM_MAX_DOUBLE;
    }

    return seed_genome;
}