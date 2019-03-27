#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <vector>
using std::vector;

#include<map>
using std::map;

#include "edge_pheromone.hxx"
#include "node_pheromone.hxx"
#include "rnn/delta_node.hxx"
#include "rnn/ugrnn_node.hxx"
#include "rnn/gru_node.hxx"
#include "rnn/mgu_node.hxx"
#include "rnn/lstm_node.hxx"
#include "rnn/rnn_edge.hxx"
#include "rnn/rnn_genome.hxx"
#include "rnn/rnn_node.hxx"
#include "rnn/rnn_node_interface.hxx"

RNN_Genome* create_ff(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs, int max_recurrent_depth) {
    //cout << "creating ff with inputs: " << number_inputs << ", hidden: " << number_hidden_layers << "x" << number_hidden_nodes << ", outputs: " << number_outputs << endl;
    vector<RNN_Node_Interface*> rnn_nodes;
    vector< vector<RNN_Node_Interface*> > layer_nodes(2 + number_hidden_layers);
    vector<RNN_Edge*> rnn_edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    int node_innovation_count = 0;
    int edge_innovation_count = 0;
    int current_layer = 0;

    for (int32_t i = 0; i < number_inputs; i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, INPUT_LAYER, current_layer, FEED_FORWARD_NODE);
        rnn_nodes.push_back(node);
        layer_nodes[current_layer].push_back(node);
    }
    current_layer++;

    for (int32_t i = 0; i < number_hidden_layers; i++) {
        for (uint32_t j = 0; j < number_hidden_nodes; j++) {
            RNN_Node *node = new RNN_Node(++node_innovation_count, HIDDEN_LAYER, current_layer, FEED_FORWARD_NODE);
            rnn_nodes.push_back(node);
            layer_nodes[current_layer].push_back(node);

            for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
                rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], node));

                for (uint32_t d = 1; d <= max_recurrent_depth; d++) {
                    recurrent_edges.push_back(new RNN_Recurrent_Edge(++edge_innovation_count, d, layer_nodes[current_layer - 1][k], node));
                }
            }

        }
        current_layer++;
    }

    for (int32_t i = 0; i < number_outputs; i++) {
        RNN_Node *output_node = new RNN_Node(++node_innovation_count, OUTPUT_LAYER, current_layer, FEED_FORWARD_NODE);
        rnn_nodes.push_back(output_node);

        for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
            rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], output_node));

            for (uint32_t d = 1; d <= max_recurrent_depth; d++) {
                recurrent_edges.push_back(new RNN_Recurrent_Edge(++edge_innovation_count, d, layer_nodes[current_layer - 1][k], output_node));
            }
        }
    }

    return new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges);
}


RNN_Genome* create_jordan(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs, int max_recurrent_depth) {
    vector<RNN_Node_Interface*> rnn_nodes;
    vector<RNN_Node_Interface*> output_layer;
    vector< vector<RNN_Node_Interface*> > layer_nodes(2 + number_hidden_layers);
    vector<RNN_Edge*> rnn_edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    int node_innovation_count = 0;
    int edge_innovation_count = 0;
    int current_layer = 0;

    for (int32_t i = 0; i < number_inputs; i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, INPUT_LAYER, current_layer, FEED_FORWARD_NODE);
        rnn_nodes.push_back(node);
        layer_nodes[current_layer].push_back(node);
    }
    current_layer++;

    for (int32_t i = 0; i < number_hidden_layers; i++) {
        for (uint32_t j = 0; j < number_hidden_nodes; j++) {
            RNN_Node *node = new RNN_Node(++node_innovation_count, HIDDEN_LAYER, current_layer, JORDAN_NODE);
            rnn_nodes.push_back(node);
            layer_nodes[current_layer].push_back(node);

            for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
                rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], node));
            }
        }
        current_layer++;
    }

    for (int32_t i = 0; i < number_outputs; i++) {
        RNN_Node *output_node = new RNN_Node(++node_innovation_count, OUTPUT_LAYER, current_layer, FEED_FORWARD_NODE);
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
                for (uint32_t d = 1; d <= max_recurrent_depth; d++) {
                    recurrent_edges.push_back(new RNN_Recurrent_Edge(++edge_innovation_count, d, output_layer[k], layer_nodes[i + 1][j]));
                }
            }
        }
    }


    return new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges);
}

RNN_Genome* create_elman(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs, int max_recurrent_depth) {
    vector<RNN_Node_Interface*> rnn_nodes;
    vector<RNN_Node_Interface*> output_layer;
    vector< vector<RNN_Node_Interface*> > layer_nodes(2 + number_hidden_layers);
    vector<RNN_Edge*> rnn_edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    int node_innovation_count = 0;
    int edge_innovation_count = 0;
    int current_layer = 0;

    for (int32_t i = 0; i < number_inputs; i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, INPUT_LAYER, current_layer, FEED_FORWARD_NODE);
        rnn_nodes.push_back(node);
        layer_nodes[current_layer].push_back(node);
    }
    current_layer++;

    for (int32_t i = 0; i < number_hidden_layers; i++) {
        for (uint32_t j = 0; j < number_hidden_nodes; j++) {
            RNN_Node *node = new RNN_Node(++node_innovation_count, HIDDEN_LAYER, current_layer, ELMAN_NODE);
            rnn_nodes.push_back(node);
            layer_nodes[current_layer].push_back(node);

            for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
                rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], node));
            }
        }
        current_layer++;
    }

    for (int32_t i = 0; i < number_outputs; i++) {
        RNN_Node *output_node = new RNN_Node(++node_innovation_count, OUTPUT_LAYER, current_layer, FEED_FORWARD_NODE);
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


                    for (uint32_t d = 1; d <= max_recurrent_depth; d++) {
                        recurrent_edges.push_back(new RNN_Recurrent_Edge(++edge_innovation_count, d, layer_nodes[i+1][j], layer_nodes[k+1][l]));
                    }
                }
            }

        }
    }

    return new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges);
}

RNN_Genome* create_lstm(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs, int max_recurrent_depth) {
    vector<RNN_Node_Interface*> rnn_nodes;
    vector< vector<RNN_Node_Interface*> > layer_nodes(2 + number_hidden_layers);
    vector<RNN_Edge*> rnn_edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    int node_innovation_count = 0;
    int edge_innovation_count = 0;
    int current_layer = 0;

    for (int32_t i = 0; i < number_inputs; i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, INPUT_LAYER, current_layer, FEED_FORWARD_NODE);
        rnn_nodes.push_back(node);
        layer_nodes[current_layer].push_back(node);
    }
    current_layer++;

    for (int32_t i = 0; i < number_hidden_layers; i++) {
        for (uint32_t j = 0; j < number_hidden_nodes; j++) {
            LSTM_Node *node = new LSTM_Node(++node_innovation_count, HIDDEN_LAYER, current_layer);
            rnn_nodes.push_back(node);
            layer_nodes[current_layer].push_back(node);

            for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
                rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], node));
            }
        }
        current_layer++;
    }

    for (int32_t i = 0; i < number_outputs; i++) {
        RNN_Node *output_node = new RNN_Node(++node_innovation_count, OUTPUT_LAYER, current_layer, FEED_FORWARD_NODE);
        rnn_nodes.push_back(output_node);

        for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
            rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], output_node));
        }
    }

    return new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges);
}


RNN_Genome* create_ugrnn(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs, int max_recurrent_depth) {
    vector<RNN_Node_Interface*> rnn_nodes;
    vector< vector<RNN_Node_Interface*> > layer_nodes(2 + number_hidden_layers);
    vector<RNN_Edge*> rnn_edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    int node_innovation_count = 0;
    int edge_innovation_count = 0;
    int current_layer = 0;

    for (int32_t i = 0; i < number_inputs; i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, INPUT_LAYER, current_layer, FEED_FORWARD_NODE);
        rnn_nodes.push_back(node);
        layer_nodes[current_layer].push_back(node);
    }
    current_layer++;

    for (int32_t i = 0; i < number_hidden_layers; i++) {
        for (uint32_t j = 0; j < number_hidden_nodes; j++) {
            UGRNN_Node *node = new UGRNN_Node(++node_innovation_count, HIDDEN_LAYER, current_layer);
            rnn_nodes.push_back(node);
            layer_nodes[current_layer].push_back(node);

            for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
                rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], node));
            }
        }
        current_layer++;
    }

    for (int32_t i = 0; i < number_outputs; i++) {
        RNN_Node *output_node = new RNN_Node(++node_innovation_count, OUTPUT_LAYER, current_layer, FEED_FORWARD_NODE);
        rnn_nodes.push_back(output_node);

        for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
            rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], output_node));
        }
    }

    return new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges);
}



RNN_Genome* create_gru(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs, int max_recurrent_depth) {
    vector<RNN_Node_Interface*> rnn_nodes;
    vector< vector<RNN_Node_Interface*> > layer_nodes(2 + number_hidden_layers);
    vector<RNN_Edge*> rnn_edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    int node_innovation_count = 0;
    int edge_innovation_count = 0;
    int current_layer = 0;

    for (int32_t i = 0; i < number_inputs; i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, INPUT_LAYER, current_layer, FEED_FORWARD_NODE);
        rnn_nodes.push_back(node);
        layer_nodes[current_layer].push_back(node);
    }
    current_layer++;

    for (int32_t i = 0; i < number_hidden_layers; i++) {
        for (uint32_t j = 0; j < number_hidden_nodes; j++) {
            GRU_Node *node = new GRU_Node(++node_innovation_count, HIDDEN_LAYER, current_layer);
            rnn_nodes.push_back(node);
            layer_nodes[current_layer].push_back(node);

            for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
                rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], node));
            }
        }
        current_layer++;
    }

    for (int32_t i = 0; i < number_outputs; i++) {
        RNN_Node *output_node = new RNN_Node(++node_innovation_count, OUTPUT_LAYER, current_layer, FEED_FORWARD_NODE);
        rnn_nodes.push_back(output_node);

        for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
            rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], output_node));
        }
    }

    return new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges);
}
RNN_Genome* create_mgu(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs, int max_recurrent_depth) {
    vector<RNN_Node_Interface*> rnn_nodes;
    vector< vector<RNN_Node_Interface*> > layer_nodes(2 + number_hidden_layers);
    vector<RNN_Edge*> rnn_edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    int node_innovation_count = 0;
    int edge_innovation_count = 0;
    int current_layer = 0;

    for (int32_t i = 0; i < number_inputs; i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, INPUT_LAYER, current_layer, FEED_FORWARD_NODE);
        rnn_nodes.push_back(node);
        layer_nodes[current_layer].push_back(node);
    }
    current_layer++;

    for (int32_t i = 0; i < number_hidden_layers; i++) {
        for (uint32_t j = 0; j < number_hidden_nodes; j++) {
            MGU_Node *node = new MGU_Node(++node_innovation_count, HIDDEN_LAYER, current_layer);
            rnn_nodes.push_back(node);
            layer_nodes[current_layer].push_back(node);

            for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
                rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], node));
            }
        }
        current_layer++;
    }

    for (int32_t i = 0; i < number_outputs; i++) {
        RNN_Node *output_node = new RNN_Node(++node_innovation_count, OUTPUT_LAYER, current_layer, FEED_FORWARD_NODE);
        rnn_nodes.push_back(output_node);

        for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
            rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], output_node));
        }
    }

    return new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges);
}


RNN_Genome* create_delta(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs, int max_recurrent_depth) {
    vector<RNN_Node_Interface*> rnn_nodes;
    vector< vector<RNN_Node_Interface*> > layer_nodes(2 + number_hidden_layers);
    vector<RNN_Edge*> rnn_edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    int node_innovation_count = 0;
    int edge_innovation_count = 0;
    int current_layer = 0;

    for (int32_t i = 0; i < number_inputs; i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, INPUT_LAYER, current_layer, FEED_FORWARD_NODE);
        rnn_nodes.push_back(node);
        layer_nodes[current_layer].push_back(node);
    }
    current_layer++;

    for (int32_t i = 0; i < number_hidden_layers; i++) {
        for (uint32_t j = 0; j < number_hidden_nodes; j++) {
            Delta_Node *node = new Delta_Node(++node_innovation_count, HIDDEN_LAYER, current_layer);
            rnn_nodes.push_back(node);
            layer_nodes[current_layer].push_back(node);

            for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
                rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], node));
            }
        }
        current_layer++;
    }

    for (int32_t i = 0; i < number_outputs; i++) {
        RNN_Node *output_node = new RNN_Node(++node_innovation_count, OUTPUT_LAYER, current_layer, FEED_FORWARD_NODE);
        rnn_nodes.push_back(output_node);

        for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
            rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], output_node));
        }
    }

    return new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges);
}




/*This function is added to use to construct the ant colony
while constructing the genome initial genome strcture --AbdElRahman

The colony structure components will be:
    a) Edges (lines) holding pheromones of the edges_pheromones-->EDGE_Pheromone objects. These objects will hold :
        -- The node's innovation number which they come out from.
        -- The value of the pheromones
        -- The input node innovation number (redundant)
        -- The output node innovation number
    b) Edges-pheromones (above elements) will be grouped in a vector along with the nodes-types pheromones (an array) will be held in-->NODE_Pheromones. These objects will hold:
        -- Node innovaton numbers
        -- An array of node-types pheromones
        -- A vector for the edges coming out of the node
*/

void create_colony_pheromones(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs, int max_recurrent_depth, map <int32_t, NODE_Pheromones*> &colony) {

    // map <int32_t, NODE_Pheromones*> colony;
    vector<vector<int>> layer_nodes(2 + number_hidden_layers);

    int32_t node_innovation_count = 0;
    int32_t edge_innovation_count = 0;
    // int current_layer = 0;
    vector<EDGE_Pheromone*> dum;

    /* Building the nodes*/
    for ( uint32_t i = 0; i<number_inputs; i++){
        layer_nodes[0].push_back(node_innovation_count++);
    }
    for ( uint32_t i = 1; i<number_hidden_layers+1; i++){
        for ( uint32_t j = 0; j<number_hidden_nodes; j++){
            layer_nodes[i].push_back(node_innovation_count++);
        }
    }
    for ( uint32_t i = 0; i<number_outputs; i++){
        layer_nodes[1 + number_hidden_layers].push_back(-(node_innovation_count++)); //-ve to indicate output node.
    }

    //Building the pheromones from the starting point
    for ( uint32_t i = 0; i < number_inputs; i++) {
        dum.push_back(new EDGE_Pheromone(-1, 0.7, 0, -1, layer_nodes[0][i]));
    }

    vector<EDGE_Pheromone*> *dum2 = new vector<EDGE_Pheromone*>;
    *dum2 = dum;
    colony[-1] = new NODE_Pheromones(NULL, dum2, -1, -1);

    double type_pheromones_initial_values[6] = {0.7, 0.7, 0.7, 0.7, 0.7, 0.7};     //initially all node type pheromones will be 1.0

    /*
       - Using the nodes the build the colony.
       - Every node holds all the edges and recurrent edges comming out of it.
       - Every node also holds the pheromones to for the node type.
   */
    for ( uint32_t i = 0; i<layer_nodes.size(); i++){
        int layer_type;
        if (i==0){
            layer_type = INPUT_LAYER;
        }
        else if(i>0 && i<layer_nodes.size()-1){
            layer_type = HIDDEN_LAYER;
        }
        else if (i==layer_nodes.size()-1){
            layer_type = OUTPUT_LAYER;
        }

        for ( uint32_t j = 0; j<layer_nodes[i].size(); j++){
            dum.clear();
            int l = i+1;
            /*Building the edges' pheromones: Edge to every node in cosecutive layer*/
            while (l<layer_nodes.size()){
              for ( uint32_t k = 0; k<layer_nodes[l].size(); k++){
                  dum.push_back(new EDGE_Pheromone(edge_innovation_count++, 0.7, 0, layer_nodes[i][j], layer_nodes[l][k]));
              }
              l++;
            }
            /*Building recurrent edges' pheromones: Re-Edge to every node in each cosecutive time step*/
            for ( uint32_t m = 1; m<layer_nodes.size(); m++){
                for ( uint32_t n = 0; n<layer_nodes[m].size(); n++){
                    for (uint32_t d = 1; d < max_recurrent_depth; d++) {
                        dum.push_back(new EDGE_Pheromone(edge_innovation_count++, 0.7, d, layer_nodes[i][j], layer_nodes[m][n]));
                    }
                }
            }

            /*Node types phermones*/
            double* type_pheromones = new double[6];
            memcpy(type_pheromones, type_pheromones_initial_values, 6*sizeof(double) );
            // double type_pheromones[6];
            // std::copy(std::begin(type_pheromones_initial_values), std::end(type_pheromones_initial_values), std::begin(type_pheromones));
            vector<EDGE_Pheromone*> *dum2 = new vector<EDGE_Pheromone*>;
            *dum2 = dum;
            // vector<EDGE_Pheromone*> dum2 = dum;
            // cout << "JJ: DUM size: " <<dum.size() << " DUM2 size: " << dum2.size() << endl;
            // dum2.push_back(new EDGE_Pheromone(22,1,2,3,2));
            // cout << "JJ: DUM size: " <<dum.size() << " DUM2 size: " << dum2.size() << endl;
            // exit(1);
            colony[layer_nodes[i][j]] = new NODE_Pheromones(type_pheromones, dum2, layer_type, i);
            // colony[layer_nodes[i][j]] = new NODE_Pheromones(type_pheromones, &dum2, layer_type, i);
        }
    }
}
