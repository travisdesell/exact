#ifndef RNN_TWO_LAYER_LSTM_HXX
#define RNN_TWO_LAYER_LSTM_HXX
// #ifndef RNN_TWO_LAYER_GRU_HXX
// #define RNN_TWO_LAYER_GRU_HXX

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <vector>
using std::vector;

#include<map>
using std::map;

#include "rnn/rnn_genome.hxx"
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



RNN_Genome* create_ff(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs, int max_recurrent_depth);

RNN_Genome* create_jordan(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs, int max_recurrent_depth);

RNN_Genome* create_elman(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs, int max_recurrent_depth);

RNN_Genome* create_lstm(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs, int max_recurrent_depth);

RNN_Genome* create_delta(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs, int max_recurrent_depth);

RNN_Genome* create_gru(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs, int max_recurrent_depth);

RNN_Genome* create_mgu(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs, int max_recurrent_depth);

RNN_Genome* create_ugrnn(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs, int max_recurrent_depth);

void         create_colony_pheromones(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs, int max_recurrent_depth,
                                      map <int32_t, NODE_Pheromones*> &colony1, double initial_pheromone);
#endif
