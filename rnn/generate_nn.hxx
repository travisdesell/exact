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

#include<tuple>
using std:tuple

#include<map>
using std::map

#include "rnn/rnn_genome.hxx"


RNN_Genome* create_ff(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs, int max_recurrent_depth);

RNN_Genome* create_jordan(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs, int max_recurrent_depth);

RNN_Genome* create_elman(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs, int max_recurrent_depth);

RNN_Genome* create_lstm(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs, int max_recurrent_depth);

RNN_Genome* create_delta(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs, int max_recurrent_depth);

RNN_Genome* create_gru(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs, int max_recurrent_depth);

RNN_Genome* create_mgu(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs, int max_recurrent_depth);

RNN_Genome* create_ugrnn(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs, int max_recurrent_depth);

void        create_ff_w_pheromones(int number_inputs, int number_hidden_layers, int number_hidden_nodes, int number_outputs, int max_recurrent_depth,
                                    RNN_Genome* genome, map <int32_t, NODE_Pheromones> &colony);

#endif
