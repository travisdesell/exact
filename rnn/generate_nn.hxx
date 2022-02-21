#ifndef RNN_GENERATE_NN_HXX
#define RNN_GENERATE_NN_HXX

#include <string>
using std::string;

#include <functional>
using std::function;

#include <vector>
using std::vector;

#include "common/weight_initialize.hxx"
#include "rnn_genome.hxx"

RNN_Genome *create_ff(const vector<string> &input_parameter_names,
                      uint32_t number_hidden_layers,
                      uint32_t number_hidden_nodes,
                      const vector<string> &output_parameter_names,
                      uint32_t max_recurrent_depth, TrainingParameters tp,
                      WeightType weight_initialize,
                      WeightType weight_inheritance,
                      WeightType mutated_component_weight);

RNN_Genome *create_jordan(const vector<string> &input_parameter_names,
                          uint32_t number_hidden_layers,
                          uint32_t number_hidden_nodes,
                          const vector<string> &output_parameter_names,
                          uint32_t max_recurrent_depth, TrainingParameters tp,
                          WeightType weight_initialize);

RNN_Genome *create_elman(const vector<string> &input_parameter_names,
                         uint32_t number_hidden_layers,
                         uint32_t number_hidden_nodes,
                         const vector<string> &output_parameter_names,
                         uint32_t max_recurrent_depth, TrainingParameters tp,
                         WeightType weight_initialize);

template <class MemoryCell>
RNN_Genome *create_memory_cell_nn(const vector<string> &input_parameter_names,
                                  uint32_t number_hidden_layers,
                                  uint32_t number_hidden_nodes,
                                  const vector<string> &output_parameter_names,
                                  uint32_t max_recurrent_depth,
                                  TrainingParameters tp,
                                  WeightType weight_initialize);

#define inst_create_memory_cell(ty)                                \
  template RNN_Genome *create_memory_cell_nn<ty>(                  \
      const vector<string> &input_parameter_names,                 \
      uint32_t number_hidden_layers, uint32_t number_hidden_nodes, \
      const vector<string> &output_parameter_names,                \
      uint32_t max_recurrent_depth, TrainingParameters tp,         \
      WeightType weight_initialize)

#include "lstm_node.hxx"
#define create_lstm create_memory_cell_nn<LSTM_Node>

#include "gru_node.hxx"
#define create_gru create_memory_cell_nn<GRU_Node>

#include "delta_node.hxx"
#define create_delta create_memory_cell_nn<Delta_Node>

#include "mgu_node.hxx"
#define create_mgu create_memory_cell_nn<MGU_Node>

#include "ugrnn_node.hxx"
#define create_ugrnn create_memory_cell_nn<UGRNN_Node>

#endif
