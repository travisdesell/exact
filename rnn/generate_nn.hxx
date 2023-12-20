#ifndef RNN_GENERATE_NN_HXX
#define RNN_GENERATE_NN_HXX

#include <functional>
#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "rnn/cos_node.hxx"
#include "rnn/delta_node.hxx"
#include "rnn/dnas_node.hxx"
#include "rnn/enarc_node.hxx"
#include "rnn/enas_dag_node.hxx"
#include "rnn/gru_node.hxx"
#include "rnn/inverse_node.hxx"
#include "rnn/lstm_node.hxx"
#include "rnn/mgu_node.hxx"
#include "rnn/multiply_node.hxx"
#include "rnn/random_dag_node.hxx"
#include "rnn/rnn_edge.hxx"
#include "rnn/rnn_genome.hxx"
#include "rnn/rnn_node.hxx"
#include "rnn/rnn_node_interface.hxx"
#include "rnn/sigmoid_node.hxx"
#include "rnn/sin_node.hxx"
#include "rnn/sum_node.hxx"
#include "rnn/tanh_node.hxx"
#include "rnn/ugrnn_node.hxx"
#include "weights/weight_rules.hxx"

template <class NodeT>
NodeT* create_hidden_memory_cell(int32_t& innovation_counter, double depth) {
    return new NodeT(++innovation_counter, HIDDEN_LAYER, depth);
}
RNN_Node_Interface* create_hidden_node(int32_t node_kind, int32_t& innovation_counter, double depth);

RNN_Genome* create_nn(
    const vector<string>& input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes,
    const vector<string>& output_parameter_names, int32_t max_recurrent_depth,
    std::function<RNN_Node_Interface*(int32_t&, double)> make_node, WeightRules* weight_rules
);

template <unsigned int Kind>
RNN_Genome* create_simple_nn(
    const vector<string>& input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes,
    const vector<string>& output_parameter_names, int32_t max_recurrent_depth, WeightRules* weight_rules
) {
    auto f = [=](int32_t& innovation_counter, double depth) -> RNN_Node_Interface* {
        return new RNN_Node(++innovation_counter, HIDDEN_LAYER, depth, Kind);
    };
    return create_nn(
        input_parameter_names, number_hidden_layers, number_hidden_nodes, output_parameter_names, max_recurrent_depth,
        f, weight_rules
    );
}

#define create_ff(...)     create_simple_nn<SIMPLE_NODE>(__VA_ARGS__)
#define create_elman(...)  create_simple_nn<ELMAN_NODE>(__VA_ARGS__)
#define create_jordan(...) create_simple_nn<JORDAN_NODE>(__VA_ARGS__)

template <typename NodeT>
RNN_Genome* create_memory_cell_nn(
    const vector<string>& input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes,
    const vector<string>& output_parameter_names, int32_t max_recurrent_depth, WeightRules* weight_rules
) {
    auto f = [=](int32_t& innovation_counter, double depth) -> RNN_Node_Interface* {
        return create_hidden_memory_cell<NodeT>(innovation_counter, depth);
    };

    return create_nn(
        input_parameter_names, number_hidden_layers, number_hidden_nodes, output_parameter_names, max_recurrent_depth,
        f, weight_rules
    );
}

#define create_mgu(...)        create_memory_cell_nn<MGU_Node>(__VA_ARGS__)
#define create_gru(...)        create_memory_cell_nn<GRU_Node>(__VA_ARGS__)
#define create_delta(...)      create_memory_cell_nn<Delta_Node>(__VA_ARGS__)
#define create_lstm(...)       create_memory_cell_nn<LSTM_Node>(__VA_ARGS__)
#define create_enarc(...)      create_memory_cell_nn<ENARC_Node>(__VA_ARGS__)
#define create_enas_dag(...)   create_memory_cell_nn<ENAS_DAG_Node>(__VA_ARGS__)
#define create_random_dag(...) create_memory_cell_nn<RANDOM_DAG_Node>(__VA_ARGS__)
#define create_ugrnn(...)      create_memory_cell_nn<UGRNN_Node>(__VA_ARGS__)

// new simple nodes
#define create_sin(...)      create_memory_cell_nn<SIN_Node>(__VA_ARGS__)
#define create_sum(...)      create_memory_cell_nn<SUM_Node>(__VA_ARGS__)
#define create_cos(...)      create_memory_cell_nn<COS_Node>(__VA_ARGS__)
#define create_tanh(...)     create_memory_cell_nn<TANH_Node>(__VA_ARGS__)
#define create_sigmoid(...)  create_memory_cell_nn<SIGMOID_Node>(__VA_ARGS__)
#define create_inverse(...)  create_memory_cell_nn<INVERSE_Node>(__VA_ARGS__)
#define create_multiply(...) create_memory_cell_nn<MULTIPLY_Node>(__VA_ARGS__)

DNASNode* create_dnas_node(int32_t& innovation_counter, double depth, const vector<int32_t>& node_types);

RNN_Genome* create_dnas_nn(
    const vector<string>& input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes,
    const vector<string>& output_parameter_names, int32_t max_recurrent_depth, vector<int32_t>& node_types,
    WeightRules* weight_rules
);

RNN_Genome* create_nn(
    const vector<string>& input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes,
    const vector<string>& output_parameter_names, int32_t max_recurrent_depth, WeightRules* weight_rules
);
RNN_Genome* get_seed_genome(
    const vector<string>& arguments, TimeSeriesSets* time_series_sets, WeightRules* weight_rules
);

#endif
