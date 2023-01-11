#include "rnn/generate_nn.hxx"

#include <cassert>

#include <string>
using std::string;

#include <vector>
using std::vector;

/*
 * node_kind is the type of memory cell (e.g. LSTM, UGRNN)
 * innovation_counter - reference to an integer used to keep track if innovation numbers. it will be incremented once.
 */
RNN_Node_Interface *create_hidden_memory_cell(int32_t node_kind, int32_t &innovation_counter, double depth) {
    switch (node_kind) {
        case SIMPLE_NODE:
            return new RNN_Node(++innovation_counter, HIDDEN_LAYER, depth, SIMPLE_NODE);
        case JORDAN_NODE:
            return new RNN_Node(++innovation_counter, HIDDEN_LAYER, depth, JORDAN_NODE);
        case ELMAN_NODE:
            return new RNN_Node(++innovation_counter, HIDDEN_LAYER, depth, ELMAN_NODE);
        case UGRNN_NODE:
            return new UGRNN_Node(++innovation_counter, HIDDEN_LAYER, depth);
        case MGU_NODE:
            return new MGU_Node(++innovation_counter, HIDDEN_LAYER, depth);
        case GRU_NODE:
            return new GRU_Node(++innovation_counter, HIDDEN_LAYER, depth);
        case DELTA_NODE:
            return new Delta_Node(++innovation_counter, HIDDEN_LAYER, depth);
        case LSTM_NODE:
            return new LSTM_Node(++innovation_counter, HIDDEN_LAYER, depth);
        case ENARC_NODE:
            return new ENARC_Node(++innovation_counter, HIDDEN_LAYER, depth);
        case ENAS_DAG_NODE:
            return new ENAS_DAG_Node(++innovation_counter, HIDDEN_LAYER, depth);
        case RANDOM_DAG_NODE:
            return new RANDOM_DAG_Node(++innovation_counter, HIDDEN_LAYER, depth);
        case DNAS_NODE:
            Log::fatal("You shouldn't be creating DNAS nodes using generate_nn::create_hidden_node.\n");
            exit(1);
    }
}

RNN_Node *create_hidden_node(int32_t node_kind, int32_t &innovation_counter, double depth) {
    assert(node_kind == SIMPLE_NODE || node_kind == JORDAN_NODE || node_kind == ELMAN_NODE);
    return new RNN_Node(++innovation_counter, HIDDEN_LAYER, depth, node_kind);
}
