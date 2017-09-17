#ifndef EXALT_LSTM_NODE_HXX
#define EXALT_LSTM_NODE_HXX

#include <vector>
using std::vector;

#include "rnn_node_interface.hxx"

class LSTM_Node : public RNN_Node_Interface {
    private:
        double input_gate_weight;
        double output_gate_weight;
        double forget_gate_weight;
        double cell_weight;

        double input_gate_update_weight;
        double output_gate_update_weight;
        double forget_gate_update_weight;

        double input_gate_value;
        double output_gate_value;
        double forget_gate_value;
        double cell_value;
        double previous_cell_value;

        double input_gate_bias;
        double output_gate_bias;
        double forget_gate_bias;
        double cell_bias;

    public:

        LSTM_Node(int _innovation_number, int _type);

        void input_fired();
        void output_fired();

        uint32_t get_number_weights();
        void set_weights(uint32_t &offset, const vector<double> &parameters);

        void reset();
        void full_reset();

        void print_cell_values();

        friend class RNN_Edge;
};
#endif
