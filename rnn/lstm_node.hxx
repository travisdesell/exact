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

        double ld_output_gate;
        double ld_input_gate;
        double ld_forget_gate;

        double cell_in_tanh;
        double cell_out_tanh;
        double ld_cell_in;
        double ld_cell_out;

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

#ifdef LSTM_TEST
        friend int main(int argc, char **argv);
#endif
};
#endif
