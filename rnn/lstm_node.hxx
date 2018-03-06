#ifndef EXALT_LSTM_NODE_HXX
#define EXALT_LSTM_NODE_HXX

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "rnn_node_interface.hxx"

class LSTM_Node : public RNN_Node_Interface {
    private:
        double output_gate_update_weight;
        double output_gate_weight;
        double output_gate_bias;

        double input_gate_update_weight;
        double input_gate_weight;
        double input_gate_bias;

        double forget_gate_update_weight;
        double forget_gate_weight;
        double forget_gate_bias;

        double cell_weight;
        double cell_bias;

        vector<double> output_gate_values;
        vector<double> input_gate_values;
        vector<double> forget_gate_values;
        vector<double> cell_values;

        vector<double> ld_output_gate;
        vector<double> ld_input_gate;
        vector<double> ld_forget_gate;

        vector<double> cell_in_tanh;
        vector<double> cell_out_tanh;
        vector<double> ld_cell_in;
        vector<double> ld_cell_out;

        vector<double> d_prev_cell;

        vector<double> d_output_gate_update_weight;
        vector<double> d_output_gate_weight;
        vector<double> d_output_gate_bias;

        vector<double> d_input_gate_update_weight;
        vector<double> d_input_gate_weight;
        vector<double> d_input_gate_bias;

        vector<double> d_forget_gate_update_weight;
        vector<double> d_forget_gate_weight;
        vector<double> d_forget_gate_bias;

        vector<double> d_cell_weight;
        vector<double> d_cell_bias;

    public:

        LSTM_Node(int _innovation_number, int _type);

        double get_gradient(string gradient_name);
        void print_gradient(string gradient_name);

        void input_fired(const vector<double> &incoming_outputs);

        void try_update_deltas();
        void output_fired(double error);
        void output_fired(const vector<double> &deltas);

        uint32_t get_number_weights();
        void set_weights(const vector<double> &parameters);
        void set_weights(uint32_t &offset, const vector<double> &parameters);

        void get_weights(uint32_t &offset, vector<double> &parameters);
        void get_gradients(vector<double> &gradients);

        void reset(int _series_length);

        void print_cell_values();

        RNN_Node_Interface* copy();

        friend class RNN_Edge;

#ifdef LSTM_TEST
        friend int main(int argc, char **argv);
#endif
};
#endif
