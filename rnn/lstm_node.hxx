#ifndef EXAMM_LSTM_NODE_HXX
#define EXAMM_LSTM_NODE_HXX

#include <string>
using std::string;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <vector>
using std::vector;

#include "common/random.hxx"

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

        LSTM_Node(int _innovation_number, int _type, double _depth);
        ~LSTM_Node();

        void initialize_lamarckian(minstd_rand0 &generator, NormalDistribution &normal_distribution, double mu, double sigma);
        void initialize_xavier(minstd_rand0 &generator, uniform_real_distribution<double> &rng1_1, double range);
        void initialize_kaiming(minstd_rand0 &generator, NormalDistribution &normal_distribution, double range);
        void initialize_uniform_random(minstd_rand0 &generator, uniform_real_distribution<double> &rng);

        double get_gradient(string gradient_name);
        void print_gradient(string gradient_name);

        void input_fired(int time, double incoming_output);

        void try_update_deltas(int time);
        void error_fired(int time, double error);
        void output_fired(int time, double delta);

        uint32_t get_number_weights() const;

        void get_weights(vector<double> &parameters) const;
        void set_weights(const vector<double> &parameters);

        void get_weights(uint32_t &offset, vector<double> &parameters) const;
        void set_weights(uint32_t &offset, const vector<double> &parameters);

        void get_gradients(vector<double> &gradients);

        void reset(int _series_length);

        void write_to_stream(ostream &out);

        RNN_Node_Interface* copy() const;

        friend class RNN_Edge;
};
#endif
