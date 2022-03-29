#ifndef EXAMM_Online_LSTM_Node_HXX
#define EXAMM_Online_LSTM_Node_HXX

#include <string>
using std::string;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <vector>
using std::vector;

#include "common/random.hxx"

#include "rnn_node_interface.hxx"

class Online_LSTM_Node : public RNN_Node_Interface {
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

        double output_gate_values;
        double input_gate_values;
        double forget_gate_values;
        double cell_values;
        double previous_cell_values;

        double ld_output_gate;
        double ld_input_gate;
        double ld_forget_gate;

        double cell_in_tanh;
        double cell_out_tanh;
        double ld_cell_in;
        double ld_cell_out;

        double d_prev_cell;
        // double d_next_cell;

        double d_output_gate_update_weight;
        double d_output_gate_weight;
        double d_output_gate_bias;

        double d_input_gate_update_weight;
        double d_input_gate_weight;
        double d_input_gate_bias;

        double d_forget_gate_update_weight;
        double d_forget_gate_weight;
        double d_forget_gate_bias;

        double d_cell_weight;
        double d_cell_bias;

        // those 6 parameters override node_interface
        double input_values;
        double output_values;
        double error_values;
        double d_input;

        int32_t inputs_fired;
        int32_t outputs_fired;

    public:

        Online_LSTM_Node(int _innovation_number, int _type, double _depth);
        ~Online_LSTM_Node();

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

        double get_input_value(int32_t time) const;
        double get_output_value(int32_t time) const;
        double get_error_value(int32_t time) const;
        double get_d_input(int32_t time) const;

        int32_t get_input_fired(int32_t time) const;
        int32_t get_output_fired(int32_t time) const;

        void reset(int _series_length);

        void write_to_stream(ostream &out);

        RNN_Node_Interface* copy() const;

        friend class RNN_Edge;
};
#endif
