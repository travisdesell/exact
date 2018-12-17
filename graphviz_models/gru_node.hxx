#ifndef EXALT_GRU_NODE_HXX
#define EXALT_GRU_NODE_HXX

#include <string>
using std::string;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <vector>
using std::vector;

#include "../common/random.hxx"

#include "rnn_node_interface.hxx"

class GRU_Node : public RNN_Node_Interface {
    private:
        double update_gate_update_weight;
        double update_gate_weight;
        double update_gate_bias;

        double reset_gate_update_weight;
        double reset_gate_weight;
        double reset_gate_bias;

        double memory_gate_update_weight;
        double memory_gate_weight;
        double memory_gate_bias;

        vector<double> update_gate_values;
        vector<double> reset_gate_values;
        vector<double> memory_gate_values;

        vector<double> ld_update_gate;
        vector<double> ld_reset_gate;
        vector<double> ld_memory_gate;

        vector<double> d_prev_out;

        vector<double> d_update_gate_update_weight;
        vector<double> d_update_gate_weight;
        vector<double> d_update_gate_bias;

        vector<double> d_reset_gate_update_weight;
        vector<double> d_reset_gate_weight;
        vector<double> d_reset_gate_bias;

        vector<double> d_memory_gate_update_weight;
        vector<double> d_memory_gate_weight;
        vector<double> d_memory_gate_bias;


    public:

        GRU_Node(int _innovation_number, int _type, double _depth);
        ~GRU_Node();

        void initialize_randomly(minstd_rand0 &generator, NormalDistribution &normal_distribution, double mu, double sigma);

        double get_gradient(string gradient_name);
        void print_gradient(string gradient_name);

        void input_fired(int time, double incoming_output);

        void try_update_deltas(int time);
        void error_fired(int time, double error);
        void output_fired(int time, double delta);

        uint32_t get_number_weights() const;

        void get_weights(vector<double> &parameters) const;
        void set_weights(const vector<double> &parameters);
        double get_bias();

        void get_weights(uint32_t &offset, vector<double> &parameters) const;
        void set_weights(uint32_t &offset, const vector<double> &parameters);

        void get_gradients(vector<double> &gradients);

        void reset(int _series_length);

        void print_cell_values();

        void write_to_stream(ostream &out);

        RNN_Node_Interface* copy() const;

        friend class RNN_Edge;

#ifdef GRU_TEST
        friend int main(int argc, char **argv);
#endif
};
#endif
