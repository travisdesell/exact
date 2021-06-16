#ifndef EXAMM_GRU_NODE_HXX
#define EXAMM_GRU_NODE_HXX

#include <string>
using std::string;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <vector>
using std::vector;

#include "common/random.hxx"

#include "rnn_node_interface.hxx"

class GRU_Node : public RNN_Node_Interface {
    private:
        double zw;
        double zu;
        double z_bias;

        double rw;
        double ru;
        double r_bias;

        double hw;
        double hu;
        double h_bias;

        vector<double> d_zw;
        vector<double> d_zu;
        vector<double> d_z_bias;
        vector<double> d_rw;
        vector<double> d_ru;
        vector<double> d_r_bias;
        vector<double> d_hw;
        vector<double> d_hu;
        vector<double> d_h_bias;

        vector<double> d_h_prev;

        vector<double> z;
        vector<double> ld_z;
        vector<double> r;
        vector<double> ld_r;
        vector<double> h_tanh;
        vector<double> ld_h_tanh;

    public:

        GRU_Node(int _innovation_number, int _type, double _depth);
        ~GRU_Node();

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
