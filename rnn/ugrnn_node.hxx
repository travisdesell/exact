#ifndef EXAMM_UGRNN_NODE_HXX
#define EXAMM_UGRNN_NODE_HXX

#include <string>
using std::string;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <vector>
using std::vector;

#include "common/random.hxx"

#include "rnn_node_interface.hxx"

class UGRNN_Node : public RNN_Node_Interface {
    private:
        double cw;
        double ch;
        double c_bias;

        double gw;
        double gh;
        double g_bias;

        vector<double> d_cw;
        vector<double> d_ch;
        vector<double> d_c_bias;
        vector<double> d_gw;
        vector<double> d_gh;
        vector<double> d_g_bias;

        vector<double> d_h_prev;

        vector<double> c;
        vector<double> ld_c;
        vector<double> g;
        vector<double> ld_g;

    public:

        UGRNN_Node(int _innovation_number, int _type, double _depth);
        ~UGRNN_Node();

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
