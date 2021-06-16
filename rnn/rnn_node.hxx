#ifndef EXAMM_RNN_NODE_HXX
#define EXAMM_RNN_NODE_HXX

#include <vector>
using std::vector;

#include "rnn_node_interface.hxx"

class RNN_Node : public RNN_Node_Interface {
    private:
        double bias;
        double d_bias;

        vector<double> ld_output;

    public:

        //constructor for hidden nodes
        RNN_Node(int _innovation_number, int _layer_type, double _depth, int _node_type);

        //constructor for input and output nodes
        RNN_Node(int _innovation_number, int _layer_type, double _depth, int _node_type, string _parameter_name);
        ~RNN_Node();

        void initialize_lamarckian(minstd_rand0 &generator, NormalDistribution &normal_distribution, double mu, double sigma);
        void initialize_xavier(minstd_rand0 &generator, uniform_real_distribution<double> &rng1_1, double range);
        void initialize_kaiming(minstd_rand0 &generator, NormalDistribution &normal_distribution, double range);
        void initialize_uniform_random(minstd_rand0 &generator, uniform_real_distribution<double> &rng);
        
        void input_fired(int time, double incoming_output);

        void try_update_deltas(int time);
        void output_fired(int time, double delta);
        void error_fired(int time, double error);

        uint32_t get_number_weights() const ;
        void get_weights(vector<double> &parameters) const;
        void set_weights(const vector<double> &parameters);
        void get_weights(uint32_t &offset, vector<double> &parameters) const;
        void set_weights(uint32_t &offset, const vector<double> &parameters);

        void reset(int _series_length);

        void get_gradients(vector<double> &gradients);

        RNN_Node_Interface* copy() const;

        void write_to_stream(ostream &out);

        friend class RNN_Edge;
};

#endif
