#ifndef EXALT_RNN_NODE_HXX
#define EXALT_RNN_NODE_HXX

#include <vector>
using std::vector;

#include "rnn_node_interface.hxx"

class RNN_Node : public RNN_Node_Interface {
    private:
        double bias;
        double d_bias;

        vector<double> ld_output;

    public:

        RNN_Node(int _innovation_number, int _type, double _depth);
        ~RNN_Node();

        void initialize_randomly(minstd_rand0 &generator, NormalDistribution &normal_distribution, double mu, double sigma);

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

        friend class RNN_Edge;
};


#endif
