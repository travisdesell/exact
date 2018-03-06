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

        RNN_Node(int _innovation_number, int _type);

        void input_fired(const vector<double> &incoming_outputs);

        void try_update_deltas();
        void output_fired(const vector<double> &deltas);
        void output_fired(double error);
        uint32_t get_number_weights();

        void get_weights(uint32_t &offset, vector<double> &parameters);
        void set_weights(uint32_t &offset, const vector<double> &parameters);

        void reset(int _series_length);

        void get_gradients(vector<double> &gradients);

        RNN_Node_Interface* copy();

        friend class RNN_Edge;
};


#endif
