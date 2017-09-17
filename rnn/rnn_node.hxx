#ifndef EXALT_RNN_NODE_HXX
#define EXALT_RNN_NODE_HXX

#include <vector>
using std::vector;

#include "rnn_node_interface.hxx"

class RNN_Node : public RNN_Node_Interface {
    private:
        double bias;

    public:

        RNN_Node(int _innovation_number, int _type);

        void input_fired();
        void output_fired();
        uint32_t get_number_weights();
        void set_weights(uint32_t &offset, const vector<double> &parameters);

        void reset();
        void full_reset();

        friend class RNN_Edge;
};


#endif
