#ifndef EXALT_RNN_EDGE_HXX
#define EXALT_RNN_EDGE_HXX

#include "rnn_node_interface.hxx"


class RNN_Edge {
    private:
        int innovation_number;

        double weight;

        int input_innovation_number;
        int output_innovation_number;

        RNN_Node_Interface *input_node;
        RNN_Node_Interface *output_node;

    public:
        RNN_Edge(int _innovation_number, RNN_Node_Interface *_input_node, RNN_Node_Interface *_output_node);

        void propagate_forward();

    friend class RNN_Genome;
};

#endif
