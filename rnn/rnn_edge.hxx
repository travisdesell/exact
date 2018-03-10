#ifndef EXALT_RNN_EDGE_HXX
#define EXALT_RNN_EDGE_HXX

#include "rnn_node_interface.hxx"


class RNN_Edge {
    private:
        int innovation_number;

        vector<double> outputs;
        vector<double> deltas;

        double weight;
        double d_weight;

        int input_innovation_number;
        int output_innovation_number;

        RNN_Node_Interface *input_node;
        RNN_Node_Interface *output_node;

    public:
        RNN_Edge(int _innovation_number, RNN_Node_Interface *_input_node, RNN_Node_Interface *_output_node);

        RNN_Edge(int _innovation_number, int _input_innovation_number, int _output_innovation_number, const vector<RNN_Node_Interface*> &nodes);

        void reset(int series_length);

        void propagate_forward(int time);
        void propagate_backward(int time);

        double get_gradient();

        RNN_Edge* copy(const vector<RNN_Node_Interface*> new_nodes);

        friend class RNN_Genome;
};

#endif
