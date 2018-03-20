#ifndef EXALT_RNN_RECURRENT_EDGE_HXX
#define EXALT_RNN_RECURRENT_EDGE_HXX

class RNN;

#include "rnn_node_interface.hxx"

class RNN_Recurrent_Edge {
    private:
        int innovation_number;
        int series_length;

        vector<double> outputs;
        vector<double> deltas;

        double weight;
        double d_weight;

        int input_innovation_number;
        int output_innovation_number;

        RNN_Node_Interface *input_node;
        RNN_Node_Interface *output_node;

    public:
        RNN_Recurrent_Edge(int _innovation_number, RNN_Node_Interface *_input_node, RNN_Node_Interface *_output_node);

        RNN_Recurrent_Edge(int _innovation_number, int _input_innovation_number, int _output_innovation_number, const vector<RNN_Node_Interface*> &nodes);

        void reset(int _series_length);

        void first_propagate_forward();
        void first_propagate_backward();
        void propagate_forward(int time);
        void propagate_backward(int time);

        double get_gradient();

        RNN_Recurrent_Edge* copy(const vector<RNN_Node_Interface*> new_nodes);

        friend class RNN_Genome;
        friend class RNN;
};

#endif
