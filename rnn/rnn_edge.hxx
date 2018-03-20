#ifndef EXALT_RNN_EDGE_HXX
#define EXALT_RNN_EDGE_HXX

#include "rnn_node_interface.hxx"

class RNN_Edge {
    private:
        int innovation_number;

        vector<double> outputs;
        vector<double> deltas;
        vector<bool> dropped_out;

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

        void propagate_forward(int time, bool training, double dropout_probability);
        void propagate_backward(int time, bool training, double dropout_probability);

        double get_gradient() const;
        int get_innovation_number() const;

        const RNN_Node_Interface* get_input_node() const;
        const RNN_Node_Interface* get_output_node() const;

        RNN_Edge* copy(const vector<RNN_Node_Interface*> new_nodes);

        friend class RNN_Genome;
        friend class RNN;
};


struct sort_RNN_Edges_by_depth {
    bool operator()(RNN_Edge *n1, RNN_Edge *n2) {
        if (n1->get_input_node()->get_depth() < n2->get_input_node()->get_depth()) {
            return true;

        } else if (n1->get_input_node()->get_depth() == n2->get_input_node()->get_depth()) {
            //make sure the order of the edges is *always* the same
            //going through the edges in different orders may effect the output
            //of backpropagation
            if (n1->get_innovation_number() < n2->get_innovation_number()) {
                return true;
            } else {
                return false;
            }

        } else {
            return false;
        }
    }   
};

struct sort_RNN_Edges_by_output_depth {
    bool operator()(RNN_Edge *n1, RNN_Edge *n2) {
        if (n1->get_output_node()->get_depth() < n2->get_output_node()->get_depth()) {
            return true;

        } else if (n1->get_output_node()->get_depth() == n2->get_output_node()->get_depth()) {
            //make sure the order of the edges is *always* the same
            //going through the edges in different orders may effect the output
            //of backpropagation
            if (n1->get_innovation_number() < n2->get_innovation_number()) {
                return true;
            } else {
                return false;
            }

        } else {
            return false;
        }
    }   
};


struct sort_RNN_Edges_by_innovation {
    bool operator()(RNN_Edge *n1, RNN_Edge *n2) {
        return n1->get_innovation_number() < n2->get_innovation_number();
    }   
};





#endif
