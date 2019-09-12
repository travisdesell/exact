#ifndef EXALT_RNN_RECURRENT_EDGE_HXX
#define EXALT_RNN_RECURRENT_EDGE_HXX

class RNN;

#include "rnn_node_interface.hxx"

class RNN_Recurrent_Edge {
    private:
        int32_t innovation_number;
        int32_t series_length;

        //how far in the past to get the value
        int32_t recurrent_depth;

        vector<double> outputs;
        vector<double> deltas;

        double weight;
        double d_weight;

        bool enabled;
        bool forward_reachable;
        bool backward_reachable;

        int32_t input_innovation_number;
        int32_t output_innovation_number;

        RNN_Node_Interface *input_node;
        RNN_Node_Interface *output_node;

    public:
        RNN_Recurrent_Edge(int32_t _innovation_number, int32_t _recurrent_depth, RNN_Node_Interface *_input_node, RNN_Node_Interface *_output_node);

        RNN_Recurrent_Edge(int32_t _innovation_number, int32_t _recurrent_depth, int32_t _input_innovation_number, int32_t _output_innovation_number, const vector<RNN_Node_Interface*> &nodes);

        void reset(int32_t _series_length);

        void first_propagate_forward();
        void first_propagate_backward();
        void propagate_forward(int32_t time);
        void propagate_backward(int32_t time);

        int32_t get_recurrent_depth() const;
        double get_gradient();
        bool is_reachable() const;

        RNN_Recurrent_Edge* copy(const vector<RNN_Node_Interface*> new_nodes);

        int32_t get_innovation_number() const;
        int32_t get_input_innovation_number() const;
        int32_t get_output_innovation_number() const;

        const RNN_Node_Interface* get_input_node() const;
        const RNN_Node_Interface* get_output_node() const;

        void write_to_stream(ostream &out);

        bool equals(RNN_Recurrent_Edge *other) const;

        friend class RNN_Genome;
        friend class RNN;
        friend class EXALT;
};


struct sort_RNN_Recurrent_Edges_by_depth {
    bool operator()(RNN_Recurrent_Edge *n1, RNN_Recurrent_Edge *n2) {
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

struct sort_RNN_Recurrent_Edges_by_output_depth {
    bool operator()(RNN_Recurrent_Edge *n1, RNN_Recurrent_Edge *n2) {
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


struct sort_RNN_Recurrent_Edges_by_innovation {
    bool operator()(RNN_Recurrent_Edge *n1, RNN_Recurrent_Edge *n2) {
        return n1->get_innovation_number() < n2->get_innovation_number();
    }   
};


#endif
