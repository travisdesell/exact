#ifndef EXAMM_RNN_EDGE_HXX
#define EXAMM_RNN_EDGE_HXX

#include "rnn_node_interface.hxx"

class RNN_Edge {
    private:
        int32_t innovation_number;

        vector<double> outputs;
        vector<double> deltas;
        vector<bool> dropped_out;

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
        RNN_Edge(int32_t _innovation_number, RNN_Node_Interface *_input_node, RNN_Node_Interface *_output_node);

        RNN_Edge(int32_t _innovation_number, int32_t _input_innovation_number, int32_t _output_innovation_number, const vector<RNN_Node_Interface*> &nodes);

        RNN_Edge* copy(const vector<RNN_Node_Interface*> new_nodes);

        void reset(int32_t series_length);

        void propagate_forward(int32_t time);
        void propagate_backward(int32_t time);

        void propagate_forward(int32_t time, bool training, double dropout_probability);
        void propagate_backward(int32_t time, bool training, double dropout_probability);

        double get_gradient() const;
        int32_t get_innovation_number() const;
        int32_t get_input_innovation_number() const;
        int32_t get_output_innovation_number() const;

        const RNN_Node_Interface* get_input_node() const;
        const RNN_Node_Interface* get_output_node() const;

        bool is_enabled() const;
        bool is_reachable() const;

        bool equals (RNN_Edge *other) const;

        void write_to_stream(ostream &out);

        friend class RNN_Genome;
        friend class RNN;
        friend class EXAMM;
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
