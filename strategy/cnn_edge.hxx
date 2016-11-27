#ifndef CNN_EDGE_H
#define CNN_EDGE_H

#include <fstream>
using std::ofstream;
using std::ifstream;

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
using std::ostream;
using std::istream;

#include <random>
using std::mt19937;
using std::normal_distribution;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "image_tools/image_set.hxx"
#include "cnn_node.hxx"

#define LEARNING_RATE 0.0005
#define WEIGHT_DECAY 0.0005

class CNN_Edge {
    private:
        int innovation_number;

        int input_node_innovation_number;
        int output_node_innovation_number;

        CNN_Node *input_node;
        CNN_Node *output_node;

        int filter_x, filter_y;
        vector< vector<double> > weights;
        vector< vector<double> > previous_velocity;

        bool fixed;
        bool disabled;
        bool reverse_filter_x;
        bool reverse_filter_y;

    public:
        CNN_Edge();

        CNN_Edge(CNN_Node *_input_node, CNN_Node *_output_node, bool _fixed, int _innovation_number);

        CNN_Edge* copy() const;

        ~CNN_Edge();

        bool set_nodes(const vector<CNN_Node*> nodes);
        void initialize_weights(mt19937 &generator);
        void reinitialize(mt19937 &generator);

        void disable();
        void enable();
        bool is_disabled() const;

        bool is_filter_correct() const;

        int get_number_weights() const;

        int get_innovation_number() const;
        int get_input_innovation_number() const;
        int get_output_innovation_number() const;

        bool connects(int n1, int n2) const;

        CNN_Node* get_input_node();

        CNN_Node* get_output_node();

        void print(ostream &out);

        void propagate_forward();
        void propagate_backward(double mu);

        friend ostream &operator<<(ostream &os, const CNN_Edge* flight);
        friend istream &operator>>(istream &is, CNN_Edge* flight);
};

struct sort_CNN_Edges_by_depth {
    bool operator()(CNN_Edge *n1, CNN_Edge *n2) {
        return n1->get_input_node()->get_depth() < n2->get_input_node()->get_depth();
    }   
};


#endif
