#ifndef TEST_EDGE_HXX
#define TEST_EDGE_HXX

class Node;
#include "node.hxx"

class Edge {
    public:
        double weight;
        double next_weight;

        Node* input;
        Node* output;

        Edge(double _weight, Node *_input, Node *_output);
};

#endif
