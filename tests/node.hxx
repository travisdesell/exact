#ifndef TEST_NODE_HXX
#define TEST_NODE_HXX

#include <vector>
using std::vector;

class Edge;

#include "edge.hxx"

class Node {
    public:
        double out;
        double bias;
        double error;

        double dtotal_dout;
        double dout_dnet;

        vector<Edge*> input_edges;
        vector<Edge*> output_edges;

        Node();
        Node(double _out, double _bias);

        void fire();
        void activation_function();
};

#endif
