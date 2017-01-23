#include "node.hxx"
#include "edge.hxx"

Edge::Edge(double _weight, Node *_input, Node *_output) {
    weight = _weight;
    next_weight = 0.0;

    input = _input;
    input->output_edges.push_back(this);

    output = _output;
    output->input_edges.push_back(this);
}
