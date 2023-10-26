#include <cmath>
#include <vector>
using std::vector;

#include "common/log.hxx"
#include "sin_node.hxx"

SIN_Node::SIN_Node(int32_t _innovation_number, int32_t _layer_type, double _depth)
    : RNN_Node(_innovation_number, _layer_type, _depth, SIN_NODE){
    // node type will be simple, jordan or elman
    Log::info("created node: %d, layer type: %d, node type: %d\n", innovation_number, layer_type, node_type);
}

SIN_Node::~SIN_Node() {
}

double SIN_Node::activation_function(double input){
    return sin(input);
}

double SIN_Node::derivative_function(double input){
    return cos(input);
}


