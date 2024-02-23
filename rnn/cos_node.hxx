#ifndef EXAMM_COS_NODE_HXX
#define EXAMM_COS_NODE_HXX

#include <vector>
using std::vector;

#include "rnn_node.hxx"

class COS_Node : public RNN_Node {
   public:
    // constructor for hidden nodes
    COS_Node(int32_t _innovation_number, int32_t _layer_type, double _depth);

    ~COS_Node();

    virtual double activation_function(double input) override;
    virtual double derivative_function(double input) override;

    virtual RNN_Node_Interface* copy() const override;
};

#endif
