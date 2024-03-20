#ifndef EXAMM_INVERSE_NODE_HXX
#define EXAMM_INVERSE_NODE_HXX

#include <vector>
using std::vector;

#include "rnn_node.hxx"

class INVERSE_Node : public RNN_Node {
   public:
    // constructor for hidden nodes
    INVERSE_Node(int32_t _innovation_number, int32_t _layer_type, double _depth);

    ~INVERSE_Node();

    virtual double activation_function(double input) override;
    virtual double derivative_function(double input) override;

    virtual RNN_Node_Interface* copy() const override;
};

#endif
