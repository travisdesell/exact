#ifndef EXAMM_INVERSE_NODE_GP_HXX
#define EXAMM_INVERSE_NODE_GP_HXX

#include <vector>
using std::vector;

#include "rnn_node.hxx"

class INVERSE_Node_GP : public RNN_Node {
   public:
    // constructor for hidden nodes
    INVERSE_Node_GP(int32_t _innovation_number, int32_t _layer_type, double _depth);

    ~INVERSE_Node_GP();

    virtual double activation_function(double input) override;
    virtual double derivative_function(double input) override;

    virtual RNN_Node_Interface* copy() const override;
};

#endif
