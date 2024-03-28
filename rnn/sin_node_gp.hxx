#ifndef EXAMM_SIN_NODE_GP_HXX
#define EXAMM_SIN_NODE_GP_HXX

#include <vector>
using std::vector;

#include "rnn_node.hxx"

class SIN_Node_GP : public RNN_Node {
   public:
    // constructor for hidden nodes
    SIN_Node_GP(int32_t _innovation_number, int32_t _layer_type, double _depth);

    ~SIN_Node_GP();

    virtual double activation_function(double input) override;
    virtual double derivative_function(double input) override;

    virtual void input_fired(int32_t time, double incoming_output) override;
    virtual void try_update_deltas(int32_t time) override;

    virtual RNN_Node_Interface* copy() const override;
};

#endif
