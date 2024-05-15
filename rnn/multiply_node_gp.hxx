#ifndef EXAMM_MULTIPLY_NODE_GP_HXX
#define EXAMM_MULTIPLY_NODE_GP_HXX

#include <vector>
using std::vector;

#include "multiply_node.hxx"

class MULTIPLY_Node_GP : public MULTIPLY_Node {
   public:
    // constructor for hidden nodes
    MULTIPLY_Node_GP(int32_t _innovation_number, int32_t _layer_type, double _depth);

    ~MULTIPLY_Node_GP();

    virtual void input_fired(int32_t time, double incoming_output) override;
    virtual void try_update_deltas(int32_t time) override;

    virtual RNN_Node_Interface* copy() const override;
};

#endif
