#ifndef EXALT_RNN_NODE_INTERFACE_HXX
#define EXALT_RNN_NODE_INTERFACE_HXX

#include <cstdint>

#include <vector>
using std::vector;

#define RNN_INPUT_NODE 0
#define RNN_HIDDEN_NODE 1
#define RNN_OUTPUT_NODE 2


double sigmoid(double value);
double sigmoid_derivative(double value);
double tanh_derivative(double value);

void bound_value(double &value);
void bound_value(double min, double max, double &value);

class RNN_Node_Interface {
    protected:
        int innovation_number;
        int type;

        double input_value;
        double output_value;

        int inputs_fired;
        int outputs_fired;
        int total_inputs;
        int total_outputs;
    public:
        RNN_Node_Interface(int _innovation_number, int _type);

        virtual void input_fired() = 0;
        virtual void output_fired() = 0;

        virtual uint32_t get_number_weights() = 0;

        virtual void set_weights(uint32_t &offset, const vector<double> &parameters) = 0;
        virtual void reset() = 0;
        virtual void full_reset() = 0;

        int get_type();
        int get_innovation_number();

    friend class RNN_Edge;
    friend class RNN_Genome;
};


#endif

