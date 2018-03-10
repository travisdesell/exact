#ifndef EXALT_RNN_NODE_INTERFACE_HXX
#define EXALT_RNN_NODE_INTERFACE_HXX

#include <cstdint>

#include <vector>
using std::vector;

#include "rnn_genome.hxx"

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

        int series_length;

        vector<double> input_values;
        vector<double> output_values;
        vector<double> error_values;
        vector<double> d_input;

        vector<int> inputs_fired;
        vector<int> outputs_fired;
        int total_inputs;
        int total_outputs;
    public:
        RNN_Node_Interface(int _innovation_number, int _type);

        virtual void input_fired(int time, double incoming_output) = 0;
        virtual void output_fired(int time, double delta) = 0;
        virtual void error_fired(int time, double error) = 0;

        virtual uint32_t get_number_weights() = 0;

        virtual void get_weights(uint32_t &offset, vector<double> &parameters) = 0;
        virtual void set_weights(uint32_t &offset, const vector<double> &parameters) = 0;
        virtual void reset(int _series_length) = 0;

        virtual void get_gradients(vector<double> &gradients) = 0;

        virtual RNN_Node_Interface* copy() = 0;

        int get_type();
        int get_innovation_number();

        friend class RNN_Edge;
        friend class RNN_Genome;

        friend void get_mse(RNN_Genome* genome, const vector< vector<double> > &expected, double &mse, vector< vector<double> > &deltas);
        friend void get_mae(RNN_Genome* genome, const vector< vector<double> > &expected, double &mae, vector< vector<double> > &deltas);

};


#endif

