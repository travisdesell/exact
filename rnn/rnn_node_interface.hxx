#ifndef EXALT_RNN_NODE_INTERFACE_HXX
#define EXALT_RNN_NODE_INTERFACE_HXX

#include <cstdint>

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <vector>
using std::vector;

#include "common/random.hxx"

class RNN;

#define RNN_INPUT_NODE 0
#define RNN_HIDDEN_NODE 1
#define RNN_OUTPUT_NODE 2

#define LSTM_NODE 0
#define RNN_NODE 1


double sigmoid(double value);
double sigmoid_derivative(double value);
double tanh_derivative(double value);

void bound_value(double &value);
void bound_value(double min, double max, double &value);

class RNN_Node_Interface {
    protected:
        int innovation_number;
        int type;
        int node_type;

        double depth;

        bool enabled;
        bool backward_reachable;
        bool forward_reachable;

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
        RNN_Node_Interface(int _innovation_number, int _type, double _depth);
        virtual ~RNN_Node_Interface();

        virtual void initialize_randomly(minstd_rand0 &generator, NormalDistribution &normal_distribution, double mu, double sigma) = 0;

        virtual void input_fired(int time, double incoming_output) = 0;
        virtual void output_fired(int time, double delta) = 0;
        virtual void error_fired(int time, double error) = 0;

        virtual uint32_t get_number_weights() const = 0;

        virtual void get_weights(vector<double> &parameters) const = 0;
        virtual void set_weights(const vector<double> &parameters) = 0;
        virtual void get_weights(uint32_t &offset, vector<double> &parameters) const = 0;
        virtual void set_weights(uint32_t &offset, const vector<double> &parameters) = 0;
        virtual void reset(int _series_length) = 0;

        virtual void get_gradients(vector<double> &gradients) = 0;

        virtual RNN_Node_Interface* copy() const = 0;

        int get_type() const;
        int get_innovation_number() const;
        double get_depth() const;
        bool equals(RNN_Node_Interface *other) const;

        bool is_reachable() const;

        friend class RNN_Edge;
        friend class RNN_Recurrent_Edge;
        friend class RNN;
        friend class RNN_Genome;

        friend void get_mse(RNN* genome, const vector< vector<double> > &expected, double &mse, vector< vector<double> > &deltas);
        friend void get_mae(RNN* genome, const vector< vector<double> > &expected, double &mae, vector< vector<double> > &deltas);
};


struct sort_RNN_Nodes_by_innovation {
    bool operator()(RNN_Node_Interface *n1, RNN_Node_Interface *n2) {
        return n1->get_innovation_number() < n2->get_innovation_number();
    }   
};


struct sort_RNN_Nodes_by_depth {
    bool operator()(RNN_Node_Interface *n1, RNN_Node_Interface *n2) {
        return n1->get_depth() < n2->get_depth();
    }   
};



#endif

