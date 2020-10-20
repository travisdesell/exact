#ifndef EXAMM_RNN_NODE_INTERFACE_HXX
#define EXAMM_RNN_NODE_INTERFACE_HXX

#include <cstdint>

#include <fstream>
using std::ostream;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/random.hxx"

class RNN;

#define INPUT_LAYER 0   /**< Specifices the Input Layer of the rnn network. */
#define HIDDEN_LAYER 1  /**< Specifices the Hidden Layer of the rnn network. */
#define OUTPUT_LAYER 2  /**< Specifices the Output Layer of the rnn network. */

extern const int32_t NUMBER_NODE_TYPES;
extern const string NODE_TYPES[];

#define SIMPLE_NODE 0  /**< Specifices the rnn network to use the Simple Node as the memory cell. */
#define JORDAN_NODE 1  /**< Specifices the rnn network to use the Jordan Node as the memory cell. */
#define ELMAN_NODE 2  /**< Specifices the rnn network to use the Elman Node as the memory cell. */
#define UGRNN_NODE 3  /**< Specifices the rnn network to use the UGRNN Node as the memory cell. */
#define MGU_NODE 4  /**< Specifices the rnn network to use the MGU Node as the memory cell. */
#define GRU_NODE 5  /**< Specifices the rnn network to use the GRU Node as the memory cell. */
#define DELTA_NODE 6  /**< Specifices the rnn network to use the Delta Node as the memory cell. */
#define LSTM_NODE 7  /**< Specifices the rnn network to use the LSTM Node as the memory cell. */
#define ENARC_NODE 8  /**< Specifices the rnn network to use the ENARC Node as the memory cell. */
#define ENAS_DAG_NODE 9  /**< Specifices the rnn network to use the ENAS_DAG Node as the memory cell. */
#define RANDOM_DAG_NODE 10  /**< Specifices the rnn network to use the Random_DAG Node as the memory cell. */



/**
 * Gives the sigmoid of the value to apply non-linearity in the network.
 *
 * \param value is the output of the node after multiplying with weights. 
 *
 * \return sigmoid of the value
 */
double sigmoid(double value);

/**
 * Gives the swish  = (identity * sigmoid) of the value to apply non-linearity in the network.
 *
 * \param value is the output of the node after multiplying with weights. 
 *
 * \return swish of the value
 */

double swish(double value);

/**
 * Gives the leakyReLU of the value to apply non-linearity in the network. 
 *
 * \param value is the output of the node after multiplying with weights. 
 *
 * \return LeakyReLU of the value
 */

double leakyReLU(double value);

/**
 * Gives the identity of the value. Linearlty is preserved in this case.
 *
 * \param value is the output of the node after multiplying with weights. 
 *
 * \return value
 */

double identity(double value);


/**
 * Gives the sigmoid derivative of the value used during the backward pass.
 *
 * \param value is the output of the node after multiplying with weights. 
 *
 * \return sigmoid derivative of the value
 */

double sigmoid_derivative(double value);

/**
 * Gives the tanh derivative of the value used during the backward pass.
 *
 * \param value is the output of the node after multiplying with weights. 
 *
 * \return tanh derivative of the value
 */

double tanh_derivative(double value);

/**
 * Gives the swish derivative of the value used during the backward pass.
 *
 * \param input is the input of the node before multiplying with weights. 
 * \param value is the output of the node after multiplying with weights. 
 *
 * \return swish derivative
 */
double swish_derivative(double value, double input);

/**
 * Gives the leakyReLU derivative of the value used during the backward pass.
 *
 * \param input is the output of the node after multiplying with weights. 
 *
 * \return leakyReLU derivative of the input
 */
double leakyReLU_derivative(double input);

/**
 * Gives the derivative of the identity used during the backward pass.
 *
 * \return unit
 */
double identity_derivative();


double bound(double value);

class RNN_Node_Interface {
    protected:
        int32_t innovation_number;
        int32_t layer_type;
        int32_t node_type;

        double depth;

        //this is only used for input and output nodes to track
        //which parameter they are assigned to
        string parameter_name;

        bool enabled;
        bool backward_reachable;
        bool forward_reachable;

        int32_t series_length;

        vector<double> input_values;
        vector<double> output_values;
        vector<double> error_values;
        vector<double> d_input;

        vector<int32_t> inputs_fired;
        vector<int32_t> outputs_fired;
        int32_t total_inputs;
        int32_t total_outputs;
    public:
        //this constructor is for hidden nodes
        RNN_Node_Interface(int32_t _innovation_number, int32_t _layer_type, double _depth);

        //this constructor is for input and output nodes (as they have an associated parameter name
        RNN_Node_Interface(int32_t _innovation_number, int32_t _layer_type, double _depth, string _parameter_name);
        virtual ~RNN_Node_Interface();

        virtual void initialize_lamarckian(minstd_rand0 &generator, NormalDistribution &normal_distribution, double mu, double sigma) = 0;
        virtual void initialize_xavier(minstd_rand0 &generator, uniform_real_distribution<double> &rng_1_1, double range) = 0;
        virtual void initialize_kaiming(minstd_rand0 &generator, NormalDistribution &normal_distribution, double range) = 0;
        virtual void initialize_uniform_random(minstd_rand0 &generator, uniform_real_distribution<double> &rng) = 0;

        virtual void input_fired(int32_t time, double incoming_output) = 0;
        virtual void output_fired(int32_t time, double delta) = 0;
        virtual void error_fired(int32_t time, double error) = 0;

        virtual uint32_t get_number_weights() const = 0;

        virtual void get_weights(vector<double> &parameters) const = 0;
        virtual void set_weights(const vector<double> &parameters) = 0;
        virtual void get_weights(uint32_t &offset, vector<double> &parameters) const = 0;
        virtual void set_weights(uint32_t &offset, const vector<double> &parameters) = 0;
        virtual void reset(int32_t _series_length) = 0;

        virtual void get_gradients(vector<double> &gradients) = 0;

        virtual RNN_Node_Interface* copy() const = 0;

        void write_to_stream(ostream &out);

        int32_t get_node_type() const;
        int32_t get_layer_type() const;
        int32_t get_innovation_number() const;
        int32_t get_total_inputs() const;
        int32_t get_total_outputs() const;

        double get_depth() const;
        bool equals(RNN_Node_Interface *other) const;

        bool is_reachable() const;
        bool is_enabled() const;

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
