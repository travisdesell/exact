#ifndef RNN_BPTT_HXX
#define RNN_BPTT_HXX

#include <vector>
using std::vector;

#include "rnn.hxx"
#include "rnn_node_interface.hxx"
#include "rnn_edge.hxx"
#include "rnn_recurrent_edge.hxx"


class RNN_Genome {
    private:
        vector<RNN_Node_Interface*> nodes;
        vector<RNN_Edge*> edges;
        vector<RNN_Recurrent_Edge*> recurrent_edges;

    public:

        RNN_Genome(vector<RNN_Node_Interface*> &_nodes, vector<RNN_Edge*> &_edges);

        RNN_Genome(vector<RNN_Node_Interface*> &_nodes, vector<RNN_Edge*> &_edges, vector<RNN_Recurrent_Edge*> &_recurrent_edges);

        void get_weights(vector<double> &parameters);
        void set_weights(const vector<double> &parameters);
        uint32_t get_number_weights();
        void initialize_randomly();

        RNN* get_rnn();

        void get_analytic_gradient(vector<RNN*> &rnns, const vector<double> &parameters, const vector< vector< vector<double> > > &series_data, const vector< vector< vector<double> > > &expected_outputs, double &mse, vector<double> &analytic_gradient, bool using_dropout, bool training, double dropout_probability);

        void backpropagate(const vector< vector< vector<double> > > &series_data, const vector< vector< vector<double> > > &expected_outputs, int max_iterations, double learning_rate, bool nesterov_momentum, bool adapt_learning_rate, bool reset_weights, bool use_high_norm, double high_threshold, bool use_low_norm, double low_threshold, bool using_dropout, double dropout_probability, string log_filename);

        void backpropagate_stochastic(const vector< vector< vector<double> > > &series_data, const vector< vector< vector<double> > > &expected_outputs, int max_iterations, double learning_rate, bool nesterov_momentum, bool adapt_learning_rate, bool reset_weights, bool use_high_norm, double high_threshold, bool use_low_norm, double low_threshold, bool using_dropout, double dropout_probability, string log_filename);
};

#endif
