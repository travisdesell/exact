#ifndef EXAMM_RNN_GENOME_HXX
#define EXAMM_RNN_GENOME_HXX

#include <string>
using std::string;

#include <vector>
using std::vector;


#include "rnn_node_interface.hxx"
#include "rnn_edge.hxx"
#include "rnn_recurrent_edge.hxx"

#include "time_series/time_series.hxx"
#include "word_series/word_series.hxx"

class RNN {
    private:
        int series_length;
        bool use_regression;

        vector<RNN_Node_Interface*> input_nodes;
        vector<RNN_Node_Interface*> output_nodes;

        vector<RNN_Node_Interface*> nodes;
        vector<RNN_Edge*> edges;
        vector<RNN_Recurrent_Edge*> recurrent_edges;

    public:
        RNN(vector<RNN_Node_Interface*> &_nodes, vector<RNN_Edge*> &_edges, const vector<string> &input_parameter_names, const vector<string> &output_parameter_names);
        RNN(vector<RNN_Node_Interface*> &_nodes, vector<RNN_Edge*> &_edges, vector<RNN_Recurrent_Edge*> &_recurrent_edges, const vector<string> &input_parameter_names, const vector<string> &output_parameter_names);
        ~RNN();

        void fix_parameter_orders(const vector<string> &input_parameter_names, const vector<string> &output_parameter_names);
        void validate_parameters(const vector<string> &input_parameter_names, const vector<string> &output_parameter_names);

        int get_number_nodes();
        int get_number_edges();

        RNN_Node_Interface* get_node(int i);
        RNN_Edge* get_edge(int i);

        void forward_pass(const vector< vector<double> > &series_data, bool using_dropout, bool training, double dropout_probability);
        void backward_pass(double error, bool using_dropout, bool training, double dropout_probability);

        double calculate_error_softmax(const vector< vector<double> > &expected_outputs);
        double calculate_error_mse(const vector< vector<double> > &expected_outputs);
        double calculate_error_mae(const vector< vector<double> > &expected_outputs);

        double prediction_softmax(const vector< vector<double> > &series_data, const vector< vector<double> > &expected_outputs, bool using_dropout, bool training, double dropout_probability);
        double prediction_mse(const vector< vector<double> > &series_data, const vector< vector<double> > &expected_outputs, bool using_dropout, bool training, double dropout_probability);
        double prediction_mae(const vector< vector<double> > &series_data, const vector< vector<double> > &expected_outputs, bool using_dropout, bool training, double dropout_probability);


        vector<double> get_predictions(const vector< vector<double> > &series_data, const vector< vector<double> > &expected_outputs, bool usng_dropout, double dropout_probability);

        void write_predictions(string output_filename, const vector<string> &input_parameter_names, const vector<string> &output_parameter_names, const vector< vector<double> > &series_data, const vector< vector<double> > &expected_outputs, TimeSeriesSets *time_series_sets, bool using_dropout, double dropout_probability);
        void write_predictions(string output_filename, const vector<string> &input_parameter_names, const vector<string> &output_parameter_names, const vector< vector<double> > &series_data, const vector< vector<double> > &expected_outputs, Corpus *word_series_sets, bool using_dropout, double dropout_probability);

        void initialize_randomly();
        void get_weights(vector<double> &parameters);
        void set_weights(const vector<double> &parameters);
        void enable_use_regression(bool _use_regression);

        uint32_t get_number_weights();

        void get_analytic_gradient(const vector<double> &test_parameters, const vector< vector<double> > &inputs, const vector< vector<double> > &outputs, double &mse, vector<double> &analytic_gradient, bool using_dropout, bool training, double dropout_probability);
        void get_empirical_gradient(const vector<double> &test_parameters, const vector< vector<double> > &inputs, const vector< vector<double> > &outputs, double &mae, vector<double> &empirical_gradient, bool using_dropout, bool training, double dropout_probability);

        //RNN* copy();

        friend void get_mse(RNN* genome, const vector< vector<double> > &expected, double &mse, vector< vector<double> > &deltas);
        friend void get_mae(RNN* genome, const vector< vector<double> > &expected, double &mae, vector< vector<double> > &deltas);
};

#endif
