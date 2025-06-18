#ifndef EXAMM_RNN_GENOME_HXX
#define EXAMM_RNN_GENOME_HXX

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "rnn_edge.hxx"
#include "rnn_node_interface.hxx"
#include "rnn_recurrent_edge.hxx"
#include "time_series/time_series.hxx"
// #include "word_series/word_series.hxx"

class RNN {
   private:
    int32_t series_length;

    vector<RNN_Node_Interface*> input_nodes;
    vector<RNN_Node_Interface*> output_nodes;

    vector<RNN_Node_Interface*> nodes;
    vector<RNN_Edge*> edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    vector<string> arguments;

    string loss;

   public:
    RNN(vector<RNN_Node_Interface*>& _nodes, vector<RNN_Edge*>& _edges, const vector<string>& input_parameter_names,
        const vector<string>& output_parameter_names);
    RNN(vector<RNN_Node_Interface*>& _nodes, vector<RNN_Edge*>& _edges, vector<RNN_Recurrent_Edge*>& _recurrent_edges,
        const vector<string>& input_parameter_names, const vector<string>& output_parameter_names, const vector<string>& arguments);
    ~RNN();

    void fix_parameter_orders(
        const vector<string>& input_parameter_names, const vector<string>& output_parameter_names
    );
    void validate_parameters(const vector<string>& input_parameter_names, const vector<string>& output_parameter_names);

    int32_t get_number_nodes();
    int32_t get_number_edges();

    RNN_Node_Interface* get_node(int32_t i);
    RNN_Edge* get_edge(int32_t i);

    void forward_pass(
        const vector<vector<double> >& series_data, bool using_dropout, bool training, double dropout_probability
    );
    void backward_pass(double error, bool using_dropout, bool training, double dropout_probability);

    double calculate_error_softmax(const vector<vector<double> >& expected_outputs);
    double calculate_error_mse(const vector<vector<double> >& expected_outputs);
    double calculate_error_mae(const vector<vector<double> >& expected_outputs);

    // Stock Loss
    double calculate_error_stock_loss(const vector<vector<double> >& return_at_t,
        const vector<vector<double> >& return_at_t_plus_1);

    double prediction_softmax(
        const vector<vector<double> >& series_data, const vector<vector<double> >& expected_outputs, bool using_dropout,
        bool training, double dropout_probability
    );
    double prediction_mse(
        const vector<vector<double> >& series_data, const vector<vector<double> >& expected_outputs, bool using_dropout,
        bool training, double dropout_probability
    );
    double prediction_mae(
        const vector<vector<double> >& series_data, const vector<vector<double> >& expected_outputs, bool using_dropout,
        bool training, double dropout_probability
    );
    double prediction_stock_loss(
        const vector<vector<double> >& series_data, const vector<vector<double> >& expected_outputs, bool using_dropout,
        bool training, double dropout_probability
    );

    vector<double> get_predictions(
        const vector<vector<double> >& series_data, const vector<vector<double> >& expected_outputs, bool usng_dropout,
        double dropout_probability
    );

    void write_predictions(
        string output_filename, const vector<string>& input_parameter_names,
        const vector<string>& output_parameter_names, const vector<vector<double> >& series_data,
        const vector<vector<double> >& expected_outputs, TimeSeriesSets* time_series_sets, bool using_dropout,
        double dropout_probability
    );

    void initialize_randomly();
    void get_weights(vector<double>& parameters);
    void set_weights(const vector<double>& parameters);

    int32_t get_number_weights();

    void get_analytic_gradient( /// gradients
        const vector<double>& test_parameters, const vector<vector<double> >& inputs,
        const vector<vector<double> >& outputs, double& mse, vector<double>& analytic_gradient, bool using_dropout,
        bool training, double dropout_probability
    );
    void get_empirical_gradient(
        const vector<double>& test_parameters, const vector<vector<double> >& inputs,
        const vector<vector<double> >& outputs, double& mae, vector<double>& empirical_gradient, bool using_dropout,
        bool training, double dropout_probability
    );

    string get_loss();

    // RNN* copy();

    friend void get_mse(
        RNN* genome, const vector<vector<double> >& expected, double& mse, vector<vector<double> >& deltas
    );
    friend void get_mae(
        RNN* genome, const vector<vector<double> >& expected, double& mae, vector<vector<double> >& deltas
    );
    // Stock Loss
    friend void get_stock_loss(RNN* genome, const vector<vector<double> >& expected, double& loss, vector<vector<double> >& deltas, const vector<double> return_at_t,
        const vector<double> return_at_t_plus_1);
};

#endif
