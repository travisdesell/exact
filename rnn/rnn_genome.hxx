#ifndef EXALT_RNN_GENOME_HXX
#define EXALT_RNN_GENOME_HXX

#include <string>
using std::string;

#include <vector>
using std::vector;

class RNN_Node_Interface;
class RNN_Edge;
class RNN_Genome;

#include "rnn_node_interface.hxx"
#include "rnn_edge.hxx"

class RNN_Genome {
    private:
        vector<RNN_Node_Interface*> input_nodes;
        vector<RNN_Node_Interface*> output_nodes;

        vector<RNN_Node_Interface*> nodes;
        vector<RNN_Edge*> edges;

    public:
        RNN_Genome(vector<RNN_Node_Interface*> &_nodes, vector<RNN_Edge*> &_edges);

        int get_number_nodes();
        int get_number_edges();

        RNN_Node_Interface* get_node(int i);
        RNN_Edge* get_edge(int i);

        void forward_pass(const vector< vector<double> > &series_data);
        void backward_pass(double error);

        double calculate_error_mse(const vector< vector<double> > &expected_outputs);
        double calculate_error_mae(const vector< vector<double> > &expected_outputs);

        double prediction_mse(const vector< vector<double> > &series_data, const vector< vector<double> > &expected_outputs);
        double prediction_mae(const vector< vector<double> > &series_data, const vector< vector<double> > &expected_outputs);

        void initialize_randomly();
        void get_weights(vector<double> &parameters);
        void set_weights(const vector<double> &parameters);

        uint32_t get_number_weights();

        void get_analytic_gradient(const vector<double> &test_parameters, const vector< vector<double> > &inputs, const vector< vector<double> > &outputs, double &mse, vector<double> &analytic_gradient);
        void get_empirical_gradient(const vector<double> &test_parameters, const vector< vector<double> > &inputs, const vector< vector<double> > &outputs, double &mae, vector<double> &empirical_gradient);

        RNN_Genome* copy();

        friend void get_mse(RNN_Genome* genome, const vector< vector<double> > &expected, double &mse, vector< vector<double> > &deltas);
        friend void get_mae(RNN_Genome* genome, const vector< vector<double> > &expected, double &mae, vector< vector<double> > &deltas);


};

#endif
