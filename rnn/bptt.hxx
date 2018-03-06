#ifndef RNN_BPTT_HXX
#define RNN_BPTT_HXX

#include <vector>
using std::vector;

void backpropagate(RNN_Genome *genome, const vector< vector< vector<double> > > &series_data, const vector< vector< vector<double> > > &expected_outputs, int max_iterations, double learning_rate, bool nesterov_momentum, bool adapt_learning_rate, bool reset_weights, bool use_high_norm, bool use_low_norm, string log_filename);

void backpropagate_stochastic(RNN_Genome *genome, const vector< vector< vector<double> > > &series_data, const vector< vector< vector<double> > > &expected_outputs, int max_iterations, double learning_rate, bool nesterov_momentum, bool adapt_learning_rate, bool reset_weights, bool use_high_norm, bool use_low_norm, string log_filename);

#endif
