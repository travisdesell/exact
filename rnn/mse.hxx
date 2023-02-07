#ifndef EXAMM_MSE_HXX
#define EXAMM_MSE_HXX

#include <vector>
using std::vector;

#include "rnn.hxx"

void get_mse(const vector<double> &output_values, const vector<double> &expected, double &mse, vector<double> &deltas);
void get_mse(RNN* genome, const vector< vector<double> > &expected, double &mse, vector< vector<double> > &deltas);

void get_mae(const vector<double> &output_values, const vector<double> &expected, double &mae, vector<double> &deltas);
void get_mae(RNN* genome, const vector< vector<double> > &expected, double &mae, vector< vector<double> > &deltas);


#endif
