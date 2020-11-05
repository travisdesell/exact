#ifndef EXAMM_MSE_HXX
#define EXAMM_MSE_HXX

#include <vector>
using std::vector;

#include<algorithm>
using std::max;

#include "rnn.hxx"

void get_mse(const vector<double> &output_values, const vector<double> &expected, double &mse, vector<double> &deltas);
void get_mse(RNN* genome, const vector< vector<double> > &expected, double &mse_sum, vector< vector<double> > &deltas);

void get_mae(const vector<double> &output_values, const vector<double> &expected, double &mae, vector<double> &deltas);
void get_mae(RNN* genome, const vector< vector<double> > &expected, double &mae_sum, vector< vector<double> > &deltas);

//Have to do this
//void get_se(const vector<double> &output_values, const vector<double> &expected, double &ce, vector<double> &deltas);
void get_se(RNN* genome, const vector< vector<double> > &expected, double &ce_sum, vector< vector<double> > &deltas);

#endif
