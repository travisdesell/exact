#ifndef EXAMM_STOCK_LOSSES_HXX
#define EXAMM_STOCK_LOSSES_HXX

#include <vector>
using std::vector;

#include "rnn.hxx"


void get_stock_loss(const vector<double>& output_values, const vector<double>& expected, double& loss, vector<double>& deltas, const vector<double> return_at_t,
    const vector<double> return_at_t_plus_1);
void get_stock_loss(RNN* genome, const vector<vector<double> >& expected, double& loss, vector<vector<double> >& deltas, const vector<vector<double>> return_at_t,
    const vector<vector<double>> return_at_t_plus_1);

#endif