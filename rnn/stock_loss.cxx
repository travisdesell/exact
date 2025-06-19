#include <cmath>
#include <vector>
using std::vector;

#include "common/log.hxx"
#include "rnn.hxx"

template <typename T>
double signum(T x) {
  if (x > 0) {
    return 1.0;
  } else if (x < 0) {
    return -1.0;
  } else {
    return 0.0;
  }
}

void get_stock_loss (const vector<double>& output_values, const vector<double>& expected, double& loss_sum, vector<double>& deltas,
    const vector<double> return_at_t,
    const vector<double> return_at_t_plus_1)
{
    deltas.assign(expected.size(), 0.0);  
  
    loss_sum = 0.0;
    double sum_v_i = 0.0;

    for (int32_t i = 0; i < (int32_t) expected.size(); i++) {
        sum_v_i += fabs(output_values[i]);
    }

    for (int32_t i = 0; i < (int32_t) expected.size(); i++) {

        double v_i = 1.0 * fabs(output_values[i]) / sum_v_i;
        loss_sum += v_i * (return_at_t_plus_1[i] - return_at_t[i]) * signum(output_values[i]);

    }

    double d_loss = loss_sum;
    for (int32_t i = 0; i < (int32_t) expected.size(); i++) {
        deltas[i] *= d_loss;
    }
}

void get_stock_loss (RNN* genome, const vector<vector<double> >& expected, double& loss_sum, vector<vector<double> >& deltas,
    const vector<vector<double>> return_at_t,
    const vector<vector<double>> return_at_t_plus_1)
{
    deltas.assign(genome->output_nodes.size(), vector<double>(expected[0].size(), 0.0));

    loss_sum = 0.0;
    double sum_v_i = 0.0;
    double loss;

    for (int32_t i = 0; i < (int32_t) genome->output_nodes.size(); i++) {
        for (int32_t j = 0; j < (int32_t) genome->output_nodes[i]->output_values.size(); j++) {
            sum_v_i += fabs(genome->output_nodes[i]->output_values[j]);
        }
    }

    for (int32_t i = 0; i < (int32_t) genome->output_nodes.size(); i++) {

        loss = 0.0;
        double v_i = 0.0;

        for (int32_t j = 0; j < (int32_t) genome->output_nodes[i]->output_values.size(); j++) {
            sum_v_i += fabs(genome->output_nodes[i]->output_values[j]);
        }

        for (int32_t j = 0; j < (int32_t) genome->output_nodes[i]->output_values.size(); j++) {
            v_i = 1.0 * fabs(genome->output_nodes[i]->output_values[j]) / sum_v_i;
            loss_sum += v_i * (return_at_t_plus_1[i][j] - return_at_t[i][j]) * signum(genome->output_nodes[i]->output_values[j]);
        }
    }

    double d_loss = loss_sum * (1.0 / expected[0].size());
    for (int32_t i = 0; i < (int32_t) genome->output_nodes.size(); i++) {
        for (int32_t j = 0; j < (int32_t) expected[i].size(); j++) {
            deltas[i][j] *= d_loss;
        }
    }
}