#include <cmath>

#include <vector>
using std::vector;

#include "rnn.hxx"

void get_mse(const vector<double> &output_values, const vector<double> &expected, double &mse, vector<double> &deltas) {
    deltas.assign(expected.size(), 0.0);

    mse = 0.0;
    double error;

    for (uint32_t j = 0; j < expected.size(); j++) {
        error = output_values[j] - expected[j];
        deltas[j] = error;

        mse += error * error;
    }

    mse /= expected.size();

    double d_mse = mse * (1.0 / expected.size()) * 2.0;
    for (uint32_t j = 0; j < expected.size(); j++) {
        deltas[j] *= d_mse;
    }
}

void get_mse(RNN *genome, const vector< vector<double> > &expected, double &mse_sum, vector< vector<double> > &deltas) {
    deltas.assign(genome->output_nodes.size(), vector<double>(expected[0].size(), 0.0));

    mse_sum = 0.0;
    double mse;
    double error;

    for (uint32_t i = 0; i < genome->output_nodes.size(); i++) {
        mse = 0.0;
        for (uint32_t j = 0; j < expected[i].size(); j++) {
            error = genome->output_nodes[i]->output_values[j] - expected[i][j];
            deltas[i][j] = error;

            mse += error * error;
        }

        mse /= expected[i].size();
        mse_sum += mse;
    }

    double d_mse = mse_sum * (1.0 / expected[0].size()) * 2.0;
    for (uint32_t i = 0; i < genome->output_nodes.size(); i++) {
        for (uint32_t j = 0; j < expected[i].size(); j++) {
            deltas[i][j] *= d_mse;
        }
    }
}

void get_mae(const vector<double> &output_values, const vector<double> &expected, double &mae, vector<double> &deltas) {
    deltas.assign(expected.size(), 0.0);

    mae = 0.0;
    double error;

    for (uint32_t j = 0; j < expected.size(); j++) {
        error = fabs(output_values[j] - expected[j]);
        if (error == 0) {
            deltas[j] = 0;
        } else {
            deltas[j] = (output_values[j] - expected[j]) / error;
        }

        mae += error;
    }

    mae /= expected.size();

    double d_mae = mae * (1.0 / expected.size());
    for (uint32_t j = 0; j < expected.size(); j++) {
        deltas[j] *= d_mae;
    }
}

void get_mae(RNN *genome, const vector< vector<double> > &expected, double &mae_sum, vector< vector<double> > &deltas) {
    deltas.assign(genome->output_nodes.size(), vector<double>(expected[0].size(), 0.0));

    mae_sum = 0.0;
    double mae;
    double error;

    for (uint32_t i = 0; i < genome->output_nodes.size(); i++) {
        mae = 0.0;
        for (uint32_t j = 0; j < expected[i].size(); j++) {
            error = fabs(genome->output_nodes[i]->output_values[j] - expected[i][j]);
            if (error == 0) {
                deltas[i][j] = 0;
            } else {
                deltas[i][j] = (genome->output_nodes[i]->output_values[j] - expected[i][j]) / error;
            }

            mae += error;
        }

        mae /= expected[i].size();
        mae_sum += mae;
    }

    double d_mae = mae_sum * (1.0 / expected[0].size());
    for (uint32_t i = 0; i < genome->output_nodes.size(); i++) {
        for (uint32_t j = 0; j < expected[i].size(); j++) {
            deltas[i][j] *= d_mae;
        }
    }
}
