#include <chrono>

#include <fstream>
using std::getline;
using std::ifstream;

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/arguments.hxx"

#include "rnn/lstm_node.hxx"
#include "rnn/rnn_edge.hxx"
#include "rnn/rnn_genome.hxx"
#include "rnn/rnn_node.hxx"
#include "rnn/rnn_node_interface.hxx"

#include "rnn/generate_nn.hxx"

#include "time_series/time_series.hxx"

#include "gradient_test.hxx"


minstd_rand0 generator;
uniform_real_distribution<double> rng(-0.5, 0.5);

int test_iterations = 1000;

void initialize_generator() {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    //seed = 1337;
    generator = minstd_rand0(seed);
}

void generate_random_vector(int number_parameters, vector<double> &v) {
    v.resize(number_parameters);

    for (uint32_t i = 0; i < number_parameters; i++) {
        v[i] = rng(generator);
    }
}

void gradient_test(string name, RNN_Genome *genome, const vector< vector<double> > &inputs, const vector< vector<double> > &outputs, bool verbose) {
    double analytic_mse, empirical_mse;
    vector<double> parameters;
    vector<double> analytic_gradient, empirical_gradient;

    cout << "testing gradient on '" << name << "' ... ";
    bool failed = false;

    RNN* rnn = genome->get_rnn();


    for (uint32_t i = 0; i < test_iterations; i++) {
        if (verbose) {
            if (i == 0) cout << endl;
            cout << "\tAttempt " << i << endl;
        }

        generate_random_vector(rnn->get_number_weights(), parameters);

        rnn->get_analytic_gradient(parameters, inputs, outputs, analytic_mse, analytic_gradient, false, true, 0.0);
        rnn->get_empirical_gradient(parameters, inputs, outputs, empirical_mse, empirical_gradient, false, true, 0.0);


        for (uint32_t j = 0; j < analytic_gradient.size(); j++) {
            double difference = analytic_gradient[j] - empirical_gradient[j];

            if (verbose) {
                cout << "\t\tanalytic gradient[" << j << "]: " << analytic_gradient[j] << ", empirical gradient[" << j << "]: " << empirical_gradient[j] << ", difference: " << difference << endl;
            }

            if (fabs(difference) > 10e-10) {
                failed = true;
                cout << "FAILED on attempt " << i << "!" << endl;
                cout << "analytic gradient[" << j << "]: " << analytic_gradient[j] << ", empirical gradient[" << j << "]: " << empirical_gradient[j] << ", difference: " << difference << endl;
                //exit(1);
            }
        }
    }

    if (!failed) {
        cout << "SUCCESS!" << endl;
    }
}


