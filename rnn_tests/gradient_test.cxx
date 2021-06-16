#include <chrono>

#include <fstream>
using std::getline;
using std::ifstream;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"

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

int test_iterations = 10;

void initialize_generator() {
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	//seed = 1337;
	generator = minstd_rand0(seed);
}

void generate_random_vector(int number_parameters, vector<double> &v) {
	v.resize(number_parameters);

	for (int32_t i = 0; i < number_parameters; i++) {
		v[i] = rng(generator);
	}
}

void gradient_test(string name, RNN_Genome *genome, const vector< vector<double> > &inputs, const vector< vector<double> > &outputs) {
	double analytic_mse, empirical_mse;
	vector<double> parameters;
	vector<double> analytic_gradient, empirical_gradient;

    Log::info("\ttesting gradient on '%s'...\n", name.c_str());
	bool failed = false;

	RNN* rnn = genome->get_rnn();
    Log::debug("got genome \n");

    rnn->enable_use_regression(true);

	for (int32_t i = 0; i < test_iterations; i++) {
        if (i == 0) Log::debug_no_header("\n");
        Log::debug("\tAttempt %d USING REGRESSION\n", i);

		generate_random_vector(rnn->get_number_weights(), parameters);
		Log::debug("DEBUG: firing weights are %d \n", rnn->get_number_weights());    

		rnn->get_analytic_gradient(parameters, inputs, outputs, analytic_mse, analytic_gradient, false, true, 0.0);
		rnn->get_empirical_gradient(parameters, inputs, outputs, empirical_mse, empirical_gradient, false, true, 0.0);

        bool iteration_failed = false;

		for (uint32_t j = 0; j < analytic_gradient.size(); j++) {
			double difference = analytic_gradient[j] - empirical_gradient[j];

			if (fabs(difference) > 10e-10) {
				failed = true;
                iteration_failed = true;
                Log::info("\t\tFAILED analytic gradient[%d]: %lf, empirical gradient[%d]: %lf, difference: %lf, REGRESSION\n", j, analytic_gradient[j], j, empirical_gradient[j], difference);
				//exit(1);
			} else {
                Log::debug("\t\tPASSED analytic gradient[%d]: %lf, empirical gradient[%d]: %lf, difference: %lf, REGRESSION\n", j, analytic_gradient[j], j, empirical_gradient[j], difference);
            }
		}

        if (iteration_failed) {
            Log::info("\tITERATION %d FAILED!\n\n", i);
        } else {
            Log::debug("\tITERATION %d PASSED!\n\n", i);
        }
	}

    rnn->enable_use_regression(false);

	for (int32_t i = 0; i < test_iterations; i++) {
        if (i == 0) Log::debug_no_header("\n");
        Log::debug("\tAttempt %d USING SOFTMAX\n", i);

		generate_random_vector(rnn->get_number_weights(), parameters);
		Log::debug("DEBUG: firing weights are %d \n", rnn->get_number_weights());    

		rnn->get_analytic_gradient(parameters, inputs, outputs, analytic_mse, analytic_gradient, false, true, 0.0);
		rnn->get_empirical_gradient(parameters, inputs, outputs, empirical_mse, empirical_gradient, false, true, 0.0);

        bool iteration_failed = false;

		for (uint32_t j = 0; j < analytic_gradient.size(); j++) {
			double difference = analytic_gradient[j] - empirical_gradient[j];

			if (fabs(difference) > 10e-10) {
				failed = true;
                iteration_failed = true;
                Log::info("\t\tFAILED analytic gradient[%d]: %lf, empirical gradient[%d]: %lf, difference: %lf, SOFTMAX\n", j, analytic_gradient[j], j, empirical_gradient[j], difference);
				//exit(1);
			} else {
                Log::debug("\t\tPASSED analytic gradient[%d]: %lf, empirical gradient[%d]: %lf, difference: %lf, SOFTMAX\n", j, analytic_gradient[j], j, empirical_gradient[j], difference);
            }
		}

        if (iteration_failed) {
            Log::info("\tITERATION %d FAILED!\n\n", i);
        } else {
            Log::debug("\tITERATION %d PASSED!\n\n", i);
        }
	}

    delete rnn;

	if (!failed) {
        Log::info("ALL PASSED!\n");
	} else {
        Log::info("SOME FAILED!\n");
    }
}
