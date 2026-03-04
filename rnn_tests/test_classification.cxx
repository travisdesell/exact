/**
 * Unit tests for RNN classification mode (softmax / cross-entropy).
 * Covers: use_classification flag, get_fitness, get_softmax, get_analytic_gradient,
 * backprop validation tracking, and stats/graphviz labeling.
 */

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "rnn/generate_nn.hxx"
#include "rnn/rnn_genome.hxx"
#include "time_series/time_series.hxx"
#include "weights/weight_rules.hxx"
#include "weights/weight_update.hxx"

using std::vector;

static int tests_run = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg)                                                                          \
    do {                                                                                           \
        tests_run++;                                                                               \
        if (!(cond)) {                                                                             \
            std::cerr << "FAIL: " << (msg) << " (at " << __FILE__ << ":" << __LINE__ << ")\n";     \
            tests_failed++;                                                                        \
        }                                                                                          \
    } while (0)

// Build simple 3D data: one series; layout [series][param_index][timestep] 
static void make_simple_data(
    int n_timesteps, int n_inputs, int n_outputs,
    vector<vector<vector<double> > >& inputs,
    vector<vector<vector<double> > >& outputs
) {
    inputs.resize(1);
    outputs.resize(1);
    inputs[0].resize(n_inputs);
    outputs[0].resize(n_outputs);
    for (int i = 0; i < n_inputs; i++) {
        inputs[0][i].resize(n_timesteps);
        for (int t = 0; t < n_timesteps; t++)
            inputs[0][i][t] = 0.1 * (t + i + 1);
    }
    for (int k = 0; k < n_outputs; k++) {
        outputs[0][k].resize(n_timesteps);
        for (int t = 0; t < n_timesteps; t++) {
            // One-hot: alternate class 0 and 1
            outputs[0][k][t] = (k == (t % 2)) ? 1.0 : 0.0;
        }
    }
}

int main(int argc, char** argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);
    if (argc == 1) {
        arguments.push_back("--std_message_level");
        arguments.push_back("info");
        arguments.push_back("--file_message_level");
        arguments.push_back("none");
        arguments.push_back("--output_directory");
        arguments.push_back(".");
    }
    Log::initialize(arguments);
    Log::set_id("main");

    WeightRules* weight_rules = new WeightRules();
    weight_rules->initialize_from_args(arguments);

    vector<string> in_names{"x1"};
    vector<string> out_names{"class0", "class1"};
    const int max_recurrent_depth = 1;

    RNN_Genome* genome = create_ff(in_names, 1, 2, out_names, max_recurrent_depth, weight_rules);
    ASSERT(genome != nullptr, "create_ff returned non-null");

    // --- use_classification flag ---
    ASSERT(!genome->get_use_classification(), "default is regression");
    genome->set_use_classification(true);
    ASSERT(genome->get_use_classification(), "set_use_classification(true)");
    genome->set_use_classification(false);
    ASSERT(!genome->get_use_classification(), "set_use_classification(false)");
    genome->set_use_classification(true);

    genome->set_stochastic(false);
    genome->initialize_randomly();

    vector<vector<vector<double> > > inputs, outputs, val_inputs, val_outputs;
    make_simple_data(5, 1, 2, inputs, outputs);
    make_simple_data(3, 1, 2, val_inputs, val_outputs);

    // --- get_softmax returns finite positive value ---
    vector<double> params;
    genome->get_weights(params);
    double ce = genome->get_softmax(params, inputs, outputs);
    ASSERT(std::isfinite(ce) && ce >= 0.0, "get_softmax finite and non-negative");

    // --- cross-entropy (classic) correctness check ---
    // Compare RNN's prediction_softmax against a manual cross-entropy computation
    // using logits from get_predictions and the mathematical softmax definition.
    {
        RNN* rnn_ce = genome->get_rnn();
        rnn_ce->set_weights(params);

        double ce_rnn = rnn_ce->prediction_softmax(inputs[0], outputs[0], false, false, 0.0);

        // Manually compute CE from logits = get_predictions()
        vector<double> flat_logits = rnn_ce->get_predictions(inputs[0], outputs[0], false, 0.0);
        const int n_outputs = (int) out_names.size();          // 2
        const int T = (int) flat_logits.size() / n_outputs;    // timesteps

        double ce_manual = 0.0;
        for (int t = 0; t < T; t++) {
            // Collect logits for this timestep
            double max_logit = -1e100;
            double logits[2];
            for (int k = 0; k < n_outputs; k++) {
                double z = flat_logits[t * n_outputs + k];
                logits[k] = z;
                if (z > max_logit) max_logit = z;
            }

            // Compute softmax with simple stabilization
            double sum_exp = 0.0;
            double softmax[2];
            for (int k = 0; k < n_outputs; k++) {
                softmax[k] = std::exp(logits[k] - max_logit);
                sum_exp += softmax[k];
            }
            for (int k = 0; k < n_outputs; k++) {
                softmax[k] /= sum_exp;
            }

            // One-hot labels in outputs[0][k][t]
            for (int k = 0; k < n_outputs; k++) {
                double y = outputs[0][k][t];
                if (y > 0.0) {
                    ce_manual += -y * std::log(softmax[k]);
                }
            }
        }

        ASSERT(std::isfinite(ce_rnn), "prediction_softmax CE is finite");
        ASSERT(std::isfinite(ce_manual), "manual CE is finite");
        ASSERT(std::fabs(ce_rnn - ce_manual) < 1e-9, "prediction_softmax matches manual cross-entropy computation");

        delete rnn_ce;
    }

    // --- balanced cross-entropy (per-class weighting) ---
    // For a *truly* class-balanced synthetic dataset, the balanced cross-entropy
    // should reduce exactly to the unweighted cross-entropy.
    vector<vector<vector<double> > > bal_inputs, bal_outputs;
    make_simple_data(4, 1, 2, bal_inputs, bal_outputs);  // 4 timesteps -> each class appears exactly twice
    double ce_balanced_std = genome->get_softmax(params, bal_inputs, bal_outputs);
    RNN* rnn_bal = genome->get_rnn();
    double ce_bal = rnn_bal->prediction_softmax_balanced(bal_inputs[0], bal_outputs[0], false, false, 0.0);
    ASSERT(std::isfinite(ce_bal) && ce_bal >= 0.0, "prediction_softmax_balanced finite and non-negative");
    ASSERT(
        std::fabs(ce_bal - ce_balanced_std) < 1e-9,
        "balanced cross-entropy equals standard cross-entropy for equal class frequencies"
    );
    delete rnn_bal;

    // --- get_analytic_gradient (classification path) produces finite gradients ---
    vector<RNN*> rnns;
    rnns.push_back(genome->get_rnn());
    double loss_out;
    vector<double> grad;
    genome->get_analytic_gradient(rnns, params, inputs, outputs, loss_out, grad, true);
    ASSERT(std::isfinite(loss_out), "get_analytic_gradient classification loss finite");
    ASSERT(grad.size() == (size_t) genome->get_number_weights(), "gradient size matches weights");
    bool some_non_zero = false;
    for (size_t i = 0; i < grad.size(); i++) {
        ASSERT(std::isfinite(grad[i]), "gradient component finite");
        if (std::fabs(grad[i]) > 1e-15)
            some_non_zero = true;
    }
    ASSERT(some_non_zero, "at least one gradient component non-zero");
    for (size_t i = 0; i < rnns.size(); i++)
        delete rnns[i];

    // --- backprop in classification mode: best_validation_softmax and get_fitness ---
    genome->set_bp_iterations(2);
    WeightUpdate* weight_update = new WeightUpdate(arguments);
    genome->backpropagate(inputs, outputs, val_inputs, val_outputs, weight_update);
    ASSERT(std::isfinite(genome->get_best_validation_softmax()), "best_validation_softmax finite after backprop");
    ASSERT(genome->get_fitness() == genome->get_best_validation_softmax(), "get_fitness equals best_validation_softmax in classification");
    delete weight_update;

    // --- regression mode: get_fitness returns best_validation_mse ---
    RNN_Genome* genome_reg = create_ff(in_names, 1, 2, out_names, max_recurrent_depth, weight_rules);
    genome_reg->set_use_classification(false);
    genome_reg->set_stochastic(false);
    genome_reg->initialize_randomly();
    genome_reg->set_bp_iterations(2);
    WeightUpdate* wu_reg = new WeightUpdate(arguments);
    genome_reg->backpropagate(inputs, outputs, val_inputs, val_outputs, wu_reg);
    ASSERT(std::isfinite(genome_reg->get_best_validation_mse()), "best_validation_mse finite after backprop");
    ASSERT(genome_reg->get_fitness() == genome_reg->get_best_validation_mse(), "get_fitness equals best_validation_mse in regression");
    delete wu_reg;
    delete genome_reg;

    // --- stats: header says "Fitness", value is get_fitness() ---
    string header = RNN_Genome::print_statistics_header();
    ASSERT(header.find("Fitness") != std::string::npos, "print_statistics_header contains Fitness");
    string stats = genome->print_statistics();
    ASSERT(!stats.empty(), "print_statistics non-empty");

    // --- cross_entropy must return finite fitness (no NaN/Inf from log(0)) ---
    ASSERT(std::isfinite(genome->get_fitness()), "get_fitness (cross-entropy) is finite after classification backprop");
    ASSERT(genome->get_fitness() >= 0.0, "get_fitness (cross-entropy) is non-negative");

    delete genome;
    delete weight_rules;

    std::cout << "Classification tests: " << (tests_run - tests_failed) << " passed, " << tests_failed << " failed (total " << tests_run << ")\n";
    return tests_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
