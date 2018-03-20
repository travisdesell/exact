#include <cmath>

#include <fstream>
using std::ofstream;

#include <iomanip>
using std::setw;

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <thread>
using std::thread;

#include <vector>
using std::vector;


#include "rnn.hxx"
#include "rnn_genome.hxx"

RNN_Genome::RNN_Genome(vector<RNN_Node_Interface*> &_nodes, vector<RNN_Edge*> &_edges) {
    nodes = _nodes;
    edges = _edges;

    //sort nodes by depth
    //sort edges by depth
    sort(edges.begin(), edges.end(), sort_RNN_Edges_by_depth());
}

RNN_Genome::RNN_Genome(vector<RNN_Node_Interface*> &_nodes, vector<RNN_Edge*> &_edges, vector<RNN_Recurrent_Edge*> &_recurrent_edges) {
    nodes = _nodes;
    edges = _edges;
    recurrent_edges = _recurrent_edges;

    //sort nodes by depth
    //sort edges by depth
    sort(edges.begin(), edges.end(), sort_RNN_Edges_by_depth());
}


void RNN_Genome::get_weights(vector<double> &parameters) {
    parameters.resize(get_number_weights());

    uint32_t current = 0;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        nodes[i]->get_weights(current, parameters);
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        parameters[current++] = edges[i]->weight;
    }

    for (uint32_t i = 0; i < recurrent_edges.size(); i++) {
        parameters[current++] = recurrent_edges[i]->weight;
    }


}

void RNN_Genome::set_weights(const vector<double> &parameters) {
    if (parameters.size() != get_number_weights()) {
        cerr << "ERROR! Trying to set weights where the RNN has " << get_number_weights() << " weights, and the parameters vector has << " << parameters.size() << " weights!" << endl;
        exit(1);
    }

    uint32_t current = 0;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        nodes[i]->set_weights(current, parameters);
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        edges[i]->weight = parameters[current++];
    }

    for (uint32_t i = 0; i < recurrent_edges.size(); i++) {
        recurrent_edges[i]->weight = parameters[current++];
    }

}

uint32_t RNN_Genome::get_number_weights() {
    uint32_t number_weights = 0;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        number_weights += nodes[i]->get_number_weights();
    }

    number_weights += edges.size();
    number_weights += recurrent_edges.size();

    return number_weights;
}

void RNN_Genome::initialize_randomly() {
    int number_of_weights = get_number_weights();
    vector<double> parameters(number_of_weights, 0.0);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    minstd_rand0 generator(seed);
    uniform_real_distribution<double> rng(-0.5, 0.5);
    for (uint32_t i = 0; i < parameters.size(); i++) {
        parameters[i] = rng(generator);
    }
    set_weights(parameters);
}


RNN* RNN_Genome::get_rnn() {
    vector<RNN_Node_Interface*> node_copies;
    vector<RNN_Edge*> edge_copies;
    vector<RNN_Recurrent_Edge*> recurrent_edge_copies;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        node_copies.push_back( nodes[i]->copy() );
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        edge_copies.push_back( edges[i]->copy(node_copies) );
    }

    for (uint32_t i = 0; i < recurrent_edges.size(); i++) {
        recurrent_edge_copies.push_back( recurrent_edges[i]->copy(node_copies) );
    }

    return new RNN(node_copies, edge_copies, recurrent_edge_copies);
}

void forward_pass_thread(RNN* rnn, const vector<double> &parameters, const vector< vector<double> > &series_data, const vector< vector<double> > &expected_outputs, uint32_t i, double *mses, bool using_dropout, bool training, double dropout_probability) {
    rnn->set_weights(parameters);
    rnn->forward_pass(series_data, using_dropout, training, dropout_probability);
    mses[i] = rnn->calculate_error_mse(expected_outputs);

    //mses[i] = rnn->calculate_error_mae(expected_outputs);
    //cout << "mse[" << i << "]: " << mse_current << endl;
}

void RNN_Genome::get_analytic_gradient(vector<RNN*> &rnns, const vector<double> &parameters, const vector< vector< vector<double> > > &series_data, const vector< vector< vector<double> > > &expected_outputs, double &mse, vector<double> &analytic_gradient, bool using_dropout, bool training, double dropout_probability) {

    double *mses = new double[rnns.size()];
    double mse_sum = 0.0;
    vector<thread> threads;
    for (uint32_t i = 0; i < rnns.size(); i++) {
        threads.push_back( thread(forward_pass_thread, rnns[i], parameters, series_data[i], expected_outputs[i], i, mses, using_dropout, training, dropout_probability) );
    }

    for (uint32_t i = 0; i < rnns.size(); i++) {
        threads[i].join();
        mse_sum += mses[i];
    }
    delete [] mses;

    for (uint32_t i = 0; i < rnns.size(); i++) {
        double d_mse = mse_sum * (1.0 / expected_outputs[i][0].size()) * 2.0;
        rnns[i]->backward_pass(d_mse, using_dropout, training, dropout_probability);

        //double d_mae = mse_sum * (1.0 / expected_outputs[i][0].size());
        //rnns[i]->backward_pass(d_mae);

    }

    mse = mse_sum;

    vector<double> current_gradients;
    analytic_gradient.assign(parameters.size(), 0.0);
    for (uint32_t k = 0; k < rnns.size(); k++) {

        uint32_t current = 0;
        for (uint32_t i = 0; i < rnns[k]->get_number_nodes(); i++) {
            rnns[k]->get_node(i)->get_gradients(current_gradients);

            for (uint32_t j = 0; j < current_gradients.size(); j++) {
                analytic_gradient[current] += current_gradients[j];
                current++;
            }
        }

        for (uint32_t i = 0; i < rnns[k]->get_number_edges(); i++) {
            analytic_gradient[current] += rnns[k]->get_edge(i)->get_gradient();
            current++;
        }
    }
}


void RNN_Genome::backpropagate(const vector< vector< vector<double> > > &series_data, const vector< vector< vector<double> > > &expected_outputs, int max_iterations, double learning_rate, bool nesterov_momentum, bool adapt_learning_rate, bool reset_weights, bool use_high_norm, double high_threshold, bool use_low_norm, double low_threshold, bool using_dropout, double dropout_probability, string log_filename) {

    int32_t n_series = series_data.size();
    vector<RNN*> rnns;
    for (int32_t i = 0; i < n_series; i++) {
        rnns.push_back( this->get_rnn() );
    }

    vector<double> parameters;
    this->get_weights(parameters);

    int n_parameters = this->get_number_weights();
    vector<double> prev_parameters(n_parameters, 0.0);

    vector<double> prev_velocity(n_parameters, 0.0);
    vector<double> prev_prev_velocity(n_parameters, 0.0);

    vector<double> analytic_gradient;
    vector<double> prev_gradient(n_parameters, 0.0);

    double mu = 0.9;
    double original_learning_rate = learning_rate;

    double prev_mu;
    double prev_norm;
    double prev_learning_rate;
    double prev_mse;
    double mse;

    double parameter_norm = 0.0;
    double velocity_norm = 0.0;
    double norm = 0.0;

    //initialize the initial previous values
    get_analytic_gradient(rnns, parameters, series_data, expected_outputs, mse, analytic_gradient, using_dropout, true, dropout_probability);

    norm = 0.0;
    for (int32_t i = 0; i < parameters.size(); i++) {
        norm += analytic_gradient[i] * analytic_gradient[i];
    }
    norm = sqrt(norm);
    
    ofstream output_log(log_filename);

    bool was_reset = false;
    double reset_count = 0;
    for (uint32_t iteration = 0; iteration < max_iterations; iteration++) {
        prev_mu = mu;
        prev_norm  = norm;
        prev_mse = mse;
        prev_learning_rate = learning_rate;


        prev_gradient = analytic_gradient;

        get_analytic_gradient(rnns, parameters, series_data, expected_outputs, mse, analytic_gradient, using_dropout, true, dropout_probability);

        norm = 0.0;
        velocity_norm = 0.0;
        parameter_norm = 0.0;
        for (int32_t i = 0; i < parameters.size(); i++) {
            norm += analytic_gradient[i] * analytic_gradient[i];
            velocity_norm += prev_velocity[i] * prev_velocity[i];
            parameter_norm += parameters[i] * parameters[i];
        }
        norm = sqrt(norm);
        velocity_norm = sqrt(velocity_norm);

        output_log << iteration
             << " " << mse 
             << " " << norm
             << " " << learning_rate << endl;

        cout << "iteration " << setw(10) << iteration
             << ", mse: " << setw(10) << mse 
             << ", lr: " << setw(10) << learning_rate 
             << ", norm: " << setw(10) << norm
             << ", p_norm: " << setw(10) << parameter_norm
             << ", v_norm: " << setw(10) << velocity_norm;

        if (reset_weights && prev_mse * 1.25 < mse) {
            cout << ", RESETTING WEIGHTS " << reset_count << endl;
            parameters = prev_parameters;
            //prev_velocity = prev_prev_velocity;
            prev_velocity.assign(parameters.size(), 0.0);
            mse = prev_mse;
            mu = prev_mu;
            learning_rate = prev_learning_rate;
            analytic_gradient = prev_gradient;


            //learning_rate *= 0.5;
            //if (learning_rate < 0.0000001) learning_rate = 0.0000001;

            reset_count++;
            if (reset_count > 20) break;

            was_reset = true;
            continue;
        }

        if (was_reset) {
            was_reset = false;
        } else {
            reset_count -= 0.1;
            if (reset_count < 0) reset_count = 0;
            if (adapt_learning_rate) learning_rate = original_learning_rate;
        }


        if (adapt_learning_rate) {
            if (prev_mse > mse) {
                learning_rate *= 1.10;
                if (learning_rate > 1.0) learning_rate = 1.0;

                cout << ", INCREASING LR";
            }
        }

        if (use_high_norm && norm > high_threshold) {
            double high_threshold_norm = high_threshold / norm;

            cout << ", OVER THRESHOLD, multiplier: " << high_threshold_norm;

            for (int32_t i = 0; i < parameters.size(); i++) {
                analytic_gradient[i] = high_threshold_norm * analytic_gradient[i];
            }

            if (adapt_learning_rate) {
                learning_rate *= 0.5;
                if (learning_rate < 0.0000001) learning_rate = 0.0000001;
            }

        } else if (use_low_norm && norm < low_threshold) {
            double low_threshold_norm = low_threshold / norm;
            cout << ", UNDER THRESHOLD, multiplier: " << low_threshold_norm;

            for (int32_t i = 0; i < parameters.size(); i++) {
                analytic_gradient[i] = low_threshold_norm * analytic_gradient[i];
            }

            if (adapt_learning_rate) {
                if (prev_mse * 1.05 < mse) {
                    cout << ", WORSE";
                    learning_rate *= 0.5;
                    if (learning_rate < 0.0000001) learning_rate = 0.0000001;
                }
            }
        }

        if (reset_count > 0) {
            double reset_penalty = pow(5.0, -reset_count);
            cout << ", RESET PENALTY (" << reset_count << "): " << reset_penalty;

            for (int32_t i = 0; i < parameters.size(); i++) {
                analytic_gradient[i] = reset_penalty * analytic_gradient[i];
            }

        }


        cout << endl;

        if (nesterov_momentum) {
            for (int32_t i = 0; i < parameters.size(); i++) {
                prev_parameters[i] = parameters[i];
                prev_prev_velocity[i] = prev_velocity[i];

                double mu_v = prev_velocity[i] * prev_mu;

                prev_velocity[i] = mu_v  - (prev_learning_rate * prev_gradient[i]);
                parameters[i] += mu_v + ((mu + 1) * prev_velocity[i]);
            }
        } else {
            for (int32_t i = 0; i < parameters.size(); i++) {
                prev_parameters[i] = parameters[i];
                prev_gradient[i] = analytic_gradient[i];
                parameters[i] -= learning_rate * analytic_gradient[i];
            }
        }
    }

    RNN *g;
    for (int32_t i = 0; i < n_series; i++) {
        g = rnns.back();
        rnns.pop_back();
        delete g;

    }


    this->set_weights(parameters);
}


void RNN_Genome::backpropagate_stochastic(const vector< vector< vector<double> > > &series_data, const vector< vector< vector<double> > > &expected_outputs, int max_iterations, double learning_rate, bool nesterov_momentum, bool adapt_learning_rate, bool reset_weights, bool use_high_norm, double high_threshold, bool use_low_norm, double low_threshold, bool using_dropout, double dropout_probability, string log_filename) {

    vector<double> parameters;
    this->get_weights(parameters);

    int n_parameters = this->get_number_weights();
    vector<double> prev_parameters(n_parameters, 0.0);

    vector<double> prev_velocity(n_parameters, 0.0);
    vector<double> prev_prev_velocity(n_parameters, 0.0);

    vector<double> analytic_gradient;
    vector<double> prev_gradient(n_parameters, 0.0);

    double mu = 0.9;
    double original_learning_rate = learning_rate;

    int n_series = series_data.size();
    double prev_mu[n_series];
    double prev_norm[n_series];
    double prev_learning_rate[n_series];
    double prev_mse[n_series];
    double mse;

    double norm = 0.0;

    RNN* rnn = get_rnn();
    rnn->set_weights(parameters);

    //initialize the initial previous values
    for (uint32_t i = 0; i < n_series; i++) {
        rnn->get_analytic_gradient(parameters, series_data[i], expected_outputs[i], mse, analytic_gradient, using_dropout, true, dropout_probability);

        norm = 0.0;
        for (int32_t i = 0; i < parameters.size(); i++) {
            norm += analytic_gradient[i] * analytic_gradient[i];
        }
        norm = sqrt(norm);
        prev_mu[i] = mu;
        prev_norm[i] = norm;
        prev_mse[i] = mse;
        prev_learning_rate[i] = learning_rate;
    }

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    minstd_rand0 generator(seed);
    uniform_real_distribution<double> rng(0, 1);

    int random_selection = rng(generator);
    mu = prev_mu[random_selection];
    norm = prev_norm[random_selection];
    mse = prev_mse[random_selection];
    learning_rate = prev_learning_rate[random_selection];

    ofstream output_log(log_filename);

    bool was_reset = false;
    int reset_count = 0;
    for (uint32_t iteration = 0; iteration < max_iterations; iteration++) {
        prev_mu[random_selection] = mu;
        prev_norm[random_selection] = norm;
        prev_mse[random_selection] = mse;
        prev_learning_rate[random_selection] = learning_rate;

        prev_gradient = analytic_gradient;

        if (!was_reset) {
            random_selection = rng(generator) * series_data.size();
        }

        rnn->get_analytic_gradient(parameters, series_data[random_selection], expected_outputs[random_selection], mse, analytic_gradient, using_dropout, true, dropout_probability);

        norm = 0.0;
        for (int32_t i = 0; i < parameters.size(); i++) {
            norm += analytic_gradient[i] * analytic_gradient[i];
        }
        norm = sqrt(norm);

        output_log << iteration
             << " " << mse 
             << " " << norm
             << " " << learning_rate << endl;

        cout << "iteration " << iteration
             << ", series: " << random_selection
             << ", mse: " << mse 
             << ", lr: " << learning_rate 
             << ", norm: " << norm;

        if (reset_weights && prev_mse[random_selection] * 2 < mse) {
            cout << ", RESETTING WEIGHTS" << endl;
            parameters = prev_parameters;
            //prev_velocity = prev_prev_velocity;
            prev_velocity.assign(parameters.size(), 0.0);
            mse = prev_mse[random_selection];
            mu = prev_mu[random_selection];
            learning_rate = prev_learning_rate[random_selection];
            analytic_gradient = prev_gradient;

            random_selection = rng(generator) * series_data.size();

            learning_rate *= 0.5;
            if (learning_rate < 0.0000001) learning_rate = 0.0000001;

            reset_count++;
            if (reset_count > 20) break;

            was_reset = true;
            continue;
        }

        if (was_reset) {
            was_reset = false;
        } else {
            reset_count = 0;
            learning_rate = original_learning_rate;
        }


        if (adapt_learning_rate) {
            if (prev_mse[random_selection] > mse) {
                learning_rate *= 1.10;
                if (learning_rate > 1.0) learning_rate = 1.0;

                cout << ", INCREASING LR";
            }
        }

        if (use_high_norm && norm > high_threshold) {
            double high_threshold_norm = high_threshold / norm;
            cout << ", OVER THRESHOLD, multiplier: " << high_threshold_norm;

            for (int32_t i = 0; i < parameters.size(); i++) {
                analytic_gradient[i] = high_threshold_norm * analytic_gradient[i];
            }

            if (adapt_learning_rate) {
                learning_rate *= 0.5;
                if (learning_rate < 0.0000001) learning_rate = 0.0000001;
            }

        } else if (use_low_norm && norm < low_threshold) {
            double low_threshold_norm = low_threshold / norm;
            cout << ", UNDER THRESHOLD, multiplier: " << low_threshold_norm;


            for (int32_t i = 0; i < parameters.size(); i++) {
                analytic_gradient[i] = low_threshold_norm * analytic_gradient[i];
            }

            if (adapt_learning_rate) {
                if (prev_mse[random_selection] * 1.05 < mse) {
                    cout << ", WORSE";
                    learning_rate *= 0.5;
                    if (learning_rate < 0.0000001) learning_rate = 0.0000001;
                }
            }
        }

        cout << endl;

        if (nesterov_momentum) {
            for (int32_t i = 0; i < parameters.size(); i++) {
                prev_parameters[i] = parameters[i];
                prev_prev_velocity[i] = prev_velocity[i];

                double mu_v = prev_velocity[i] * prev_mu[random_selection];

                prev_velocity[i] = mu_v  - (prev_learning_rate[random_selection] * prev_gradient[i]);
                parameters[i] += mu_v + ((mu + 1) * prev_velocity[i]);
            }
        } else {
            for (int32_t i = 0; i < parameters.size(); i++) {
                prev_parameters[i] = parameters[i];
                prev_gradient[i] = analytic_gradient[i];
                parameters[i] -= learning_rate * analytic_gradient[i];
            }
        }
    }

    this->set_weights(parameters);
}
