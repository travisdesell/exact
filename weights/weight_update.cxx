#include "weight_update.hxx"

#include <cmath>

#include "common/arguments.hxx"
#include "common/log.hxx"
#define EXAMM_MAX_DOUBLE 10000000

WeightUpdate::WeightUpdate() {
    // By default use ADAM weight update
    momentum = 0.9;
    weight_update_method = ADAM;
    epsilon = 1e-8;
    decay_rate = 0.9;
    beta1 = 0.9;
    beta2 = 0.99;

    learning_rate = 0.001;
    high_threshold = 1.0;
    low_threshold = 0.05;
    use_high_norm = true;
    use_low_norm = true;

    int32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator = minstd_rand0(seed);
    rng_0_1 = uniform_real_distribution<double>(0.0, 1.0);

    // Set the ranges of the SHO tuned hyperparameters
    learning_rate_min = 0.00001;
    learning_rate_max = 0.3;

    epsilon_min = 1e-9;
    epsilon_max = 1e-7;
    
    beta1_min = 0.88;
    beta1_max = 0.93;
    
    beta2_min = 0.95;
    beta2_max = 0.999;

    // Set the endpoints of the distribution from which we have to set the initial SHO hyperparameters
    rng_ilr_min = uniform_real_distribution<double>(0.001, 0.01);
    rng_ilr_max = uniform_real_distribution<double>(0.011, 0.05);

    rng_ieps_min = uniform_real_distribution<double>(1e-9, 1e-8);
    rng_ieps_max = uniform_real_distribution<double>(1e-8, 1e-7);

    rng_ib1_min = uniform_real_distribution<double>(0.88, 0.905);
    rng_ib1_max = uniform_real_distribution<double>(0.906, 0.93);

    rng_ib2_min = uniform_real_distribution<double>(0.95, 0.9745);
    rng_ib2_max = uniform_real_distribution<double>(0.9746, 0.999);

    // Set the values of the initial SHO hyperparameters from the above set distributions
    initial_learning_rate_min = rng_ilr_min(generator);
    initial_learning_rate_max = rng_ilr_max(generator);

    initial_epsilon_min = rng_ieps_min(generator);
    initial_epsilon_max = rng_ieps_max(generator);

    initial_beta1_min = rng_ib1_min(generator);
    initial_beta1_max = rng_ib1_max(generator);

    initial_beta2_min = rng_ib2_min(generator);
    initial_beta2_max = rng_ib2_max(generator);
}

bool WeightUpdate::use_SHO = false;

WeightUpdate::WeightUpdate(const vector<string>& arguments) : WeightUpdate() {
    generate_from_arguments(arguments);
}

void WeightUpdate::generate_from_arguments(const vector<string>& arguments) {
    Log::info("In weight_update.cxx, checking if Simplex method (SHO) is to be used\n");
    if (argument_exists(arguments, "--use_SHO")) {
        WeightUpdate::use_SHO = true;
        Log::debug("AT: SHO is used for this execution");
    } else {
        Log::debug("AT: SHO is not used for this execution");
    } 

    Log::info("Getting infomation on weight update methods for backprop\n");
    if (argument_exists(arguments, "--weight_update")) {
        string weight_update_method_string;
        get_argument(arguments, "--weight_update", true, weight_update_method_string);
        weight_update_method = get_enum_method_from_string(weight_update_method_string);
        Log::info(
            "Doing backprop with weight update method: %s\n", WEIGHT_UPDATE_METHOD_STRING[weight_update_method].c_str()
        );
        if (weight_update_method == MOMENTUM) {
            get_argument(arguments, "--mu", false, momentum);
            Log::info("Momentum weight update mu=%f\n", momentum);
        } else if (weight_update_method == NESTEROV) {
            get_argument(arguments, "--mu", false, momentum);
            Log::info("Nesterov weight update mu=%f\n", momentum);
        } else if (weight_update_method == ADAGRAD) {
            get_argument(arguments, "--eps", false, epsilon);
            Log::info("Adagrad weight update eps=%f\n", epsilon);
        } else if (weight_update_method == RMSPROP) {
            get_argument(arguments, "--eps", false, epsilon);
            get_argument(arguments, "--decay_rate", false, decay_rate);
            Log::info("RMSProp weight update eps=%f, decay_rate=%f\n", epsilon, decay_rate);
        } else if (weight_update_method == ADAM) {
            get_argument(arguments, "--eps", false, epsilon);
            get_argument(arguments, "--beta1", false, beta1);
            get_argument(arguments, "--beta2", false, beta2);
            Log::info("Adam weight update eps=%f, beta1=%f, beta2=%f\n", epsilon, beta1, beta2);
        } else if (weight_update_method == ADAM_BIAS) {
            get_argument(arguments, "--eps", false, epsilon);
            get_argument(arguments, "--beta1", false, beta1);
            get_argument(arguments, "--beta2", false, beta2);
            Log::info("Adam-bias weight update eps=%f, beta1=%f, beta2=%f\n", epsilon, beta1, beta2);
        }
    } else {
        Log::info(
            "Backprop weight update method not set, using default method %s and default parameters\n",
            WEIGHT_UPDATE_METHOD_STRING[weight_update_method].c_str()
        );
    }

    get_argument(arguments, "--learning_rate", false, learning_rate);
    get_argument(arguments, "--high_threshold", false, high_threshold);
    get_argument(arguments, "--low_threshold", false, low_threshold);
    Log::info("Backprop learning rate: %f\n", learning_rate);
    Log::info("Use high norm is set to %s, high norm is %f\n", use_high_norm ? "True" : "False", high_threshold);
    Log::info("Use low norm is set to %s, low norm is %f\n", use_low_norm ? "True" : "False", low_threshold);
}

void WeightUpdate::update_weights(
    vector<double>& parameters, vector<double>& velocity, vector<double>& prev_velocity, vector<double>& gradient,
    int32_t epoch, double _learning_rate, double _epsilon, double _beta1, double _beta2
) {
    if (weight_update_method == VANILLA) {
        vanilla_weight_update(parameters, velocity, prev_velocity, gradient, epoch, _learning_rate);
    } else if (weight_update_method == MOMENTUM) {
        momentum_weight_update(parameters, velocity, prev_velocity, gradient, epoch, _learning_rate);
    } else if (weight_update_method == NESTEROV) {
        nesterov_weight_update(parameters, velocity, prev_velocity, gradient, epoch, _learning_rate);
    } else if (weight_update_method == ADAGRAD) {
        adagrad_weight_update(parameters, velocity, prev_velocity, gradient, epoch, _learning_rate);
    } else if (weight_update_method == RMSPROP) {
        rmsprop_weight_update(parameters, velocity, prev_velocity, gradient, epoch, _learning_rate);
    } else if (weight_update_method == ADAM) {
        adam_weight_update(parameters, velocity, prev_velocity, gradient, epoch, _learning_rate, _epsilon, _beta1, _beta2);
    } else if (weight_update_method == ADAM_BIAS) {
        adam_bias_weight_update(parameters, velocity, prev_velocity, gradient, epoch, _learning_rate);
    } else {
        Log::fatal(
            "Unrecognized weight update method's enom number: %d, this should never happen!\n", weight_update_method
        );
        exit(1);
    }
}

void WeightUpdate::vanilla_weight_update(
    vector<double>& parameters, vector<double>& velocity, vector<double>& prev_velocity, vector<double>& gradient,
    int32_t epoch, double _learning_rate
) {
    Log::trace("Doing weight update with method: %s \n", WEIGHT_UPDATE_METHOD_STRING[weight_update_method].c_str());
    for (int32_t i = 0; i < (int32_t) parameters.size(); i++) {
        Log::debug("AT: SHO is used = %s\n",WeightUpdate::use_SHO?"true":"false");
        if (WeightUpdate::use_SHO) {
            parameters[i] -= _learning_rate * gradient[i];
        } else {
            parameters[i] -= learning_rate * gradient[i];
        }
        gradient_clip(parameters[i]);
    }
}

void WeightUpdate::momentum_weight_update(
    vector<double>& parameters, vector<double>& velocity, vector<double>& prev_velocity, vector<double>& gradient,
    int32_t epoch, double _learning_rate
) {
    Log::trace("Doing weight update with method: %s \n", WEIGHT_UPDATE_METHOD_STRING[weight_update_method].c_str());
    for (int32_t i = 0; i < (int32_t) parameters.size(); i++) {
        Log::debug("AT: SHO is used = %s\n",WeightUpdate::use_SHO?"true":"false");
        if (WeightUpdate::use_SHO) {
            velocity[i] = momentum * velocity[i] - _learning_rate * gradient[i];    
        } else {
            velocity[i] = momentum * velocity[i] - learning_rate * gradient[i];
        }
        parameters[i] += velocity[i];
        gradient_clip(parameters[i]);
    }
}

void WeightUpdate::nesterov_weight_update(
    vector<double>& parameters, vector<double>& velocity, vector<double>& prev_velocity, vector<double>& gradient,
    int32_t epoch, double _learning_rate
) {
    Log::info("Doing weight update with method: %s \n", WEIGHT_UPDATE_METHOD_STRING[weight_update_method].c_str());
    for (int32_t i = 0; i < (int32_t) parameters.size(); i++) {
        prev_velocity[i] = velocity[i];
        Log::debug("AT: SHO is used = %s\n",WeightUpdate::use_SHO?"true":"false");
        if (WeightUpdate::use_SHO) {
            velocity[i] = momentum * velocity[i] - _learning_rate * gradient[i];
        } else {
            velocity[i] = momentum * velocity[i] - learning_rate * gradient[i];
        }
        parameters[i] += -momentum * prev_velocity[i] + (1 + momentum) * velocity[i];
        gradient_clip(parameters[i]);
    }
}

void WeightUpdate::adagrad_weight_update(
    vector<double>& parameters, vector<double>& velocity, vector<double>& prev_velocity, vector<double>& gradient,
    int32_t epoch, double _learning_rate
) {
    Log::trace("Doing weight update with method: %s \n", WEIGHT_UPDATE_METHOD_STRING[weight_update_method].c_str());
    for (int32_t i = 0; i < (int32_t) parameters.size(); i++) {
        // here the velocity is the "cache" in Adagrad
        velocity[i] += gradient[i] * gradient[i];
        Log::debug("AT: SHO is used = %s\n",WeightUpdate::use_SHO?"true":"false");
        if (WeightUpdate::use_SHO) {
            parameters[i] += -_learning_rate * gradient[i] / (sqrt(velocity[i]) + epsilon);
        } else {
            parameters[i] += -learning_rate * gradient[i] / (sqrt(velocity[i]) + epsilon);
        }
        gradient_clip(parameters[i]);
    }
}

void WeightUpdate::rmsprop_weight_update(
    vector<double>& parameters, vector<double>& velocity, vector<double>& prev_velocity, vector<double>& gradient,
    int32_t epoch, double _learning_rate
) {
    Log::trace("Doing weight update with method: %s \n", WEIGHT_UPDATE_METHOD_STRING[weight_update_method].c_str());
    for (int32_t i = 0; i < (int32_t) parameters.size(); i++) {
        // here the velocity is the "cache" in RMSProp
        velocity[i] = decay_rate * velocity[i] + (1 - decay_rate) * gradient[i] * gradient[i];
        Log::debug("AT: SHO is used = %s\n",WeightUpdate::use_SHO?"true":"false");
        if (WeightUpdate::use_SHO) {
            parameters[i] += -_learning_rate * gradient[i] / (sqrt(velocity[i]) + epsilon);
        } else {
            parameters[i] += -learning_rate * gradient[i] / (sqrt(velocity[i]) + epsilon);
        }
        gradient_clip(parameters[i]);
    }
}

void WeightUpdate::adam_weight_update(
    vector<double>& parameters, vector<double>& velocity, vector<double>& prev_velocity, vector<double>& gradient,
    int32_t epoch, double _learning_rate, double _epsilon, double _beta1, double _beta2
) {
    Log::trace("Doing weight update with method: %s \n", WEIGHT_UPDATE_METHOD_STRING[weight_update_method].c_str());
    for (int32_t i = 0; i < (int32_t) parameters.size(); i++) {
        // here the velocity is the "v" in adam, the prev_velocity is "m" in adam
        Log::debug("AT: SHO is used = %s\n",WeightUpdate::use_SHO?"true":"false");
        if (WeightUpdate::use_SHO) {
            prev_velocity[i] = _beta1 * prev_velocity[i] + (1 - _beta1) * gradient[i];
            velocity[i] = _beta2 * velocity[i] + (1 - _beta2) * (gradient[i] * gradient[i]);
            parameters[i] += -_learning_rate * prev_velocity[i] / (sqrt(velocity[i]) + _epsilon);
        } else {
            prev_velocity[i] = beta1 * prev_velocity[i] + (1 - beta1) * gradient[i];
            velocity[i] = beta2 * velocity[i] + (1 - beta2) * (gradient[i] * gradient[i]);
            parameters[i] += -learning_rate * prev_velocity[i] / (sqrt(velocity[i]) + epsilon);
        }
        gradient_clip(parameters[i]);
    }
}

void WeightUpdate::adam_bias_weight_update(
    vector<double>& parameters, vector<double>& velocity, vector<double>& prev_velocity, vector<double>& gradient,
    int32_t epoch, double _learning_rate
) {
    Log::trace("Doing weight update with method: %s \n", WEIGHT_UPDATE_METHOD_STRING[weight_update_method].c_str());
    for (int32_t i = 0; i < (int32_t) parameters.size(); i++) {
        // here the velocity is the "v" in adam, the prev_velocity is "m" in adam
        prev_velocity[i] = beta1 * prev_velocity[i] + (1 - beta1) * gradient[i];
        double mt = prev_velocity[i] / (1 - pow(beta1, epoch));
        velocity[i] = beta2 * velocity[i] + (1 - beta2) * (gradient[i] * gradient[i]);
        double vt = velocity[i] / (1 - pow(beta2, epoch));
        Log::debug("AT: SHO is used = %s\n",WeightUpdate::use_SHO?"true":"false");
        if (WeightUpdate::use_SHO) {
            parameters[i] += -_learning_rate * mt / (sqrt(vt) + epsilon);
        } else {
            parameters[i] += -learning_rate * mt / (sqrt(vt) + epsilon);
        }        
        gradient_clip(parameters[i]);
    }
}

void WeightUpdate::gradient_clip(double& parameter) {
    if (parameter < -10.0) {
        parameter = -10.0;
    } else if (parameter > 10.0) {
        parameter = 10.0;
    }
}

// Definition of Getters for SHO tuned hyperparameters
double WeightUpdate::get_learning_rate() {
    return learning_rate;
}

double WeightUpdate::get_epsilon() {
    return epsilon;
}

double WeightUpdate::get_beta1() {
    return beta1;
}

double WeightUpdate::get_beta2() {
    return beta2;
}

double WeightUpdate::get_low_threshold() {
    return low_threshold;
}

double WeightUpdate::get_high_threshold() {
    return high_threshold;
}

// Definition of Setters for SHO tuned hyperparameters
void WeightUpdate::set_learning_rate(double _learning_rate) {
    learning_rate = _learning_rate;
}

void WeightUpdate::set_epsilon(double _epsilon) {
    epsilon = _epsilon;
}

void WeightUpdate::set_beta1(double _beta1) {
    beta1 = _beta1;
}

void WeightUpdate::set_beta2(double _beta2) {
    beta2 = _beta2;
}

void WeightUpdate::disable_high_threshold() {
    use_high_norm = false;
}

void WeightUpdate::enable_high_threshold(double _high_threshold) {
    use_high_norm = true;
    high_threshold = _high_threshold;
}

void WeightUpdate::disable_low_threshold() {
    use_low_norm = false;
}

void WeightUpdate::enable_low_threshold(double _low_threshold) {
    use_low_norm = true;
    low_threshold = _low_threshold;
}

double WeightUpdate::get_norm(vector<double>& analytic_gradient) {
    double norm = 0.0;
    for (int32_t i = 0; i < (int32_t) analytic_gradient.size(); i++) {
        norm += analytic_gradient[i] * analytic_gradient[i];
    }
    norm = sqrt(norm);

    return norm;
}

void WeightUpdate::norm_gradients(vector<double>& analytic_gradient, double norm) {
    if (use_high_norm && norm > high_threshold) {
        double high_threshold_norm = high_threshold / norm;
        Log::debug_no_header(", OVER THRESHOLD, multiplier: %lf", high_threshold_norm);

        for (int32_t i = 0; i < (int32_t) analytic_gradient.size(); i++) {
            analytic_gradient[i] = high_threshold_norm * analytic_gradient[i];
        }

    } else if (use_low_norm && norm < low_threshold) {
        double low_threshold_norm = low_threshold / norm;
        Log::debug_no_header(", UNDER THRESHOLD, multiplier: %lf", low_threshold_norm);

        for (int32_t i = 0; i < (int32_t) analytic_gradient.size(); i++) {
            analytic_gradient[i] = low_threshold_norm * analytic_gradient[i];
        }
    }
}


// Definition of functions for generating SHO tuned hyperparameters

double WeightUpdate::generate_simplex_learning_rate(vector<vector<double> > genome_information, int simplex_count) {

    stringstream ss;
    for (int i = 0; i < genome_information.size(); i++) {
        for (int j = 0; j < genome_information[i].size(); j++) {
            ss << genome_information[i][j] << " ";
        }
        ss << endl;
    }
    string genome_informationString = ss.str();

    Log::debug("AT: genome_information = %s\n",genome_informationString.c_str());
    
    double tuned_learning_rate;
    double best_learning_rate;
    double avg_learning_rate = 0;

    double best_fitness = EXAMM_MAX_DOUBLE;
    Log::debug("AT: Simplex Count for Learning Rate = %d\n",simplex_count);

    for(int i=0; i<simplex_count;i++) {
        
        Log::debug("AT: After current genome\n");
        Log::debug("AT: Best Fitness Before = %lg\n",best_fitness);
        
        if( i == 0 || genome_information[i][1] < best_fitness) {
            best_fitness = genome_information[i][1];
            Log::debug("AT: Best Fitness after = %lg\n",best_fitness);
            Log::debug("AT: Inside if statement after rnn \n");
            best_learning_rate = genome_information[i][0];
            Log::debug("AT: Best learning rate = %lg\n",best_learning_rate);
        }
        
        Log::debug("AT: Before Average learning rate\n");
        avg_learning_rate += genome_information[i][0];
        Log::debug("AT: Avg Learning Rate = %lg\n",avg_learning_rate);
        Log::debug("AT: After Average learning rate\n");

    }
    
    Log::debug("AT: avg Learning Rate before division = %lg\n",avg_learning_rate);
    avg_learning_rate = avg_learning_rate /  simplex_count;
    Log::debug("AT: avg Learning Rate after division = %lg\n",avg_learning_rate);
    
    double scale = (rng_0_1(generator) * 2.0) - 0.5;
    Log::debug("AT: avg_learning_rate after adding avg and mul = %lg\n",avg_learning_rate);
    Log::debug("AT: best_learning_rate after adding avg and mul = %lg\n",best_learning_rate);
    Log::debug("AT: scale after adding avg and mul = %lg\n",scale);

    tuned_learning_rate = avg_learning_rate + ((best_learning_rate - avg_learning_rate) * scale);
    
    Log::debug("AT: learning rate after adding avg and mul = %lg\n",tuned_learning_rate);
    if (tuned_learning_rate < learning_rate_min) tuned_learning_rate = learning_rate_min;
    if (tuned_learning_rate > learning_rate_max) tuned_learning_rate = learning_rate_max;
    Log::debug("AT: learning rate finally = %lg\n",tuned_learning_rate);
    
    return tuned_learning_rate;
}

double WeightUpdate::generate_simplex_epsilon(vector<vector<double> > genome_information, int simplex_count) {

    stringstream ss;
    for (int i = 0; i < genome_information.size(); i++) {
        for (int j = 0; j < genome_information[i].size(); j++) {
            ss << genome_information[i][j] << " ";
        }
        ss << endl;
    }
    string genome_informationString = ss.str();

    Log::debug("AT: genome_information = %s\n",genome_informationString.c_str());

    double tuned_epsilon;
    double best_epsilon;
    double avg_epsilon = 0;

    double best_fitness = EXAMM_MAX_DOUBLE;
    Log::debug("AT: Simplex Count for Epsilon = %d\n",simplex_count);

    for(int i=0; i<simplex_count;i++) {
    
        Log::debug("AT: After current genome\n");
        Log::debug("AT: Best Fitness Before = %lg\n",best_fitness);
    
        if( i == 0 || genome_information[i][1] < best_fitness) {
            best_fitness = genome_information[i][1];
            Log::debug("AT: Best Fitness after = %lg\n",best_fitness);
            Log::debug("AT: Inside if statement after rnn genome\n");
            
            best_epsilon = genome_information[i][2];
            Log::debug("AT: Best Epsilon = %lg\n",best_epsilon);
            
        }

        Log::debug("AT: Before Average Epsilon\n");
        avg_epsilon += genome_information[i][2];
        Log::debug("AT: Avg Epsilon = %lg\n",avg_epsilon);
        Log::debug("AT: After Average Epsilon\n");

    }
    
    Log::debug("AT: avg Epsilon before division = %lg\n",avg_epsilon);
    avg_epsilon = avg_epsilon /  simplex_count;
    Log::debug("AT: avg Epsilon after division = %lg\n",avg_epsilon);
    
    double scale = (rng_0_1(generator) * 2.0) - 0.5;
    Log::debug("AT: avg_epsilon after adding avg and mul = %lg\n",avg_epsilon);
    Log::debug("AT: best_epsilon after adding avg and mul = %lg\n",best_epsilon);
    Log::debug("AT: scale after adding avg and mul = %lg\n",scale);

    tuned_epsilon = avg_epsilon + ((best_epsilon - avg_epsilon) * scale);

    Log::debug("AT: Epsilon after adding avg and mul = %lg\n",tuned_epsilon);
    if (tuned_epsilon < epsilon_min) tuned_epsilon = epsilon_min;
    if (tuned_epsilon > epsilon_max) tuned_epsilon = epsilon_max;
    Log::debug("AT: Epsilon finally = %lg\n",tuned_epsilon);
    
    return tuned_epsilon;
}


double WeightUpdate::generate_simplex_beta1(vector<vector<double> > genome_information, int simplex_count) {

    stringstream ss;
    for (int i = 0; i < genome_information.size(); i++) {
        for (int j = 0; j < genome_information[i].size(); j++) {
            ss << genome_information[i][j] << " ";
        }
        ss << endl;
    }
    string genome_informationString = ss.str();

    Log::debug("AT: genome_information = %s\n",genome_informationString.c_str());

    double tuned_beta1;
    double best_beta1;
    double avg_beta1 = 0;

    double best_fitness = EXAMM_MAX_DOUBLE;
    Log::debug("AT: Simplex Count for Beta1 = %d\n",simplex_count);

    for(int i=0; i<simplex_count;i++) {
    
        Log::debug("AT: After current genome\n");
        Log::debug("AT: Best Fitness Before = %lg\n",best_fitness);
    
        if( i == 0 || genome_information[i][1] < best_fitness) {
            best_fitness = genome_information[i][1];
            Log::debug("AT: Best Fitness after = %lg\n",best_fitness);
            Log::debug("AT: Inside if statement after rnn genome\n");
            
            best_beta1 = genome_information[i][3];
            Log::debug("AT: Best beta1 = %lg\n",best_beta1);
            
        }

        Log::debug("AT: Before Average beta1\n");
        avg_beta1 += genome_information[i][3];
        Log::debug("AT: Avg beta1 = %lg\n",avg_beta1);
        Log::debug("AT: After Average beta1\n");

    }
    
    Log::debug("AT: avg beta1 before division = %lg\n",avg_beta1);
    avg_beta1 = avg_beta1 /  simplex_count;
    Log::debug("AT: avg beta1 after division = %lg\n",avg_beta1);
    
    double scale = (rng_0_1(generator) * 2.0) - 0.5;
    Log::debug("AT: avg_beta1 after adding avg and mul = %lg\n",avg_beta1);
    Log::debug("AT: best_beta1 after adding avg and mul = %lg\n",best_beta1);
    Log::debug("AT: scale after adding avg and mul = %lg\n",scale);

    tuned_beta1 = avg_beta1 + ((best_beta1 - avg_beta1) * scale);

    Log::debug("AT: beta1 after adding avg and mul = %lg\n",tuned_beta1);
    if (tuned_beta1 < beta1_min) tuned_beta1 = beta1_min;
    if (tuned_beta1 > beta1_max) tuned_beta1 = beta1_max;
    Log::debug("AT: beta1 finally = %lg\n",tuned_beta1);
    
    return tuned_beta1;
}

double WeightUpdate::generate_simplex_beta2(vector<vector<double> > genome_information, int simplex_count) {

    stringstream ss;
    for (int i = 0; i < genome_information.size(); i++) {
        for (int j = 0; j < genome_information[i].size(); j++) {
            ss << genome_information[i][j] << " ";
        }
        ss << endl;
    }
    string genome_informationString = ss.str();

    Log::debug("AT: genome_information = %s\n",genome_informationString.c_str());

    double tuned_beta2;
    double best_beta2;
    double avg_beta2 = 0;

    double best_fitness = EXAMM_MAX_DOUBLE;
    Log::debug("AT: Simplex Count for Beta2 = %d\n",simplex_count);

    for(int i=0; i<simplex_count;i++) {
    
        Log::debug("AT: After current genome\n");
        Log::debug("AT: Best Fitness Before = %lg\n",best_fitness);
    
        if( i == 0 || genome_information[i][1] < best_fitness) {
            best_fitness = genome_information[i][1];
            Log::debug("AT: Best Fitness after = %lg\n",best_fitness);
            Log::debug("AT: Inside if statement after rnn genome\n");
            
            best_beta2 = genome_information[i][4];
            Log::debug("AT: Best beta2 = %lg\n",best_beta2);
            
        }

        Log::debug("AT: Before Average beta2\n");
        avg_beta2 += genome_information[i][4];
        Log::debug("AT: Avg beta2 = %lg\n",avg_beta2);
        Log::debug("AT: After Average beta2\n");

    }
    
    Log::debug("AT: avg beta2 before division = %lg\n",avg_beta2);
    avg_beta2 = avg_beta2 /  simplex_count;
    Log::debug("AT: avg beta2 after division = %lg\n",avg_beta2);
    
    double scale = (rng_0_1(generator) * 2.0) - 0.5;
    Log::debug("AT: avg_beta2 after adding avg and mul = %lg\n",avg_beta2);
    Log::debug("AT: best_beta2 after adding avg and mul = %lg\n",best_beta2);
    Log::debug("AT: scale after adding avg and mul = %lg\n",scale);

    tuned_beta2 = avg_beta2 + ((best_beta2 - avg_beta2) * scale);

    Log::debug("AT: beta2 after adding avg and mul = %lg\n",tuned_beta2);
    if (tuned_beta2 < beta2_min) tuned_beta2 = beta2_min;
    if (tuned_beta2 > beta2_max) tuned_beta2 = beta2_max;
    Log::debug("AT: beta2 finally = %lg\n",tuned_beta2);
    
    return tuned_beta2;
}

// Definition of functions for generating SHO initial hyperparameters

double WeightUpdate::generate_initial_learning_rate() {
    double tuned_learning_rate;

    tuned_learning_rate = (rng_0_1(generator) * (initial_learning_rate_max - initial_learning_rate_min)) + initial_learning_rate_min;

    Log::debug("AT: Generated Initial learning_rate = %lg\n",tuned_learning_rate);
    return tuned_learning_rate;
}

double WeightUpdate::generate_initial_epsilon() {
    double tuned_epsilon;

    tuned_epsilon = (rng_0_1(generator) * (initial_epsilon_max - initial_epsilon_min)) + initial_epsilon_min;

    Log::debug("AT: Generated Initial epsilon = %lg\n",tuned_epsilon);
    return tuned_epsilon;
}

double WeightUpdate::generate_initial_beta1() {
    double tuned_beta1;

    tuned_beta1 = (rng_0_1(generator) * (initial_beta1_max - initial_beta1_min)) + initial_beta1_min;

    Log::debug("AT: Generated Initial beta1 = %lg\n",tuned_beta1);
    return tuned_beta1;
}

double WeightUpdate::generate_initial_beta2() {
    double tuned_beta2;

    tuned_beta2 = (rng_0_1(generator) * (initial_beta2_max - initial_beta2_min)) + initial_beta2_min;

    Log::debug("AT: Generated Initial beta2 = %lg\n",tuned_beta2);
    return tuned_beta2;
}
