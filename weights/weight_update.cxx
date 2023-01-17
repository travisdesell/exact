#include <cmath>

#include "weights/weight_update.hxx"

#include "common/arguments.hxx"
#include "common/log.hxx"

WeightUpdate::WeightUpdate() {
    // By default use RMSProp weight update
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
}

void WeightUpdate::generate_from_arguments(const vector<string> &arguments) {
    Log::info("Getting infomation on weight update methods for backprop\n");
    if (argument_exists(arguments, "--weight_update")) {
        string weight_update_method_string;
        get_argument(arguments, "--weight_update", true, weight_update_method_string);
        weight_update_method = get_enum_method_from_string(weight_update_method_string);
        Log::info("Doing backprop with weight update method: %s\n", WEIGHT_UPDATE_METHOD_STRING[weight_update_method].c_str());
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
    } else Log::info("Backprop weight update method not set, using default method %s and default parameters\n", WEIGHT_UPDATE_METHOD_STRING[weight_update_method].c_str());

    get_argument(arguments, "--learning_rate", false, learning_rate);
    get_argument(arguments, "--high_threshold", false, high_threshold);
    get_argument(arguments, "--low_threshold", false, low_threshold);
    Log::info("Backprop learning rate: %f\n", learning_rate);
    Log::info("Use high norm is set to %s, high norm is %f\n", use_high_norm ? "True" : "False", high_threshold);
    Log::info("Use low norm is set to %s, low norm is %f\n", use_low_norm ? "True" : "False", low_threshold);
}

void WeightUpdate::update_weights(vector<double> &parameters, vector<double> &velocity, vector<double> &prev_velocity, vector<double> &gradient, int32_t epoch) {
    if (weight_update_method == VANILLA) {
        vanilla_weight_update(parameters, velocity, prev_velocity, gradient, epoch);
    } else if (weight_update_method == MOMENTUM) {
        momentum_weight_update(parameters, velocity, prev_velocity, gradient, epoch);
    } else if (weight_update_method == NESTEROV) {
        nesterov_weight_update(parameters, velocity, prev_velocity, gradient, epoch);
    } else if (weight_update_method == ADAGRAD) {
        adagrad_weight_update(parameters, velocity, prev_velocity, gradient, epoch);
    } else if (weight_update_method == RMSPROP) {
        rmsprop_weight_update(parameters, velocity, prev_velocity, gradient, epoch);
    } else if (weight_update_method == ADAM) {
        adam_weight_update(parameters, velocity, prev_velocity, gradient, epoch);
    } else if (weight_update_method == ADAM_BIAS) {
        adam_bias_weight_update(parameters, velocity, prev_velocity, gradient, epoch);
    } else {
        Log::fatal("Unrecognized weight update method's enom number: %d, this should never happen!\n", weight_update_method);
        exit(1);
    }

}

void WeightUpdate::vanilla_weight_update(vector<double> &parameters, vector<double> &velocity, vector<double> &prev_velocity, vector<double> &gradient, int32_t epoch) {
    Log::trace("Doing weight update with method: %s \n", WEIGHT_UPDATE_METHOD_STRING[weight_update_method].c_str());
    for (int32_t i = 0; i < (int32_t)parameters.size(); i++) {
        parameters[i] -= learning_rate * gradient[i];
        gradient_clip(parameters[i]);
    }
}

void WeightUpdate::momentum_weight_update(vector<double> &parameters, vector<double> &velocity, vector<double> &prev_velocity, vector<double> &gradient, int32_t epoch) {
    Log::trace("Doing weight update with method: %s \n", WEIGHT_UPDATE_METHOD_STRING[weight_update_method].c_str());
    for (int32_t i = 0; i < (int32_t)parameters.size(); i++) {
        velocity[i] = momentum * velocity[i] - learning_rate * gradient[i];
        parameters[i] += velocity[i];
        gradient_clip(parameters[i]);
    }
}

void WeightUpdate::nesterov_weight_update(vector<double> &parameters, vector<double> &velocity, vector<double> &prev_velocity, vector<double> &gradient, int32_t epoch)  {
    Log::info("Doing weight update with method: %s \n", WEIGHT_UPDATE_METHOD_STRING[weight_update_method].c_str());
    for (int32_t i = 0; i < (int32_t)parameters.size(); i++) {
        prev_velocity[i] = velocity[i];
        velocity[i] = momentum * velocity[i] - learning_rate * gradient[i];
        parameters[i] += -momentum * prev_velocity[i] + (1 + momentum) * velocity[i];
        gradient_clip(parameters[i]);
    }
}

void WeightUpdate::adagrad_weight_update(vector<double> &parameters, vector<double> &velocity, vector<double> &prev_velocity, vector<double> &gradient, int32_t epoch) {
    Log::trace("Doing weight update with method: %s \n", WEIGHT_UPDATE_METHOD_STRING[weight_update_method].c_str());
    for (int32_t i = 0; i < (int32_t)parameters.size(); i++) {
        // here the velocity is the "cache" in Adagrad
        velocity[i] += gradient[i] * gradient[i];
        parameters[i] += -learning_rate * gradient[i] / (sqrt(velocity[i]) + epsilon);
        gradient_clip(parameters[i]);
    }
}

void WeightUpdate::rmsprop_weight_update(vector<double> &parameters, vector<double> &velocity, vector<double> &prev_velocity, vector<double> &gradient, int32_t epoch) {
    Log::trace("Doing weight update with method: %s \n", WEIGHT_UPDATE_METHOD_STRING[weight_update_method].c_str());
    for (int32_t i = 0; i < (int32_t)parameters.size(); i++) {
        // here the velocity is the "cache" in RMSProp
        velocity[i] = decay_rate * velocity[i] + (1 - decay_rate) * gradient[i] * gradient[i];
        parameters[i] += -learning_rate * gradient[i] / (sqrt(velocity[i]) + epsilon);
        gradient_clip(parameters[i]);
    }
}

void WeightUpdate::adam_weight_update(vector<double> &parameters, vector<double> &velocity, vector<double> &prev_velocity, vector<double> &gradient, int32_t epoch) {
    Log::trace("Doing weight update with method: %s \n", WEIGHT_UPDATE_METHOD_STRING[weight_update_method].c_str());
    for (int32_t i = 0; i < (int32_t)parameters.size(); i++) {
        // here the velocity is the "v" in adam, the prev_velocity is "m" in adam
        prev_velocity[i] = beta1 * prev_velocity[i] + (1 - beta1) * gradient[i];
        velocity[i] = beta2 * velocity[i] + (1 - beta2) * (gradient[i] * gradient[i]);
        parameters[i] += -learning_rate * prev_velocity[i] / (sqrt(velocity[i]) + epsilon);
        gradient_clip(parameters[i]);
    }
}

void WeightUpdate::adam_bias_weight_update(vector<double> &parameters, vector<double> &velocity, vector<double> &prev_velocity, vector<double> &gradient, int32_t epoch) {
    Log::trace("Doing weight update with method: %s \n", WEIGHT_UPDATE_METHOD_STRING[weight_update_method].c_str());
    for (int32_t i = 0; i < (int32_t)parameters.size(); i++) {
        // here the velocity is the "v" in adam, the prev_velocity is "m" in adam
        prev_velocity[i] = beta1 * prev_velocity[i] + (1 - beta1) * gradient[i];
        double mt = prev_velocity[i] / (1 - pow(beta1, epoch));
        velocity[i] = beta2 * velocity[i] + (1 - beta2) * (gradient[i] * gradient[i]);
        double vt = velocity[i] / (1 - pow(beta2, epoch));
        parameters[i] += -learning_rate * mt / (sqrt(vt) + epsilon);
        gradient_clip(parameters[i]);
    }
}

void WeightUpdate::gradient_clip(double &parameter) {
    if (parameter < -10.0) parameter = -10.0;
    else if (parameter > 10.0) parameter = 10.0;
}

double WeightUpdate::get_learning_rate() {
    return learning_rate;
}

double WeightUpdate::get_low_threshold() {
    return low_threshold;
}

double WeightUpdate::get_high_threshold() {
    return high_threshold;
}

void WeightUpdate::set_learning_rate(double _learning_rate) {
    learning_rate = _learning_rate;
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

double WeightUpdate::get_norm(vector<double> &analytic_gradient) {
    double norm = 0.0;
    for (int32_t i = 0; i < (int32_t)analytic_gradient.size(); i++) {
        norm += analytic_gradient[i] * analytic_gradient[i];
    }
    norm = sqrt(norm);
    return norm;
}

void WeightUpdate::norm_gradients(vector<double> &analytic_gradient, double norm) {
    if (use_high_norm && norm > high_threshold) {
    double high_threshold_norm = high_threshold / norm;
    Log::debug_no_header(", OVER THRESHOLD, multiplier: %lf", high_threshold_norm);

    for (int32_t i = 0; i < (int32_t)analytic_gradient.size(); i++) {
        analytic_gradient[i] = high_threshold_norm * analytic_gradient[i];
    }

    } else if (use_low_norm && norm < low_threshold) {
        double low_threshold_norm = low_threshold / norm;
        Log::debug_no_header(", UNDER THRESHOLD, multiplier: %lf", low_threshold_norm);

        for (int32_t i = 0; i < (int32_t)analytic_gradient.size(); i++) {
            analytic_gradient[i] = low_threshold_norm * analytic_gradient[i];
        }
    }
}