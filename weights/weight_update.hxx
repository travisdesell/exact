#ifndef WEIGHT_UPDATE_HXX
#define WEIGHT_UPDATE_HXX

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/arguments.hxx"

enum WeightUpdateMethod {
    VANILLA = 0, 
    MOMENTUM = 1, 
    NESTEROV = 2, 
    ADAGRAD = 3,
    RMSPROP = 4,
    ADAM = 5,
    ADAM_BIAS = 6
};

static string WEIGHT_UPDATE_METHOD_STRING[] = {"vanilla", "momentum", "nesterov", "adagrad", "rmsprop", "adam", "adam-bias"};
static int32_t NUM_WEIGHT_UPDATE_TYPES = 7;


inline WeightUpdateMethod get_enum_method_from_string(string input_string) {
    WeightUpdateMethod method;
    for (int i = 0; i < NUM_WEIGHT_UPDATE_TYPES; i++) {
        if (input_string.compare(WEIGHT_UPDATE_METHOD_STRING[i]) == 0) {
            method = static_cast<WeightUpdateMethod>(i);
        }
    }
    return method;
}

template <typename Weight_Update_Type>
int32_t enum_method_to_integer(Weight_Update_Type method) {
    return static_cast<typename std::underlying_type<Weight_Update_Type>::type>(method);
}

class WeightUpdate {
    private:
        WeightUpdateMethod weight_update_method;
        double momentum;
        double epsilon;
        double decay_rate;
        double beta1;
        double beta2;
        double learning_rate;

        bool use_high_norm;
        double high_threshold;
        bool use_low_norm;
        double low_threshold;

    public:
        WeightUpdate();
        explicit WeightUpdate(const vector<string> &arguments);
        void generate_from_arguments(const vector<string> &arguments);

        void update_weights(vector<double> &parameters, vector<double> &velocity, vector<double> &prev_velocity, vector<double> &gradient, int32_t epoch);
        
        void vanilla_weight_update(vector<double> &parameters, vector<double> &velocity, vector<double> &prev_velocity, vector<double> &gradient, int32_t epoch);
        void momentum_weight_update(vector<double> &parameters, vector<double> &velocity, vector<double> &prev_velocity, vector<double> &gradient, int32_t epoch);
        void nesterov_weight_update(vector<double> &parameters, vector<double> &velocity, vector<double> &prev_velocity, vector<double> &gradient, int32_t epoch);
        void adagrad_weight_update(vector<double> &parameters, vector<double> &velocity, vector<double> &prev_velocity, vector<double> &gradient, int32_t epoch);
        void rmsprop_weight_update(vector<double> &parameters, vector<double> &velocity, vector<double> &prev_velocity, vector<double> &gradient, int32_t epoch);
        void adam_weight_update(vector<double> &parameters, vector<double> &velocity, vector<double> &prev_velocity, vector<double> &gradient, int32_t epoch);
        void adam_bias_weight_update(vector<double> &parameters, vector<double> &velocity, vector<double> &prev_velocity, vector<double> &gradient, int32_t epoch);
        
        void gradient_clip(double &parameter);

        void set_learning_rate(double _learning_rate);
        void disable_high_threshold();
        void enable_high_threshold(double _high_threshold);
        void disable_low_threshold();
        void enable_low_threshold(double _low_threshold);

        double get_learning_rate();
        double get_low_threshold();
        double get_high_threshold();

        double get_norm(vector<double> &analytic_gradient);
        void norm_gradients(vector<double> &analytic_gradient, double norm);
};



#endif
