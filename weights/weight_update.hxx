#ifndef WEIGHT_UPDATE_HXX
#define WEIGHT_UPDATE_HXX

#include <string>
using std::string;

#include <vector>
using std::vector;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include "common/arguments.hxx"

enum WeightUpdateMethod { VANILLA = 0, MOMENTUM = 1, NESTEROV = 2, ADAGRAD = 3, RMSPROP = 4, ADAM = 5, ADAM_BIAS = 6 };

static string WEIGHT_UPDATE_METHOD_STRING[] = {"vanilla", "momentum", "nesterov", "adagrad",
                                               "rmsprop", "adam",     "adam-bias"};
static int32_t NUM_WEIGHT_UPDATE_TYPES = 7;

inline WeightUpdateMethod get_enum_method_from_string(string input_string) {
    WeightUpdateMethod method = ADAM;
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
    
    minstd_rand0 generator;
    uniform_real_distribution<double> rng_0_1;
    
    // Declaring variables for keeping track of ranges of SHO tuned hyperparameters 
    double learning_rate_min;
    double learning_rate_max;

    double epsilon_min;
    double epsilon_max;

    double beta1_min;
    double beta1_max;

    double beta2_min;
    double beta2_max;

    // Initial SHO hyperparameter range values to be pulled from uniform real distribution 
    uniform_real_distribution<double> rng_ilr_min;
    uniform_real_distribution<double> rng_ilr_max;
    uniform_real_distribution<double> rng_ieps_min;
    uniform_real_distribution<double> rng_ieps_max;
    uniform_real_distribution<double> rng_ib1_min;
    uniform_real_distribution<double> rng_ib1_max;
    uniform_real_distribution<double> rng_ib2_min;
    uniform_real_distribution<double> rng_ib2_max;

    // Declaring variables for keeping track of ranges of initial SHO hyperparameters 
    double initial_learning_rate_min;
    double initial_learning_rate_max;

    double initial_epsilon_min;
    double initial_epsilon_max;
    
    double initial_beta1_min;
    double initial_beta1_max;

    double initial_beta2_min;
    double initial_beta2_max;

   public:
    static bool use_SHO;
    WeightUpdate();
    explicit WeightUpdate(const vector<string>& arguments);
    void generate_from_arguments(const vector<string>& arguments);

    // Add the optional SHO hyperparameters to each of the declaration of weight update methods
    void update_weights(
        vector<double>& parameters, vector<double>& velocity, vector<double>& prev_velocity, vector<double>& gradient,
        int32_t epoch, double _learning_rate=NULL, double _epsilon=NULL, double _beta1=NULL, double _beta2=NULL
    );

    void vanilla_weight_update(
        vector<double>& parameters, vector<double>& velocity, vector<double>& prev_velocity, vector<double>& gradient,
        int32_t epoch, double _learning_rate
    );
    void momentum_weight_update(
        vector<double>& parameters, vector<double>& velocity, vector<double>& prev_velocity, vector<double>& gradient,
        int32_t epoch, double _learning_rate
    );
    void nesterov_weight_update(
        vector<double>& parameters, vector<double>& velocity, vector<double>& prev_velocity, vector<double>& gradient,
        int32_t epoch, double _learning_rate
    );
    void adagrad_weight_update(
        vector<double>& parameters, vector<double>& velocity, vector<double>& prev_velocity, vector<double>& gradient,
        int32_t epoch, double _learning_rate
    );
    void rmsprop_weight_update(
        vector<double>& parameters, vector<double>& velocity, vector<double>& prev_velocity, vector<double>& gradient,
        int32_t epoch, double _learning_rate
    );
    void adam_weight_update(
        vector<double>& parameters, vector<double>& velocity, vector<double>& prev_velocity, vector<double>& gradient,
        int32_t epoch, double _learning_rate, double _epsilon, double _beta1, double _beta2
    );
    void adam_bias_weight_update(
        vector<double>& parameters, vector<double>& velocity, vector<double>& prev_velocity, vector<double>& gradient,
        int32_t epoch, double _learning_rate
    );

    void gradient_clip(double& parameter);

    // Declaration of Setters for SHO tuned hyperparameters
    void set_learning_rate(double _learning_rate);
    void set_epsilon(double _epsilon);
    void set_beta1(double _beta1);
    void set_beta2(double _beta2);

    void disable_high_threshold();
    void enable_high_threshold(double _high_threshold);
    void disable_low_threshold();
    void enable_low_threshold(double _low_threshold);

    // Declaration of Getters for SHO tuned hyperparameters
    double get_learning_rate();
    double get_epsilon();
    double get_beta1();
    double get_beta2();

    double get_low_threshold();
    double get_high_threshold();

    double get_norm(vector<double>& analytic_gradient);
    void norm_gradients(vector<double>& analytic_gradient, double norm);
    
    // Declaration of functions for generating SHO tuned hyperparameters
    double generate_simplex_learning_rate(vector<vector<double> > genome_information, int simplex_count);
    double generate_simplex_epsilon(vector<vector<double> > genome_information, int simplex_count);
    double generate_simplex_beta1(vector<vector<double> > genome_information, int simplex_count);
    double generate_simplex_beta2(vector<vector<double> > genome_information, int simplex_count);

    // Declaration of functions for generating SHO initial hyperparameters
    double generate_initial_learning_rate();
    double generate_initial_epsilon();
    double generate_initial_beta1();
    double generate_initial_beta2();

};

#endif
