#include "training_parameters.hxx"

TrainingParameters::TrainingParameters(
                int32_t _bp_iterations,
                double _high_threshold,
                double _low_threshold,
                double _learning_rate,
                bool _use_high_threshold,
                bool _use_low_threshold,
                bool _use_regression) 
    :   bp_iterationss(_bp_iterations), 
        high_threshold(_high_threshold),
        low_threshold(_low_threshold),
        learning_rate(_learning_rate),
        use_epigenetic_weights(_use_epigenetic_weights),
        use_high_threshold(_use_high_threshold),
        use_low_threshold(_use_low_threshold),
        use_regression(_use_regression) { }

