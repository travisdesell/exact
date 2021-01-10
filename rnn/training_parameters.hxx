#ifndef TRAINING_PARAMETERS_HXX
#define TRAINING_PARAMETERS_HXX 1

/**
 * Parameters regarding the training of genomes 
 **/
class TrainingParameters {
    public:
        /**
         * Static training parameters
         **/
        static constexpr bool epigenetic_weights = true;
       

        /**
         * Configurable training parameters
         **/
        const int32_t bp_iterations;

        const double high_threshold;
        const double low_threshold;
        const double learning_rate;
        
        const bool epigenetic_weights;
        const bool use_high_threshold;
        const bool use_low_threshold;
        const bool use_regression;


        TrainingParameters(
                int32_t _bp_iterations,
                double _high_threshold,
                double _low_threshold,
                double _learning_rate,
                bool _epigenetic_weights,
                bool _use_high_threshold,
                bool _use_low_threshold,
                bool _use_regression);
};

#endif
