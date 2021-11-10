#ifndef TRAINING_PARAMETERS_HXX
#define TRAINING_PARAMETERS_HXX 1

#include <stdint.h>

#include <fstream>
using std::ifstream;
using std::istream;
using std::ofstream;
using std::ostream;

/**
 * Parameters regarding the training of genomes
 **/
class TrainingParameters {
public:
  /**
   * Static training parameters
   **/
  static constexpr bool use_epigenetic_weights = true;

  /**
   * Configurable training parameters
   **/
  uint32_t bp_iterations;

  // Used if use_random_sequence_length is true
  uint32_t sequence_lower_bound;
  uint32_t sequence_upper_bound;

  double low_threshold;
  double high_threshold;
  double learning_rate;
  double dropout_probability;
  // Used with nesterov momentum
  double mu;

  bool use_nesterov_momentum;
  bool use_regression;
  bool use_dropout;
  bool use_low_norm;
  bool use_high_norm;
  bool use_random_sequence_length;

  TrainingParameters(uint32_t bp_iterations, uint32_t sequence_lower_bound,
                     uint32_t sequence_upper_bound, double low_threshold,
                     double high_threshold, double learning_rate,
                     double dropout_probability, double mu,
                     bool use_nesterov_momentum, bool use_regression,
                     bool use_dropout, bool use_low_norm, bool use_high_norm,
                     bool use_random_sequence_length);

  TrainingParameters(istream &bin_istream);

  // Default constructor used when being read from a genome file; should not be
  // used otherwise!
  TrainingParameters();

  void write_to_stream(ostream &bin_ostream);
};

#endif
