#include "training_parameters.hxx"

TrainingParameters::TrainingParameters(
    uint32_t bp_iterations, uint32_t sequence_lower_bound,
    uint32_t sequence_upper_bound, double low_threshold, double high_threshold,
    double learning_rate, double dropout_probability, double mu,
    bool use_nesterov_momentum, bool use_regression, bool use_dropout,
    bool use_low_norm, bool use_high_norm, bool use_random_sequence_length)
    : bp_iterations(bp_iterations), sequence_lower_bound(sequence_lower_bound),
      sequence_upper_bound(sequence_upper_bound), low_threshold(low_threshold),
      high_threshold(high_threshold), learning_rate(learning_rate),
      dropout_probability(dropout_probability), mu(mu),
      use_nesterov_momentum(use_nesterov_momentum),
      use_regression(use_regression), use_dropout(use_dropout),
      use_low_norm(use_low_norm), use_high_norm(use_high_norm),
      use_random_sequence_length(use_random_sequence_length) {}

TrainingParameters::TrainingParameters()
    : bp_iterations(10), sequence_lower_bound(10), sequence_upper_bound(100),
      low_threshold(0.05), high_threshold(1.0), learning_rate(0.001),
      dropout_probability(0.5), mu(0.9), use_nesterov_momentum(true),
      use_regression(false), use_dropout(false), use_low_norm(true),
      use_high_norm(true), use_random_sequence_length(false) {}

TrainingParameters::TrainingParameters(istream &bin_istream) {
  bin_istream.read((char *)&bp_iterations, sizeof(int32_t));
  bin_istream.read((char *)&sequence_lower_bound, sizeof(int32_t));
  bin_istream.read((char *)&sequence_upper_bound, sizeof(int32_t));

  bin_istream.read((char *)&low_threshold, sizeof(double));
  bin_istream.read((char *)&high_threshold, sizeof(double));
  bin_istream.read((char *)&learning_rate, sizeof(double));
  bin_istream.read((char *)&dropout_probability, sizeof(double));
  bin_istream.read((char *)&mu, sizeof(double));

  bin_istream.read((char *)&use_nesterov_momentum, sizeof(bool));
  bin_istream.read((char *)&use_regression, sizeof(bool));
  bin_istream.read((char *)&use_dropout, sizeof(bool));
  bin_istream.read((char *)&use_low_norm, sizeof(bool));
  bin_istream.read((char *)&use_high_norm, sizeof(bool));
  bin_istream.read((char *)&use_random_sequence_length, sizeof(bool));
}

void TrainingParameters::write_to_stream(ostream &bin_ostream) const {
  bin_ostream.write((char *)&bp_iterations, sizeof(int32_t));
  bin_ostream.write((char *)&sequence_lower_bound, sizeof(int32_t));
  bin_ostream.write((char *)&sequence_upper_bound, sizeof(int32_t));

  bin_ostream.write((char *)&low_threshold, sizeof(double));
  bin_ostream.write((char *)&high_threshold, sizeof(double));
  bin_ostream.write((char *)&learning_rate, sizeof(double));
  bin_ostream.write((char *)&dropout_probability, sizeof(double));
  bin_ostream.write((char *)&mu, sizeof(double));

  bin_ostream.write((char *)&use_nesterov_momentum, sizeof(bool));
  bin_ostream.write((char *)&use_regression, sizeof(bool));
  bin_ostream.write((char *)&use_dropout, sizeof(bool));
  bin_ostream.write((char *)&use_low_norm, sizeof(bool));
  bin_ostream.write((char *)&use_high_norm, sizeof(bool));
  bin_ostream.write((char *)&use_random_sequence_length, sizeof(bool));
}
