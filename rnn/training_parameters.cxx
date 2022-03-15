#include "training_parameters.hxx"

ArgumentSet TrainingParameters::arguments(
    "training_parameter_args",
    {
        new Argument("use_random_sequence_length", "--use-random-sequence-length",
                     "If true, rather than using whole files as training examples, random samples from the data will "
                     "be used instead",
                     false, Argument::BOOL, false),

        new Argument("sequence_length_lower_bound", "--sequence-length-lower-bound",
                     "Lower bound for the chunk size to train networks on (number of timesteps).", false, Argument::INT,
                     30),

        new Argument("sequence_length_upper_bound", "--sequence-length-upper-bound",
                     "Upper, bound for the chunk size to train networks on (number of timesteps).", false,
                     Argument::INT, 100),

        new Argument("mu", "--mu", "Mu value used for nesterov momentum.", false, Argument::DOUBLE, 0.9),

        new Argument("no_nesterov_momentum", "--no-nesterov-momentum",
                     "Truns off nesterov momentum during backpropagation.", false, Argument::BOOL, false),

        new Argument("bp_iterations", "--bp-iterations", "number of backpropagation iterations to train genomes for",
                     true, Argument::INT, 1),

        new Argument("learning_rate", "--learning-rate", "Learning rate used for backpropagation", false,
                     Argument::DOUBLE, 0.001),

        new Argument("high_threshold", "--high-threshold",
                     "Maximum norm used for gradient descent. Gradients will be adjusted to norm to less than this "
                     "value if they exceed it.",
                     false, Argument::DOUBLE, 1.0),

        new Argument("low_threshold", "--low-threshold",
                     "Minimum norm used for gradient descent. Gradients will be adjusted to norm to this value if they "
                     "norm to something less than it.",
                     false, Argument::DOUBLE, 0.05),
    });

TrainingParameters::TrainingParameters(uint32_t bp_iterations, uint32_t sequence_lower_bound,
                                       uint32_t sequence_upper_bound, double low_threshold, double high_threshold,
                                       double learning_rate, double dropout_probability, double mu,
                                       bool use_nesterov_momentum, bool use_regression, bool use_dropout,
                                       bool use_low_norm, bool use_high_norm, bool use_random_sequence_length)
    : bp_iterations(bp_iterations),
      sequence_lower_bound(sequence_lower_bound),
      sequence_upper_bound(sequence_upper_bound),
      low_threshold(low_threshold),
      high_threshold(high_threshold),
      learning_rate(learning_rate),
      dropout_probability(dropout_probability),
      mu(mu),
      use_nesterov_momentum(use_nesterov_momentum),
      use_regression(use_regression),
      use_dropout(use_dropout),
      use_low_norm(use_low_norm),
      use_high_norm(use_high_norm),
      use_random_sequence_length(use_random_sequence_length) {}

TrainingParameters::TrainingParameters()
    : bp_iterations(10),
      sequence_lower_bound(10),
      sequence_upper_bound(100),
      low_threshold(0.05),
      high_threshold(1.0),
      learning_rate(0.001),
      dropout_probability(0.5),
      mu(0.9),
      use_nesterov_momentum(true),
      use_regression(false),
      use_dropout(false),
      use_low_norm(true),
      use_high_norm(true),
      use_random_sequence_length(false) {}

TrainingParameters::TrainingParameters(istream &bin_istream) {
  bin_istream.read((char *) &bp_iterations, sizeof(int32_t));
  bin_istream.read((char *) &sequence_lower_bound, sizeof(int32_t));
  bin_istream.read((char *) &sequence_upper_bound, sizeof(int32_t));

  bin_istream.read((char *) &low_threshold, sizeof(double));
  bin_istream.read((char *) &high_threshold, sizeof(double));
  bin_istream.read((char *) &learning_rate, sizeof(double));
  bin_istream.read((char *) &dropout_probability, sizeof(double));
  bin_istream.read((char *) &mu, sizeof(double));

  bin_istream.read((char *) &use_nesterov_momentum, sizeof(bool));
  bin_istream.read((char *) &use_regression, sizeof(bool));
  bin_istream.read((char *) &use_dropout, sizeof(bool));
  bin_istream.read((char *) &use_low_norm, sizeof(bool));
  bin_istream.read((char *) &use_high_norm, sizeof(bool));
  bin_istream.read((char *) &use_random_sequence_length, sizeof(bool));
}

void TrainingParameters::write_to_stream(ostream &bin_ostream) const {
  bin_ostream.write((char *) &bp_iterations, sizeof(int32_t));
  bin_ostream.write((char *) &sequence_lower_bound, sizeof(int32_t));
  bin_ostream.write((char *) &sequence_upper_bound, sizeof(int32_t));

  bin_ostream.write((char *) &low_threshold, sizeof(double));
  bin_ostream.write((char *) &high_threshold, sizeof(double));
  bin_ostream.write((char *) &learning_rate, sizeof(double));
  bin_ostream.write((char *) &dropout_probability, sizeof(double));
  bin_ostream.write((char *) &mu, sizeof(double));

  bin_ostream.write((char *) &use_nesterov_momentum, sizeof(bool));
  bin_ostream.write((char *) &use_regression, sizeof(bool));
  bin_ostream.write((char *) &use_dropout, sizeof(bool));
  bin_ostream.write((char *) &use_low_norm, sizeof(bool));
  bin_ostream.write((char *) &use_high_norm, sizeof(bool));
  bin_ostream.write((char *) &use_random_sequence_length, sizeof(bool));
}
