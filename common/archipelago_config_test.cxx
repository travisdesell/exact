#include "archipelago_config.hxx"

#include <fstream>
#include <iostream>
#include <sstream>

#include "args.hxx"
#include "rnn/rnn_node_interface.hxx"
#include "weight_initialize.hxx"

int main(int argc, char **argv) {
  vector<string> arguments = vector<string>(argv, argv + argc);

  ArgumentSet mt_arg_set("examm_mt_args",
                         {
                             new Argument("number_threads", "--number_threads",
                                          "Number of threads to create, parallelism", true, Argument::INT, -1),
                         });

  function<bool(ArgumentSet &)> validate_time_series_dataset_arguments = [](ArgumentSet &as) {
    if (as.args.find("filenames") != as.args.end() && as.args.find("training_indices") != as.args.end() &&
        as.args.find("testing_indices") != as.args.end())
      return true;
    if (as.args.find("training_filenames") != as.args.end() && as.args.find("testing_filenames") != as.args.end())
      return true;
    return false;
  };

  // clang-format off
  ArgumentSet time_series_dataset_arguments(
      "time_series_dataset_args",
      {new Argument(
           "filenames", "--filenames",
           "Paths of all datafiles. Specify training and testing datasets using training_indices and testing_indices.",
           false, Argument::STRING_LIST, vector<string>()),

       new Argument(
           "training_indices", "--training-indices",
           "Indices into the list of datafiles supplied by the --filenames argument that will be used for training.",
           false, Argument::INT_LIST, vector<int>()),

       new Argument(
           "testing_indices", "--testing-indices",
           "Indices into the list of the datafiles supplied by the --filenames argument that will be used for testing.",
           false, Argument::INT_LIST, vector<int>()),

       new Argument("training_filenames", "--training-filenames",
                    "Filenames of datafiles that should be used for training.", false, Argument::STRING_LIST,
                    vector<string>()),

       new Argument("testing_filenames", "--testing-filenames",
                    "Filenames of datafiles that should be used for testing.", false, Argument::STRING_LIST,
                    vector<string>())},
      validate_time_series_dataset_arguments);
  // clang-format on

  function<bool(ArgumentSet &)> validate_corpus_dataset_arguments = [](ArgumentSet &as) {
    if (as.args["testing_filenames"]->get_string_list().size() == 0) {
      Log::fatal("You must specify at least one testing file\n");
      return false;
    }
    if (as.args["training_filenames"]->get_string_list().size() == 0) {
      Log::fatal("You must specify at least one training file\n");
      return false;
    }

    return true;
  };
  // clang-format off
  ArgumentSet corpus_dataset_arguments(
      "corpus_dataset_args",
      {
        new Argument("training_filenames", "--training-filenames", "Filenames of datafiles that should be used for training.", true,Argument::STRING_LIST, vector<string>()),
      
        new Argument("testing_filenames", "--testing-filenames", "Filenames of datafiles that should be used for testing.", true, Argument::STRING_LIST, vector<string>())
      },
      validate_corpus_dataset_arguments);

  enum speciation_method { ISLAND, NEAT };
  const map<string, int> SPECIATION_STRATEGY_MAP = {
      {"island", ISLAND},
      {"neat",   NEAT  }
  };

  // Bit flags.
  enum transfer_learning_version { v1 = 1, v2 = 2, v3 = 4 };
  const map<string, int> TRANSFER_LEARNING_MAP = {
      {"v1", v1},
      {"v2", v2},
      {"v3", v3}
  };

  // clang-format off
  ArgumentSet arg_set(
      "examm_base_args",
      {
          new Argument("time_offset", "--time_offset", "Number of timesteps in the future to predict", false,
                       Argument::INT, 1),

          new Argument("max_genomes", "--max-genomes",
                       "number of genomes to be generated before termiinating the program.", true, Argument::INT, 01),

          new EnumArgument("speciation_method", "--speciation-method", "What speciation strategy to use.", false,
                           "island", SPECIATION_STRATEGY_MAP, false),

          new Argument("max_time_minutes", "--max-time-minutes",
                       "maximum amount of time the program can run. by default there is no time limit, indicated by a "
                       "negative value.",
                       false, Argument::INT, -1),

          new ConstrainedArgument(
              "dropout_probability", "--dropout-probability", "Dropout probability.",
              [](Argument::data &data) { return 0.0 <= get<double>(data) && 1.0 >= get<double>(data); }, false,
              Argument::DOUBLE, 0.0),

          new Argument("output_directory", "--output-directory", "Directory to write results and logs to", false,
                       Argument::STRING, ""),

          new Argument("seed_genome_path", "--seed-genome-path",
                       "Path to a binary RNN Genome that should be used as the first genome.", false, Argument::STRING,
                       "INVALID_SEED"),
      });
  // clang-format on

  // clang-format off
  ArgumentSet transfer_learning_set(
      "transfer_learning_args",
      {new EnumArgument("transfer_learning_version", "--transfer-learning-version",
                        "Which version of transfer learning to use", true, vector<int>(), TRANSFER_LEARNING_MAP, false),

       new Argument("epigenetic_weights", "--epigenetic-weights",
                    "When this flag is present, epigenetic weight information will be used to create new weights.",
                    true, Argument::BOOL, false)},
      [](ArgumentSet &as) {
        if (as.args["transfer_learning_version"]->get_bitflags() == 0) {
          Log::fatal("Transfer learning version is invalid\n");
          return false;
        }
        return true;
      });
  // clang-format on

  // clang-format off
  ArgumentSet training_parameters_set(
      "training_parameter_args",
      {
          new Argument("use_random_sequence_length", "--use-random-sequence-length",
                       "If true, rather than using whole files as training examples, random samples from the data will "
                       "be used instead",
                       false, Argument::BOOL, false),

          new Argument("sequence_length_lower_bound", "--sequence-length-lower-bound",
                       "Lower bound for the chunk size to train networks on (number of timesteps).", false,
                       Argument::INT, 30),

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

          new Argument(
              "low_threshold", "--low-threshold",
              "Minimum norm used for gradient descent. Gradients will be adjusted to norm to this value if they "
              "norm to something less than it.",
              false, Argument::DOUBLE, 0.05),
      });
  // clang-format on

  // clang-format off
  ArgumentSet genome_operators_set(
      "genome_operators_args",
      {
          new Argument("max_intra_crossover_parents", "--max-intra-crossover-parents",
                       "Maximum n of parents used during intra island crossover", false, Argument::INT, 2),
          new Argument("min_intra_crossover_parents", "--min-intra-crossover-parents",
                       "Minimum n of parents used during intra island crossover", false, Argument::INT, 2),

          new Argument("max_inter_crossover_parents", "--max-inter-crossover-parents",
                       "Maximum n of parents used during inter island crossover", false, Argument::INT, 2),
          new Argument("min_inter_crossover_parents", "--min-inter-crossover-parents",
                       "Minimum n of parents used during inter island crossover", false, Argument::INT, 2),

          new Argument("max_mutations", "--max-mutations", "Maximum number of mutations that will be applied", false,
                       Argument::INT, 1),
          new Argument("min_mutations", "--min-mutations", "Minimum number of mutations that will be applied", false,
                       Argument::INT, 1),

          new Argument("min_recurrent_depth", "--min-recurrent-depth",
                       "Minimum number of timesteps recurrent edge should go back.", false, Argument::INT, 1),

          new Argument("max_recurrent_depth", "--max-recurrent-depth",
                       "Maximum number of timesteps a recurrent edge can go back.", false, Argument::INT, 1),

          new EnumArgument("possible_node_types", "--possible-node-types", "List of RNN node types", false,
                           vector<string>({"simple"}), NODE_TYPE_MAP, false),

          new EnumArgument("weight_initialize", "--weight-initialize", "method of generating new weights", false,
                           "lamarckian", WEIGHT_INITIALIZE_MAP, false),

          new EnumArgument("weight_inheritance", "--weight-inheritance", "method of inheriting weights", false,
                           "lamarckian", WEIGHT_INITIALIZE_MAP, false),

          new EnumArgument("weight_mutated_component", "--weight-mutated-component",
                           "method of generating new weights when a new component is created through mutation", false,
                           "lamarckian", WEIGHT_INITIALIZE_MAP, false),

      },
      [](ArgumentSet &as) {
        if (as.args["min_mutations"]->get_int() > as.args["max_mutations"]->get_int()) {
          Log::fatal(
              "Minimum mutations is greater than the maximum number of mutations. Fix your configuration file.\n");
          return false;
        }

        int max_inter = as.args["max_inter_crossover_parents"]->get_int();
        int min_inter = as.args["min_inter_crossover_parents"]->get_int();

        if (max_inter < min_inter) {
          Log::fatal(
              "Minimum number of parents for inter island crossover is greater than the maximum. Fix your "
              "configuration.\n");
          return false;
        }

        int max_intra = as.args["max_intra_crossover_parents"]->get_int();
        int min_intra = as.args["min_intra_crossover_parents"]->get_int();

        if (max_intra < min_intra) {
          Log::fatal(
              "Minimum number of parents for intra island crossover is greater than the maximum. Fix your "
              "configuration.\n");
          return false;
        }

        int island_size = as.args["island_size"]->get_int();
        if (max_intra > island_size) {
          Log::fatal(
              "Maximum number of parents for intra island crossover is greater than the island size. Fix your "
              "configuration.\n");
          return false;
        }

        if (max_inter > island_size) {
          Log::fatal(
              "Maximum number of parents for inter island crossover is greater than the island size. Fix your "
              "configuration.\n");
          return false;
        }

        return true;
      });
  // clang-format on

  // clang-format off
  ArgumentSet island_arg_set(
      "island_args",
      {
          new Argument("n_islands", "--n_islands", "number of islands", false, Argument::INT, 8),

          new Argument("island_size", "--island_size", "maximum number of genomes per island", false, Argument::INT, 8),

          new Argument("start_filled", "--start-filled",
                       "Whether to start the algorithm with islands full of random genomes.", false, Argument::BOOL,
                       false),

          new Argument("extinction_event_generation_number", "--extinction-event-generation-number",
                       "When to perform an island extinction.", false, Argument::INT, INT_MAX),

          new Argument(
              "repeat_extinction", "--repeat-extinction",
              "Whether or not to repeat the extinction event every $extinction_event_generation_number genomes.", false,
              Argument::BOOL, false),
      });
  // clang-format on

  // clang-format off
  ArgumentSet neat_arg_set(
      "neat_args", {new ConstrainedArgument(
                       "neat_params", "--neat-parameters", "c1, c2, and c3 values for the NEAT speciation strategy",
                       [](Argument::data &data) {
                         vector<double> &d = get<vector<double>>(data);
                         if (d.size() != 3) {
                           Log::fatal("You must supply exactly 3 arguments to --neat-parameters\n");
                           return false;
                         }
                         return true;
                       },
                       true, Argument::DOUBLE_LIST)});
  // clang-format on

  Log::initialize(arguments);
  Log::set_id("main");

  std::ifstream t(argv[1]);
  std::stringstream buffer;
  buffer << t.rdbuf();

  string s = buffer.str();

  int nn = 32;
  map<string, node_index_type> m;
  m["n_islands"] = 15;
  ArchipelagoConfig config = ArchipelagoConfig::from_string(s, nn, m);

  for (int i = 0; i < nn; i++) {
    Log::info("");
    for (int j = 0; j < nn; j++) { Log::info_no_header("%s ", config.connections[i][j] ? "X" : "~"); }
    Log::info_no_header("\n");
  }

  return 0;
}
