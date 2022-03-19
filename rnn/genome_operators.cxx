#include "genome_operators.hxx"

#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <utility>

#include "common/weight_initialize.hxx"

ArgumentSet GenomeOperators::arguments(
    "genome_operators_args",
    {
        new Argument("max_intra_crossover_parents", "--max-intra-crossover-parents",
                     "Maximum n of parents used during intra group (island / species) crossover", false, Argument::INT,
                     2),
        new Argument("min_intra_crossover_parents", "--min-intra-crossover-parents",
                     "Minimum n of parents used during intra group (island / species) crossover", false, Argument::INT,
                     2),

        new Argument("max_inter_crossover_parents", "--max-inter-crossover-parents",
                     "Maximum n of parents used during inter group (island / species) crossover", false, Argument::INT,
                     2),
        new Argument("min_inter_crossover_parents", "--min-inter-crossover-parents",
                     "Minimum n of parents used during inter group (island / species) crossover", false, Argument::INT,
                     2),

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
        Log::fatal("Minimum mutations is greater than the maximum number of mutations. Fix your configuration file.\n");
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

GenomeOperators::GenomeOperators(int32_t _number_inputs, int32_t _number_outputs,
                                 pair<int32_t, int32_t> n_parents_intra_range,
                                 pair<int32_t, int32_t> n_parents_inter_range, pair<int32_t, int32_t> n_mutations_range,
                                 int32_t _min_recurrent_depth, int32_t _max_recurrent_depth,
                                 WeightType _weight_initialize, WeightType _weight_inheritance,
                                 WeightType _mutated_component_weight, DatasetMeta _dataset_meta,
                                 TrainingParameters _training_parameters, vector<string> _possible_node_type_strings)
    : dataset_meta(_dataset_meta),
      number_inputs(_number_inputs),
      number_outputs(_number_outputs),
      n_parents_intra_range(n_parents_intra_range),
      n_parents_inter_range(n_parents_inter_range),
      n_mutations_range(n_mutations_range),
      weight_initialize(_weight_initialize),
      weight_inheritance(_weight_inheritance),
      mutated_component_weight(_mutated_component_weight),
      generator((unsigned int) time(0)),
      training_parameters(_training_parameters) {
  set_possible_node_types(_possible_node_type_strings);

  recurrent_depth_dist = uniform_int_distribution(_min_recurrent_depth, _max_recurrent_depth);
  node_index_dist = uniform_int_distribution(0, (int) possible_node_types.size() - 1);
}

void GenomeOperators::set_possible_node_types(vector<string> &possible_node_type_strings) {
  if (possible_node_type_strings.size() == 0) {
    possible_node_types = vector(
        {SIMPLE_NODE, JORDAN_NODE, ELMAN_NODE, UGRNN_NODE, MGU_NODE, GRU_NODE, LSTM_NODE, ENARC_NODE, DELTA_NODE});
    return;
  }

  possible_node_types.clear();

  for (uint32_t i = 0; i < possible_node_type_strings.size(); i++) {
    string node_type_s = possible_node_type_strings[i];
    std::transform(node_type_s.begin(), node_type_s.end(), node_type_s.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    auto it = NODE_TYPE_MAP.find(node_type_s);
    if (it == NODE_TYPE_MAP.end()) {
      Log::error("unknown node type: '%s'\n", node_type_s.c_str());
      exit(1);
    }

    possible_node_types.push_back((rnn_node_type) it->second);
  }

  if (possible_node_types.size() == 0) {
    Log::fatal(
        "failed to specify any node types: there must be at least one "
        "node type specified");
    exit(1);
  }
}

int GenomeOperators::get_random_node_type() { return GenomeOperators::possible_node_types[node_index_dist(generator)]; }

void GenomeOperators::finalize_genome(RNN_Genome *genome) {
  genome->set_parameter_names(dataset_meta.input_parameter_names, dataset_meta.output_parameter_names);
  genome->set_normalize_bounds(dataset_meta.normalize_type, dataset_meta.normalize_mins, dataset_meta.normalize_maxs,
                               dataset_meta.normalize_avgs, dataset_meta.normalize_std_devs);

  if (!TrainingParameters::use_epigenetic_weights) genome->initialize_randomly();
}

int32_t GenomeOperators::get_random_n_mutations() {
  int32_t dif = n_mutations_range.second - n_mutations_range.first + 1;
  return n_mutations_range.first + rng_0_1(generator) * dif;
}

RNN_Genome *GenomeOperators::mutate(RNN_Genome *g, int32_t n_mutations) {
  double mu, sigma;

  // g->write_graphviz("rnn_genome_premutate_" +
  // to_string(g->get_generation_id()) + ".gv");
  Log::info("mutating genome with %d mutations\n", n_mutations);

  g->get_mu_sigma(g->best_parameters, mu, sigma);
  g->clear_generated_by();

  // use best weights if available
  if (g->best_parameters.size() == 0) {
    g->set_weights(g->initial_parameters);
    g->get_mu_sigma(g->initial_parameters, mu, sigma);
  } else {
    g->set_weights(g->best_parameters);
    g->get_mu_sigma(g->best_parameters, mu, sigma);
  }

  int number_mutations = 0;
  bool modified = false;

  for (;;) {
    if (modified) {
      modified = false;
      number_mutations++;
    }

    if (number_mutations >= n_mutations) break;

    g->assign_reachability();
    double rng = rng_0_1(generator) * mutation_rates_total;
    int new_node_type = get_random_node_type();
    string node_type_str = NODE_TYPES[new_node_type];

    if (rng < clone_rate) {
      Log::debug("\tcloned\n");
      g->set_generated_by("clone");
      modified = true;
      continue;
    }

    rng -= clone_rate;
    if (rng < add_edge_rate) {
      modified = g->add_edge(mu, sigma);
      Log::debug("\tadding edge, modified: %d\n", modified);
      if (modified) g->set_generated_by("add_edge");
      continue;
    }

    rng -= add_edge_rate;
    if (rng < add_recurrent_edge_rate) {
      modified = g->add_recurrent_edge(mu, sigma, recurrent_depth_dist);
      Log::debug("\tadding recurrent edge, modified: %d\n", modified);
      if (modified) g->set_generated_by("add_recurrent_edge");
      continue;
    }

    rng -= add_recurrent_edge_rate;
    if (rng < enable_edge_rate) {
      modified = g->enable_edge();
      Log::debug("\tenabling edge, modified: %d\n", modified);
      if (modified) g->set_generated_by("enable_edge");
      continue;
    }

    rng -= enable_edge_rate;
    if (rng < disable_edge_rate) {
      modified = g->disable_edge();
      Log::debug("\tdisabling edge, modified: %d\n", modified);
      if (modified) g->set_generated_by("disable_edge");
      continue;
    }

    rng -= disable_edge_rate;
    if (rng < split_edge_rate) {
      modified = g->split_edge(mu, sigma, new_node_type, recurrent_depth_dist);
      Log::debug("\tsplitting edge, modified: %d\n", modified);
      if (modified) g->set_generated_by("split_edge(" + node_type_str + ")");
      continue;
    }

    rng -= split_edge_rate;
    if (rng < add_node_rate) {
      modified = g->add_node(mu, sigma, new_node_type, recurrent_depth_dist);
      Log::debug("\tadding node, modified: %d\n", modified);
      if (modified) g->set_generated_by("add_node(" + node_type_str + ")");
      continue;
    }

    rng -= add_node_rate;
    if (rng < enable_node_rate) {
      modified = g->enable_node();
      Log::debug("\tenabling node, modified: %d\n", modified);
      if (modified) g->set_generated_by("enable_node");
      continue;
    }

    rng -= enable_node_rate;
    if (rng < disable_node_rate) {
      modified = g->disable_node();
      Log::debug("\tdisabling node, modified: %d\n", modified);
      if (modified) g->set_generated_by("disable_node");
      continue;
    }

    rng -= disable_node_rate;
    if (rng < split_node_rate) {
      modified = g->split_node(mu, sigma, new_node_type, recurrent_depth_dist);
      Log::debug("\tsplitting node, modified: %d\n", modified);
      if (modified) g->set_generated_by("split_node(" + node_type_str + ")");
      continue;
    }

    rng -= split_node_rate;
    if (rng < merge_node_rate) {
      modified = g->merge_node(mu, sigma, new_node_type, recurrent_depth_dist);
      Log::debug("\tmerging node, modified: %d\n", modified);
      if (modified) g->set_generated_by("merge_node(" + node_type_str + ")");
      continue;
    }

    rng -= merge_node_rate;
  }

  // get the new set of parameters (as new paramters may have been
  // added duriung mutation) and set them to the initial parameters
  // for epigenetic_initialization

  vector<double> new_parameters;

  g->get_weights(new_parameters);
  g->initial_parameters = new_parameters;

  if (Log::at_level(Log::DEBUG)) { g->get_mu_sigma(new_parameters, mu, sigma); }

  g->assign_reachability();

  // reset the genomes statistics (as these carry over on copy)
  g->best_validation_mse = EXAMM_MAX_DOUBLE;
  g->best_validation_mae = EXAMM_MAX_DOUBLE;

  if (Log::at_level(Log::DEBUG)) {
    Log::debug("checking parameters after mutation\n");
    g->get_mu_sigma(g->initial_parameters, mu, sigma);
  }

  g->best_parameters.clear();
  return g;
}

RNN_Genome *GenomeOperators::crossover(RNN_Genome *more_fit, RNN_Genome *less_fit) {
  Log::debug("generating new genome by crossover!\n");
  Log::debug("more_fit->island: %d, less_fit->island: %d\n", more_fit->get_group_id(), less_fit->get_group_id());
  Log::debug("more_fit->number_inputs: %d, less_fit->number_inputs: %d\n", more_fit->get_number_inputs(),
             less_fit->get_number_inputs());

  for (uint32_t i = 0; i < more_fit->nodes.size(); i++) {
    Log::debug(
        "more_fit node[%d], in: %lld, depth: %lf, layer_type: %d, node_type: "
        "%d, "
        "reachable: %d, enabled: %d\n",
        i, more_fit->nodes[i]->inon, more_fit->nodes[i]->get_depth(), more_fit->nodes[i]->get_layer_type(),
        more_fit->nodes[i]->get_node_type(), more_fit->nodes[i]->is_reachable(), more_fit->nodes[i]->is_enabled());
  }

  for (uint32_t i = 0; i < less_fit->nodes.size(); i++) {
    Log::debug(
        "less_fit node[%d], in: %lld, depth: %lf, layer_type: %d, node_type: "
        "%d, "
        "reachable: %d, enabled: %d\n",
        i, less_fit->nodes[i]->inon, less_fit->nodes[i]->get_depth(), less_fit->nodes[i]->get_layer_type(),
        less_fit->nodes[i]->get_node_type(), less_fit->nodes[i]->is_reachable(), less_fit->nodes[i]->is_enabled());
  }

  double _mu, _sigma;
  Log::debug("getting more_fit mu/sigma!\n");
  if (more_fit->best_parameters.size() == 0) {
    more_fit->set_weights(more_fit->initial_parameters);
    more_fit->get_mu_sigma(more_fit->initial_parameters, _mu, _sigma);
  } else {
    more_fit->set_weights(more_fit->best_parameters);
    more_fit->get_mu_sigma(more_fit->best_parameters, _mu, _sigma);
  }

  Log::debug("getting less_fit mu/sigma!\n");
  if (less_fit->best_parameters.size() == 0) {
    less_fit->set_weights(less_fit->initial_parameters);
    less_fit->get_mu_sigma(less_fit->initial_parameters, _mu, _sigma);
  } else {
    less_fit->set_weights(less_fit->best_parameters);
    less_fit->get_mu_sigma(less_fit->best_parameters, _mu, _sigma);
  }

  // nodes are copied in the attempt_node_insert_function
  vector<RNN_Node_Interface *> child_nodes;
  vector<RNN_Edge *> child_edges;
  vector<RNN_Recurrent_Edge *> child_recurrent_edges;

  // edges are not sorted in order of innovation number, they need to be
  vector<RNN_Edge *> more_fit_edges = more_fit->edges;
  vector<RNN_Edge *> less_fit_edges = less_fit->edges;

  sort(more_fit_edges.begin(), more_fit_edges.end(), sort_RNN_Edges_by_inon());
  sort(less_fit_edges.begin(), less_fit_edges.end(), sort_RNN_Edges_by_inon());

  Log::debug("\tmore_fit innovation numbers AFTER SORT:\n");
  for (int32_t i = 0; i < (int32_t) more_fit_edges.size(); i++) { Log::debug("\t\t%d\n", more_fit_edges[i]->inon); }
  Log::debug("\tless_fit innovation numbers AFTER SORT:\n");
  for (int32_t i = 0; i < (int32_t) less_fit_edges.size(); i++) { Log::debug("\t\t%d\n", less_fit_edges[i]->inon); }

  vector<RNN_Recurrent_Edge *> more_fit_recurrent_edges = more_fit->recurrent_edges;
  vector<RNN_Recurrent_Edge *> less_fit_recurrent_edges = less_fit->recurrent_edges;

  sort(more_fit_recurrent_edges.begin(), more_fit_recurrent_edges.end(), sort_RNN_Recurrent_Edges_by_inon());
  sort(less_fit_recurrent_edges.begin(), less_fit_recurrent_edges.end(), sort_RNN_Recurrent_Edges_by_inon());

  int32_t more_fit_position = 0;
  int32_t less_fit_position = 0;

  while (more_fit_position < (int32_t) more_fit_edges.size() && less_fit_position < (int32_t) less_fit_edges.size()) {
    RNN_Edge *more_fit_edge = more_fit_edges[more_fit_position];
    RNN_Edge *less_fit_edge = less_fit_edges[less_fit_position];

    auto more_fit_inon = more_fit_edge->inon;
    auto less_fit_inon = less_fit_edge->inon;

    if (more_fit_inon == less_fit_inon) {
      attempt_edge_insert(child_edges, child_nodes, more_fit_edge, less_fit_edge, true);

      more_fit_position++;
      less_fit_position++;
    } else if (more_fit_inon < less_fit_inon) {
      bool set_enabled = rng_0_1(generator) < more_fit_crossover_rate;
      if (more_fit_edge->is_reachable())
        set_enabled = true;
      else
        set_enabled = false;

      attempt_edge_insert(child_edges, child_nodes, more_fit_edge, NULL, set_enabled);

      more_fit_position++;
    } else {
      bool set_enabled = rng_0_1(generator) < less_fit_crossover_rate;
      if (less_fit_edge->is_reachable())
        set_enabled = true;
      else
        set_enabled = false;

      attempt_edge_insert(child_edges, child_nodes, less_fit_edge, NULL, set_enabled);

      less_fit_position++;
    }
  }

  while (more_fit_position < (int32_t) more_fit_edges.size()) {
    RNN_Edge *more_fit_edge = more_fit_edges[more_fit_position];

    bool set_enabled = rng_0_1(generator) < more_fit_crossover_rate;
    if (more_fit_edge->is_reachable())
      set_enabled = true;
    else
      set_enabled = false;

    attempt_edge_insert(child_edges, child_nodes, more_fit_edge, NULL, set_enabled);

    more_fit_position++;
  }

  while (less_fit_position < (int32_t) less_fit_edges.size()) {
    RNN_Edge *less_fit_edge = less_fit_edges[less_fit_position];

    bool set_enabled = rng_0_1(generator) < less_fit_crossover_rate;
    if (less_fit_edge->is_reachable())
      set_enabled = true;
    else
      set_enabled = false;

    attempt_edge_insert(child_edges, child_nodes, less_fit_edge, NULL, set_enabled);

    less_fit_position++;
  }

  // do the same for recurrent_edges
  more_fit_position = 0;
  less_fit_position = 0;

  while (more_fit_position < (int32_t) more_fit_recurrent_edges.size() &&
         less_fit_position < (int32_t) less_fit_recurrent_edges.size()) {
    RNN_Recurrent_Edge *more_fit_recurrent_edge = more_fit_recurrent_edges[more_fit_position];
    RNN_Recurrent_Edge *less_fit_recurrent_edge = less_fit_recurrent_edges[less_fit_position];

    edge_inon more_fit_inon = more_fit_recurrent_edge->inon;
    edge_inon less_fit_inon = less_fit_recurrent_edge->inon;

    if (more_fit_inon == less_fit_inon) {
      // do weight crossover
      attempt_recurrent_edge_insert(child_recurrent_edges, child_nodes, more_fit_recurrent_edge,
                                    less_fit_recurrent_edge, true);

      more_fit_position++;
      less_fit_position++;
    } else if (more_fit_inon < less_fit_inon) {
      bool set_enabled = rng_0_1(generator) < more_fit_crossover_rate;
      if (more_fit_recurrent_edge->is_reachable())
        set_enabled = true;
      else
        set_enabled = false;

      attempt_recurrent_edge_insert(child_recurrent_edges, child_nodes, more_fit_recurrent_edge, NULL, set_enabled);

      more_fit_position++;
    } else {
      bool set_enabled = rng_0_1(generator) < less_fit_crossover_rate;
      if (less_fit_recurrent_edge->is_reachable())
        set_enabled = true;
      else
        set_enabled = false;

      attempt_recurrent_edge_insert(child_recurrent_edges, child_nodes, less_fit_recurrent_edge, NULL, set_enabled);

      less_fit_position++;
    }
  }

  while (more_fit_position < (int32_t) more_fit_recurrent_edges.size()) {
    RNN_Recurrent_Edge *more_fit_recurrent_edge = more_fit_recurrent_edges[more_fit_position];

    bool set_enabled = rng_0_1(generator) < more_fit_crossover_rate;
    if (more_fit_recurrent_edge->is_reachable())
      set_enabled = true;
    else
      set_enabled = false;

    attempt_recurrent_edge_insert(child_recurrent_edges, child_nodes, more_fit_recurrent_edge, NULL, set_enabled);

    more_fit_position++;
  }

  while (less_fit_position < (int32_t) less_fit_recurrent_edges.size()) {
    RNN_Recurrent_Edge *less_fit_recurrent_edge = less_fit_recurrent_edges[less_fit_position];

    bool set_enabled = rng_0_1(generator) < less_fit_crossover_rate;
    if (less_fit_recurrent_edge->is_reachable())
      set_enabled = true;
    else
      set_enabled = false;

    attempt_recurrent_edge_insert(child_recurrent_edges, child_nodes, less_fit_recurrent_edge, NULL, set_enabled);

    less_fit_position++;
  }

  sort(child_nodes.begin(), child_nodes.end(), sort_RNN_Nodes_by_depth());
  sort(child_edges.begin(), child_edges.end(), sort_RNN_Edges_by_depth());
  sort(child_recurrent_edges.begin(), child_recurrent_edges.end(), sort_RNN_Recurrent_Edges_by_depth());
  RNN_Genome *child = new RNN_Genome(child_nodes, child_edges, child_recurrent_edges, training_parameters,
                                     weight_initialize, weight_inheritance, mutated_component_weight);
  child->set_parameter_names(dataset_meta.input_parameter_names, dataset_meta.output_parameter_names);
  child->set_normalize_bounds(dataset_meta.normalize_type, dataset_meta.normalize_mins, dataset_meta.normalize_maxs,
                              dataset_meta.normalize_avgs, dataset_meta.normalize_std_devs);

  if (more_fit->get_group_id() == less_fit->get_group_id()) {
    child->set_generated_by("crossover");
  } else {
    child->set_generated_by("island_crossover");
  }

  double mu, sigma;

  vector<double> new_parameters;

  // if weight_inheritance is same, all the weights of the child genome would
  // be initialized as weight_initialize method
  if (weight_inheritance == weight_initialize) {
    Log::debug(
        "weight inheritance at crossover method is %s, setting weights "
        "to %s randomly \n",
        WEIGHT_TYPES_STRING[weight_inheritance].c_str(), WEIGHT_TYPES_STRING[weight_inheritance].c_str());
    child->initialize_randomly();
  }

  child->get_weights(new_parameters);
  Log::debug("getting mu/sigma before assign reachability\n");
  child->get_mu_sigma(new_parameters, mu, sigma);

  child->assign_reachability();

  // reset the genomes statistics (as these carry over on copy)
  child->best_validation_mse = EXAMM_MAX_DOUBLE;
  child->best_validation_mae = EXAMM_MAX_DOUBLE;

  // get the new set of parameters (as new paramters may have been
  // added duriung mutatino) and set them to the initial parameters
  // for epigenetic_initialization
  child->get_weights(new_parameters);
  child->initial_parameters = new_parameters;

  Log::debug("checking parameters after crossover\n");
  child->get_mu_sigma(child->initial_parameters, mu, sigma);

  child->best_parameters.clear();

  return child;
}

int32_t GenomeOperators::get_random_n_parents_inter() {
  int32_t dif = n_parents_inter_range.second - n_parents_inter_range.first + 1;
  return n_parents_inter_range.first + rng_0_1(generator) * dif;
}

int32_t GenomeOperators::get_random_n_parents_intra() {
  int32_t dif = n_parents_intra_range.second - n_parents_intra_range.first + 1;
  return n_parents_intra_range.first + rng_0_1(generator) * dif;
}

RNN_Genome *GenomeOperators::ncrossover(vector<const RNN_Genome *> &parents) {
  Log::debug("performing ncrossover with %d parents\n", parents.size());

  if (parents.size() == 0) {
    Log::fatal(
        "Cannot perform crossover with 0 parents - this shouldn't happen "
        "so the program will exit\n");
    exit(1);
  }

  std::sort(parents.begin(), parents.end(),
            [](const RNN_Genome *a, const RNN_Genome *b) { return a->get_fitness() < b->get_fitness(); });

  // nodes are copied in the attempt_node_insert_function
  // maps innovation number to a bool representing whether it should be enabled
  // in the child shared with rec edges too since there should be no overlap.
  unordered_map<inon_type<RNN_Edge>, bool> edge_enabled;

  unordered_map<inon_type<RNN_Node_Interface>, bool> node_enabled;

  // Maps edge innovation numbers to a vector of weights for that edge.
  // Each element in the weight vector is from a different genome.
  // Recurrent edges and regular edges will never have the same innovation
  // number so we don't have to have a separate vector.
  //
  // The order the vectors are populated ensures that the first element always
  // corresponds to the best weight. The parent rank is also stored witht he
  // weight, as it may be used to modify the crossover.
  unordered_map<edge_inon, vector<pair<int32_t, double> > > edge_weights;
  unordered_map<node_inon, vector<pair<int32_t, vector<double> > > > node_weights;

  // Maps innovation number to edge. This will be populated with every edge that
  // appears in the parents, we need a reference to it so we can insert them
  // after we figure out which edges should be enabled.
  unordered_map<edge_inon, RNN_Edge *> edge_map;
  unordered_map<edge_inon, RNN_Recurrent_Edge *> rec_edge_map;
  unordered_map<node_inon, RNN_Node_Interface *> node_map;

  vector<RNN_Node_Interface *> child_nodes;
  vector<RNN_Edge *> child_edges;
  vector<RNN_Recurrent_Edge *> child_recurrent_edges;

  for (auto i = 0; i < parents.size(); i++) {
    const RNN_Genome *p = parents[i];

    double probability = 1.0 / ((double) (1 << i));

    Log::debug("Copying nodes from parent %d\n", i);
    for (auto j = 0; j < p->nodes.size(); j++) {
      auto node = p->nodes[j];
      // Copy of the best node
      node_map.insert({node->inon, node});
      if (node_map.count(node->inon) == 0) node_map[node->inon] = node;

      if (node->enabled) {
        // Only consider weights from enabled nodes.
        vector<double> weights(node->get_number_weights());
        node->get_weights(weights);
        node_weights[node->inon].push_back({(int32_t) i, move(weights)});
        if (!node_enabled[node->inon] && rng_0_1(generator) < probability) node_enabled[node->inon] = true;
      }
    }
    Log::debug("Done copying nodes from parent %d\n", i);

    Log::debug("Copying edges from parent %d\n", i);
    for (auto j = 0; j < p->edges.size(); j++) {
      auto e = p->edges[j];
      edge_map.insert({e->inon, e});

      Log::debug("edge_enabled %d\n", e->inon);
      if (e->enabled) {
        if (e->input_node->enabled && e->output_node->enabled)
          edge_weights[e->inon].push_back({(int32_t) i, e->weight});

        if (!edge_enabled[e->inon] && rng_0_1(generator) < probability) edge_enabled[e->inon] = true;
      }
    }
    Log::debug("Done copying edges from parent %d\n", i);

    Log::debug("hmmm \n");
    Log::debug(" o %d\n", p->recurrent_edges.size());
    Log::debug("hmmm \n");
    Log::debug("Copying edges from parent %d\n", i);
    for (auto j = 0; j < p->recurrent_edges.size(); j++) {
      auto e = p->recurrent_edges[j];
      rec_edge_map.insert({e->inon, e});

      Log::debug("edge_enabled %d\n", e->inon);
      if (e->enabled) {
        if (e->input_node->enabled && e->output_node->enabled)
          edge_weights[e->inon].push_back({(int32_t) i, e->weight});

        if (!edge_enabled[e->inon] && rng_0_1(generator) < probability) edge_enabled[e->inon] = true;
      }
    }
    Log::debug("Done copying rec dges from parent %d\n", i);
  }

  Log::debug("Done selecting components\n");

  auto n_parents = parents.size();
  // Get appropriate mass (weight in weighted sum) for the given position.
  // The position corresponds to the index of the parent a particular weight
  // comes from
  auto get_centroid_mass = [n_parents](int32_t position) { return 1.0; };

  // Define crossover function for weights. Based on simplex method with
  // configurable get_centroid_mass function. These functions are basically
  // taking a weighted sum to calculate the centroid. The weight comes from
  // get_centroid_mass
  auto weight_crossover_1d = [this, get_centroid_mass](vector<pair<int32_t, double> > &weights) {
    // Sum of first n - 1 elements (weights length == n)
    double centroid_sum = 0.0;
    // sum of the "mass" prescribed to each point
    double centroid_mass_sum = 0.0;

    for (uint32_t i = 1; i < weights.size(); i += 1) {
      double weight = get_centroid_mass(weights[i].first);
      centroid_sum += weights[i].second * weight;
      centroid_mass_sum += weight;
    }
    double centroid = centroid_sum / centroid_mass_sum;
    double crossover_weight = rng_crossover_weight(generator);

    double weight = crossover_weight * (centroid - weights.back().second) + centroid;
    return weight;
  };

  auto weight_crossover_2d = [this, get_centroid_mass](vector<pair<int32_t, vector<double> > > &weights,
                                                       vector<double> &new_weights) {
    new_weights.clear();

    auto n_weights = weights[0].second.size();
    double centroid_sum[n_weights];
    memset(centroid_sum, 0, sizeof(centroid_sum));
    double centroid_mass_sum = 0.0;

    for (auto i = 1; i < weights.size(); i += 1) {
      double centroid_weight = get_centroid_mass(weights[i].first);
      vector<double> &weights_i = weights[i].second;
      for (auto j = 0; j < n_weights; j++) { centroid_sum[j] += weights_i[j] * centroid_weight; }

      centroid_mass_sum += centroid_weight;
    }

    for (auto i = 0; i < n_weights; i++) { centroid_sum[i] /= centroid_mass_sum; }

    const double *const centroid = centroid_sum;
    vector<double> &worst_weights = weights.back().second;
    double crossover_weight = rng_crossover_weight(generator);

    for (int i = 0; i < n_weights; i++) {
      new_weights.push_back(crossover_weight * (centroid[i] - worst_weights[i]) + centroid[i]);
    }
  };

  // So now we have a list of every edge and node that should be enabled,
  // possible weights for them, and a copy of every component. Time to start
  // adding everything. First, add all of the nodes. This means copying every
  // node. Since we will need to get nodes by ID later for edge insertion, we
  // will create a map
  vector<double> new_weights;
  for (auto it = node_map.begin(); it != node_map.end(); it++) {
    auto copy = it->second->copy();
    it->second = copy;
    insert_node_by_depth(child_nodes, copy);

    copy->enabled = node_enabled[it->first];

    vector<pair<int32_t, vector<double> > > &weights = node_weights[it->first];
    if (weights.size() == 0) {
    } else if (weights.size() == 1) {
      copy->set_weights(weights[0].second);
    } else {
      weight_crossover_2d(weights, new_weights);
      copy->set_weights(new_weights);
    }

    vector<double> parameters;
    it->second->get_weights(parameters);
    vector<double> copy_parameters;
    copy->get_weights(copy_parameters);

    // Update the map with the copy, so we can use this map for grabbing the
    // appropriate nodes when copying edges.
    node_map[it->second->inon] = copy;
  }

  for (auto it = edge_map.begin(); it != edge_map.end(); it++) {
    auto copy = it->second->copy(node_map);

    Log::debug("edge_enabled %d\n", it->first);
    copy->enabled = edge_enabled[it->first];

    vector<pair<int32_t, double> > &weights = edge_weights[it->first];
    switch (weights.size()) {
      case 1:
        copy->weight = weights[0].second;
      case 0:
        break;
      default:
        copy->weight = weight_crossover_1d(weights);
    }

    insert_edge_by_depth(child_edges, copy);
  }

  for (auto it = rec_edge_map.begin(); it != rec_edge_map.end(); it++) {
    auto copy = it->second->copy(node_map);

    Log::debug("edge_enabled %d\n", it->first);
    copy->enabled = edge_enabled[it->first];

    vector<pair<int32_t, double> > &weights = edge_weights[it->first];
    switch (weights.size()) {
      case 1:
        copy->weight = weights[0].second;
      case 0:
        break;
      default:
        copy->weight = weight_crossover_1d(weights);
    }

    insert_rec_edge_by_depth(child_recurrent_edges, copy);
  }

  Log::debug("OKAY\n");
  RNN_Genome *child = new RNN_Genome(child_nodes, child_edges, child_recurrent_edges, training_parameters,
                                     weight_initialize, weight_inheritance, mutated_component_weight);
  Log::debug("OKAY\n");
  child->set_parameter_names(dataset_meta.input_parameter_names, dataset_meta.output_parameter_names);
  Log::debug("OKAY\n");
  child->set_normalize_bounds(dataset_meta.normalize_type, dataset_meta.normalize_mins, dataset_meta.normalize_maxs,
                              dataset_meta.normalize_avgs, dataset_meta.normalize_std_devs);
  Log::debug("OKAY\n");

  bool intra_island_crossover = true;
  for (int i = 0; i < parents.size() - 1 && intra_island_crossover; i++) {
    intra_island_crossover = parents[i]->group_id == parents[i + 1]->group_id;
  }

  child->clear_generated_by();
  if (intra_island_crossover) {
    child->set_generated_by("crossover");
  } else {
    child->set_generated_by("island_crossover");
  }

  double mu, sigma;

  vector<double> new_parameters;

  // if weight_inheritance is same, all the weights of the child genome would
  // be initialized as weight_initialize method
  if (weight_inheritance == weight_initialize) {
    Log::debug(
        "weight inheritance at crossover method is %s, setting weights "
        "to %s randomly \n",
        WEIGHT_TYPES_STRING[weight_inheritance].c_str(), WEIGHT_TYPES_STRING[weight_inheritance].c_str());
    child->initialize_randomly();
  }

  child->get_weights(new_parameters);
  Log::debug("getting mu/sigma before assign reachability\n");
  child->get_mu_sigma(new_parameters, mu, sigma);

  child->assign_reachability();

  // reset the genomes statistics (as these carry over on copy)
  child->best_validation_mse = EXAMM_MAX_DOUBLE;
  child->best_validation_mae = EXAMM_MAX_DOUBLE;

  // get the new set of parameters (as new paramters may have been
  // added duriung mutatino) and set them to the initial parameters
  // for epigenetic_initialization
  child->initial_parameters = new_parameters;

  Log::debug("checking parameters after crossover\n");
  child->get_mu_sigma(child->initial_parameters, mu, sigma);

  child->best_parameters.clear();

  Log::info("Child with %d nodes, %d edges, %d redges\n", child->nodes.size(), child->edges.size(),
            child->recurrent_edges.size());

  return child;
}

void GenomeOperators::attempt_node_insert(vector<RNN_Node_Interface *> &child_nodes, const RNN_Node_Interface *node,
                                          const vector<double> &new_weights) {
  for (int32_t i = 0; i < (int32_t) child_nodes.size(); i++) {
    if (child_nodes[i]->inon == node->inon) return;
  }

  RNN_Node_Interface *node_copy = node->copy();
  node_copy->set_weights(new_weights);

  insert_node_by_depth(child_nodes, node_copy);
}

void GenomeOperators::attempt_edge_insert(vector<RNN_Edge *> &child_edges, vector<RNN_Node_Interface *> &child_nodes,
                                          RNN_Edge *edge, RNN_Edge *second_edge, bool set_enabled) {
  for (int32_t i = 0; i < (int32_t) child_edges.size(); i++) {
    if (child_edges[i]->inon == edge->inon) {
      Log::fatal(
          "ERROR in crossover! trying to push an edge with "
          "innovation_number: %lld and it already exists in the vector!\n",
          edge->inon);

      Log::fatal("vector innovation numbers: ");
      for (int32_t i = 0; i < (int32_t) child_edges.size(); i++) { Log::fatal("\t%d", child_edges[i]->inon); }

      Log::fatal("This should never happen!\n");
      exit(1);

      return;
    } else if (child_edges[i]->input_inon == edge->input_inon && child_edges[i]->output_inon == edge->output_inon) {
      Log::debug(
          "Not inserting edge in crossover operation as there was "
          "already an "
          "edge with the same input and output innovation numbers!\n");
      return;
    }
  }

  vector<double> new_input_weights, new_output_weights;
  double new_weight = 0.0;
  if (second_edge != NULL) {
    double crossover_value = rng_crossover_weight(generator);
    new_weight = crossover_value * (second_edge->weight - edge->weight) + edge->weight;

    Log::trace(
        "EDGE WEIGHT CROSSOVER :: better: %lf, worse: %lf, "
        "crossover_value: %lf, new_weight: %lf\n",
        edge->weight, second_edge->weight, crossover_value, new_weight);

    vector<double> input_weights1, input_weights2, output_weights1, output_weights2;
    edge->get_input_node()->get_weights(input_weights1);
    edge->get_output_node()->get_weights(output_weights1);

    second_edge->get_input_node()->get_weights(input_weights2);
    second_edge->get_output_node()->get_weights(output_weights2);

    new_input_weights.resize(input_weights1.size());
    new_output_weights.resize(output_weights1.size());

    // can check to see if input weights lengths are same
    // can check to see if output weights lengths are same

    for (int32_t i = 0; i < (int32_t) new_input_weights.size(); i++) {
      new_input_weights[i] = crossover_value * (input_weights2[i] - input_weights1[i]) + input_weights1[i];
      Log::trace("\tnew input weights[%d]: %lf\n", i, new_input_weights[i]);
    }

    for (int32_t i = 0; i < (int32_t) new_output_weights.size(); i++) {
      new_output_weights[i] = crossover_value * (output_weights2[i] - output_weights1[i]) + output_weights1[i];
      Log::trace("\tnew output weights[%d]: %lf\n", i, new_output_weights[i]);
    }

  } else {
    new_weight = edge->weight;
    edge->get_input_node()->get_weights(new_input_weights);
    edge->get_output_node()->get_weights(new_output_weights);
  }

  attempt_node_insert(child_nodes, edge->get_input_node(), new_input_weights);
  attempt_node_insert(child_nodes, edge->get_output_node(), new_output_weights);

  RNN_Edge *edge_copy = edge->copy(child_nodes);

  edge_copy->enabled = set_enabled;
  edge_copy->weight = new_weight;

  // edges have already been copied
  insert_edge_by_depth(child_edges, edge_copy);
}

void GenomeOperators::attempt_recurrent_edge_insert(vector<RNN_Recurrent_Edge *> &child_recurrent_edges,
                                                    vector<RNN_Node_Interface *> &child_nodes,
                                                    RNN_Recurrent_Edge *recurrent_edge, RNN_Recurrent_Edge *second_edge,
                                                    bool set_enabled) {
  for (int32_t i = 0; i < (int32_t) child_recurrent_edges.size(); i++) {
    if (child_recurrent_edges[i]->inon == recurrent_edge->inon) {
      Log::fatal(
          "ERROR in crossover! trying to push an recurrent_edge with "
          "innovation_number: %lld  and it already exists in the vector!\n",
          recurrent_edge->inon);
      Log::fatal("vector innovation numbers:\n");
      for (int32_t i = 0; i < (int32_t) child_recurrent_edges.size(); i++) {
        Log::fatal("\t %d", child_recurrent_edges[i]->inon);
      }

      Log::fatal("This should never happen!\n");
      exit(1);

      return;
    } else if (child_recurrent_edges[i]->input_inon == recurrent_edge->input_inon &&
               child_recurrent_edges[i]->output_inon == recurrent_edge->output_inon) {
      Log::debug(
          "Not inserting recurrent_edge in crossover operation as there "
          "was already an recurrent_edge with the same input and output "
          "innovation numbers!\n");
      return;
    }
  }

  vector<double> new_input_weights, new_output_weights;
  double new_weight = 0.0;
  if (second_edge != NULL) {
    double crossover_value = rng_crossover_weight(generator);
    new_weight = crossover_value * (second_edge->weight - recurrent_edge->weight) + recurrent_edge->weight;

    Log::debug(
        "RECURRENT EDGE WEIGHT CROSSOVER :: better: %lf, worse: %lf, "
        "crossover_value: %lf, new_weight: %lf\n",
        recurrent_edge->weight, second_edge->weight, crossover_value, new_weight);

    vector<double> input_weights1, input_weights2, output_weights1, output_weights2;
    recurrent_edge->get_input_node()->get_weights(input_weights1);
    recurrent_edge->get_output_node()->get_weights(output_weights1);

    second_edge->get_input_node()->get_weights(input_weights2);
    second_edge->get_output_node()->get_weights(output_weights2);

    new_input_weights.resize(input_weights1.size());
    new_output_weights.resize(output_weights1.size());

    for (int32_t i = 0; i < (int32_t) new_input_weights.size(); i++) {
      new_input_weights[i] = crossover_value * (input_weights2[i] - input_weights1[i]) + input_weights1[i];
      Log::trace("\tnew input weights[%d]: %lf\n", i, new_input_weights[i]);
    }

    for (int32_t i = 0; i < (int32_t) new_output_weights.size(); i++) {
      new_output_weights[i] = crossover_value * (output_weights2[i] - output_weights1[i]) + output_weights1[i];
      Log::trace("\tnew output weights[%d]: %lf\n", i, new_output_weights[i]);
    }

  } else {
    new_weight = recurrent_edge->weight;
    recurrent_edge->get_input_node()->get_weights(new_input_weights);
    recurrent_edge->get_output_node()->get_weights(new_output_weights);
  }

  attempt_node_insert(child_nodes, recurrent_edge->get_input_node(), new_input_weights);
  attempt_node_insert(child_nodes, recurrent_edge->get_output_node(), new_output_weights);

  RNN_Recurrent_Edge *recurrent_edge_copy = recurrent_edge->copy(child_nodes);

  recurrent_edge_copy->enabled = set_enabled;
  recurrent_edge_copy->weight = new_weight;

  // recurrent_edges have already been copied
  insert_rec_edge_by_depth(child_recurrent_edges, recurrent_edge_copy);
}

const vector<rnn_node_type> &GenomeOperators::get_possible_node_types() { return possible_node_types; }

int GenomeOperators::get_number_inputs() { return number_inputs; }
int GenomeOperators::get_number_outputs() { return number_outputs; }
