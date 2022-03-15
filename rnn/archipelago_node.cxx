#ifndef ARCHIPELAGO_NODE_CXX
#define ARCHIPELAGO_NODE_CXX

#include <algorithm>
using std::min;

#include <string_view>
using std::basic_string_view;

#include <fstream>
using std::ios_base;

#include <sstream>
using std::ostringstream;

#include <memory>
using std::make_shared;
using std::make_unique;

#include <cassert>

#include "rnn/archipelago_node.hxx"
#include "rnn_genome.hxx"

ArchipelagoIO::~ArchipelagoIO() {}

///
/// ArchipelagoNode
///

template <Derived<ArchipelagoIO> IO>
ArchipelagoNode<IO>::ArchipelagoNode(node_index_type node_id, ArchipelagoConfig &config, IO &io)
    : node_id(node_id), config(config), io(io), role(config.node_roles[node_id]) {
  random_device rd;
  generator = mt19937_64(rd());
}

template <Derived<ArchipelagoIO> IO>
ArchipelagoNode<IO>::~ArchipelagoNode() {}

template <Derived<ArchipelagoIO> IO>
void ArchipelagoNode<IO>::populate_relationships() {
  for (node_index_type i = 0; i < (int) config.node_roles.size(); i++) {
    if (i == node_id) continue;
    if (!(config.connections[node_id][i] || config.connections[i][node_id])) continue;

    // Relation this node has to the other node.
    // So if this node is a parent to the other, the other node is a child.
    Log::info("%d relationship with %d = %d\n", node_id, i, relationship_with(i));
    switch (relationship_with(i)) {
      case PARENT:
        children.push_back(i);
        break;
      case NEIGHBOR:
        neighbors.push_back(i);
        break;
      case CHILD:
        parents.push_back(i);
        break;
      case NONE:
        break;
      case INVALID:
        Log::fatal("Detected invalid relationship between nodes %d and %d\n", (int) node_id, (int) i);
        Log::fatal("Node role %d should not be connected to %d\n", role, config.node_roles[i]);
        exit(1);
    }
  }
}

///
/// ArchipelagoIslandCluster
///

template <Derived<ArchipelagoIO> IO>
ArchipelagoIslandCluster<IO>::ArchipelagoIslandCluster(node_index_type node_id, ArchipelagoConfig &config, IO &io,
                                                       uint32_t number_of_islands, uint32_t max_island_size,
                                                       shared_ptr<const RNN_Genome> seed_genome,
                                                       pair<uint32_t, uint32_t> n_parents_foreign_range,
                                                       GenomeOperators &go)
    : ArchipelagoNode<IO>(node_id, config, io),
      IslandSpeciationStrategy(number_of_islands, max_island_size, move(seed_genome), "", "", -1, 0, 0, true,
                               std::nullopt, go),
      max_foreign_genomes(max_island_size),
      n_parents_foreign_range(n_parents_foreign_range) {
  this->populate_relationships();
  assert(config.node_roles[node_id] == node_role::ISLANDS);
  assert(parents.size() >= 1);
  assert(children.size() >= 1);
}

template <Derived<ArchipelagoIO> IO>
ArchipelagoIslandCluster<IO>::~ArchipelagoIslandCluster() {}

template <Derived<ArchipelagoIO> IO>
unique_ptr<WorkMsg> ArchipelagoIslandCluster<IO>::generate_work_for_filled_island(Island &island) {
  // if we haven't filled ALL of the island populations yet, only use mutation
  // otherwise do mutation at %, crossover at %, and island crossover at %
  double r = rng_0_1(generator);
  if (!islands_full() || r < mutation_rate) {
    Log::info("performing mutation\n");
    shared_ptr<const RNN_Genome> genome = island.get_random_genome(generator);
    return make_unique<WorkMsg>(move(genome), genome_operators.get_random_n_mutations());
  } else {
    vector<shared_ptr<const RNN_Genome>> parents;

    if (rng_0_1(generator) < local_crossover_p || foreign_genomes.size() == 0) {
      if (r < intra_island_crossover_rate || number_of_islands == 1) {
        Log::info("performing intra-island crossover\n");
        island.get_n_random_genomes(generator, genome_operators.get_random_n_parents_intra(), parents);
      } else {
        Log::info("performing inter-island crossover\n");
        int32_t n = genome_operators.get_random_n_parents_inter();
        island.get_n_random_genomes(generator, 1, parents);
        int32_t other_island = rng_0_1(generator) * (number_of_islands - 1);
        if (other_island >= generation_island) other_island += 1;
        islands[other_island].get_n_random_genomes(generator, n - 1, parents);
      }
    } else {
      // Foreign crossover
      Log::info("performing foreign-cluster crossover\n");
      island.get_n_random_genomes(generator, 1, parents);
      uint32_t n = get_random_n_parents_foreign();
      parents.push_back(island.get_random_genome(generator));
      get_n_random_foreign_genomes(n, parents);
    }
    return make_unique<WorkMsg>(parents);
  }
}

template <Derived<ArchipelagoIO> IO>
uint32_t ArchipelagoIslandCluster<IO>::get_random_n_parents_foreign() {
  uint32_t dif = n_parents_foreign_range.second - n_parents_foreign_range.first + 1;
  uint32_t n = n_parents_foreign_range.first + rng_0_1(generator) * dif;
  return min((uint32_t) foreign_genomes.size(), n);
}

template <Derived<ArchipelagoIO> IO>
void ArchipelagoIslandCluster<IO>::get_n_random_foreign_genomes(uint32_t n,
                                                                vector<shared_ptr<const RNN_Genome>> &genomes) {
  vector<int> indices(foreign_genomes.size());
  std::iota(indices.begin(), indices.end(), 0);

  fisher_yates_shuffle(generator, indices);

  indices.resize(n);
  for (int i = 0; i < (int) indices.size(); i++) { genomes.emplace_back(foreign_genomes[i]); }
}

template <Derived<ArchipelagoIO> IO>
void ArchipelagoIslandCluster<IO>::process_msg(unique_ptr<Msg> msg, node_index_type src) {
  switch (msg->get_msg_ty()) {
    case Msg::WORK:
    case Msg::MPI_INIT:
    case Msg::TERMINATE:
    case Msg::EVAL_ACCOUNTING:
      Log::info("Island cluster should not recieve message of type %d\n", msg->get_msg_ty());
      exit(1);
      break;

    case Msg::REQUEST:
      assert(config.node_roles[src] == node_role::WORKERS);
      assert(config.connections[node_id][src]);
      process_request(dynamic_cast<RequestMsg *>(msg.get()), src);
      break;

    case Msg::RESULT:
      process_result(dynamic_cast<ResultMsg *>(msg.get()), src);
      break;

    case Msg::GENOME_SHARE:
      process_genome_share(dynamic_cast<GenomeShareMsg *>(msg.get()), src);  // FALLTHROUGH
                                                                             // case Msg::EVAL_ACCOUNTING:
      //   process_eval_accounting(dynamic_cast<EvalAccountingMsg *>(msg.get()), src);
      break;
  }
}

template <Derived<ArchipelagoIO> IO>
void ArchipelagoIslandCluster<IO>::process_request(RequestMsg *, node_index_type src) {
  Log::info("Island cluster handling request from %d\n", src);
  unique_ptr<WorkMsg> work = generate_work();
  Log::info("Work %p\n", work.get());
  io.send_msg_to(work.get(), src);
}

template <Derived<ArchipelagoIO> IO>
void ArchipelagoIslandCluster<IO>::process_result(ResultMsg *result, node_index_type src) {
  Log::info("Island cluster handling result from %d\n", src);
  unique_ptr<RNN_Genome> g = result->get_genome();
  auto [insert_position, _g] = insert_genome(move(g));

  Log::info("Insert position %d\n", insert_position);

  unrecorded_genome_count += 1;
  // New global best
  if (insert_position == 0) {
    shared_ptr<const RNN_Genome> shared = get_global_best_genome();
    this->share_genome(move(shared));
  }

  this->account_genome_evals(1);

  // TODO: Send another job to the worker, reduces the n of messages that need to be passed.
}

template <Derived<ArchipelagoIO> IO>
void ArchipelagoIslandCluster<IO>::process_genome_share(GenomeShareMsg *share, node_index_type src) {
  Log::info("Island cluster handling genome share from %d\n", src);
  unique_ptr<RNN_Genome> g = share->get_genome();
  shared_ptr<const RNN_Genome> shared = shared_ptr<const RNN_Genome>(g.release());
  foreign_genomes.emplace_front(move(shared));

  if (foreign_genomes.size() > max_foreign_genomes) foreign_genomes.pop_back();
}

template <Derived<ArchipelagoIO> IO>
typename ArchipelagoNode<IO>::node_relationship ArchipelagoIslandCluster<IO>::relationship_with(node_index_type other) {
  switch (config.node_roles[other]) {
    case node_role::MANAGERS:
      assert(config.connections[node_id][other]);
      return ArchipelagoNode<IO>::CHILD;
    case node_role::MASTER:
      assert(config.connections[node_id][other]);
      return ArchipelagoNode<IO>::CHILD;
    case node_role::WORKERS:
      assert(config.connections[other][node_id] || config.connections[node_id][other]);
      return ArchipelagoNode<IO>::PARENT;
    case node_role::ISLANDS:
      if (config.connections[node_id][other])
        return ArchipelagoNode<IO>::NEIGHBOR;
      else
        return ArchipelagoNode<IO>::NONE;
  }
}

///
/// ArchipelagoWorker
///

template <Derived<ArchipelagoIO> IO>
ArchipelagoWorker<IO>::ArchipelagoWorker(node_index_type node_id, ArchipelagoConfig &config, IO &io,
                                         GenomeOperators &go, Dataset d)
    : ArchipelagoNode<IO>(node_id, config, io), go(go), dataset(d) {
  this->populate_relationships();

  assert(config.node_roles[node_id] == node_role::WORKERS);
  assert(parents.size() == 1);
  assert(children.size() == 0);
  assert(neighbors.size() == 0);

  request_msg = unique_ptr<Msg>((Msg *) new RequestMsg());
  // Send an initial work request
  Log::info("Worker parent = %d, worker = %d\n", parents[0], node_id);
  io.send_msg_to(request_msg.get(), parents[0]);
}

template <Derived<ArchipelagoIO> IO>
ArchipelagoWorker<IO>::~ArchipelagoWorker() {}

template <Derived<ArchipelagoIO> IO>
typename ArchipelagoNode<IO>::node_relationship ArchipelagoWorker<IO>::relationship_with(node_index_type other) {
  switch (config.node_roles[other]) {
    case node_role::MANAGERS:
    case node_role::MASTER:
    case node_role::WORKERS:
      return ArchipelagoNode<IO>::INVALID;
    case node_role::ISLANDS:
      assert(config.connections[other][node_id] || config.connections[node_id][other]);
      assert(other != node_id);
      return ArchipelagoNode<IO>::CHILD;
  }
}

template <Derived<ArchipelagoIO> IO>
void ArchipelagoWorker<IO>::process_msg(unique_ptr<Msg> msg, node_index_type src) {
  assert(src == parents[0]);

  Log::info("processing msg\n");

  switch (msg->get_msg_ty()) {
    case Msg::WORK: {
      // Do the work
      WorkMsg *work = (WorkMsg *) msg.get();
      unique_ptr<RNN_Genome> genome = work->get_genome(go);

      if (go.training_parameters.bp_iterations == 0) {
        genome->calculate_fitness(dataset.training_inputs, dataset.training_outputs, dataset.validation_inputs,
                                  dataset.validation_outputs);
      } else {
        genome->backpropagate_stochastic(dataset.training_inputs, dataset.training_outputs, dataset.validation_inputs,
                                         dataset.validation_outputs);
      }

      // Send the result
      Msg *result_msg = (Msg *) new ResultMsg(genome);
      io.send_msg_to(result_msg, src);
      delete result_msg;

      // Request another task
      io.send_msg_to(request_msg.get(), parents[0]);
      break;
    }
    default:
      Log::fatal("ArchipelagoWorker should not recieve any type of Msg other than a WorkMsg, but got %d from %d\n",
                 msg->get_msg_ty(), src);
      exit(1);
  }
}

///
/// ArchipelagoManager
///

template <Derived<ArchipelagoIO> IO>
ArchipelagoManager<IO>::ArchipelagoManager(node_index_type node_id, ArchipelagoConfig &config, IO &io,
                                           uint32_t max_genomes)
    : ArchipelagoNode<IO>(node_id, config, io), max_genomes(max_genomes) {
  this->populate_relationships();
  assert(children.size() != 0);
  assert(parents.size() != 0);
}

template <Derived<ArchipelagoIO> IO>
ArchipelagoManager<IO>::~ArchipelagoManager() {}

template <Derived<ArchipelagoIO> IO>
typename ArchipelagoNode<IO>::node_relationship ArchipelagoManager<IO>::relationship_with(node_index_type other) {
  switch (config.node_roles[other]) {
    case node_role::MANAGERS:
      if (config.connections[node_id][other] ^ config.connections[other][node_id]) {
        if (config.connections[node_id][other])
          return ArchipelagoNode<IO>::CHILD;
        else
          return ArchipelagoNode<IO>::PARENT;
      } else {
        return ArchipelagoNode<IO>::NEIGHBOR;
      }

    case node_role::MASTER:
    case node_role::WORKERS:
      return ArchipelagoNode<IO>::INVALID;

    case node_role::ISLANDS:
      assert(config.connections[other][node_id]);
      return ArchipelagoNode<IO>::CHILD;
  }
}

template <Derived<ArchipelagoIO> IO>
void ArchipelagoManager<IO>::process_msg(unique_ptr<Msg> msg, node_index_type) {
  switch (msg->get_msg_ty()) {
    case Msg::GENOME_SHARE: {
      auto gs_msg = (GenomeShareMsg *) msg.get();
      shared_ptr<RNN_Genome> g = gs_msg->get_genome();

      auto index_iterator = upper_bound(genomes.begin(), genomes.end(), g, sort_genomes_by_fitness());
      uint32_t insert_index = index_iterator - genomes.begin();
      if (insert_index < max_genomes) genomes.emplace(index_iterator, g);

      Log::info("Insert index = %d\n");

      if (!gs_msg->should_propagate()) break;
      if (insert_index == 0) {
        // No need to fallthrough here because share_genome also sends genome evals.
        this->share_genome(genomes[0]);
        break;
      }
    }  /// FALLTHROUGH
    case Msg::EVAL_ACCOUNTING: {
      auto ea_msg = (EvalAccountingMsg *) msg.get();
      this->account_genome_evals(ea_msg->n_evals);
    }
    default:
      Log::fatal("ArchipelagoManager should not recieve message of type %d\n", msg->get_msg_ty());
  }
}

template <Derived<ArchipelagoIO> IO>
ArchipelagoMaster<IO>::ArchipelagoMaster(node_index_type node_id, ArchipelagoConfig &config, IO &io,
                                         string log_file_path, uint32_t max_genomes)
    : ArchipelagoNode<IO>(node_id, config, io),
      log_file_path(log_file_path),
      log_file(make_unique<ofstream>(log_file_path, ios_base::out | ios_base::trunc)),
      genome_evals(0),
      max_genomes(max_genomes) {
  this->populate_relationships();
  assert(children.size() != 0);
  assert(parents.size() == 0);
  assert(neighbors.size() == 0);

  if (!log_file->good()) {
    Log::fatal("Opening log file failed for some reason\n");
    terminate();
  }

  (*log_file) << "Inserted Genomes, Time, Best Val. MAE, Best Val. MSE, Enabled Nodes, Enabled Edges, Enabled Rec. "
                 "Edges, Fitness";

  start = std::chrono::system_clock::now();
}

template <Derived<ArchipelagoIO> IO>
ArchipelagoMaster<IO>::~ArchipelagoMaster() {}

template <Derived<ArchipelagoIO> IO>
typename ArchipelagoNode<IO>::node_relationship ArchipelagoMaster<IO>::relationship_with(node_index_type other) {
  switch (config.node_roles[other]) {
    case node_role::MANAGERS:
      if (config.connections[node_id][other] ^ config.connections[other][node_id]) {
        if (config.connections[node_id][other])
          return ArchipelagoNode<IO>::PARENT;
        else
          return ArchipelagoNode<IO>::INVALID;
      } else {
        return ArchipelagoNode<IO>::INVALID;
      }

    case node_role::MASTER:
    case node_role::WORKERS:
      return ArchipelagoNode<IO>::INVALID;

    case node_role::ISLANDS:
      assert(config.connections[other][node_id]);
      return ArchipelagoNode<IO>::PARENT;
  }
}

template <Derived<ArchipelagoIO> IO>
void ArchipelagoMaster<IO>::process_msg(unique_ptr<Msg> msg, node_index_type) {
  switch (msg->get_msg_ty()) {
    case Msg::GENOME_SHARE: {
      Log::info("Master got genome share\n");
      auto gs_msg = (GenomeShareMsg *) msg.get();
      unique_ptr<RNN_Genome> g = gs_msg->get_genome();
      if (!best_genome || g->get_fitness() < best_genome->get_fitness()) best_genome = move(g);
    }  /// FALLTHROUGH
    case Msg::EVAL_ACCOUNTING: {
      auto ea_msg = (EvalAccountingMsg *) msg.get();
      genome_evals += ea_msg->n_evals;
      update_log();
      break;
    }
    default:
      Log::fatal("ArchipelagoManager should not recieve message of type %d\n", msg->get_msg_ty());
  }

  Log::info("Genome evals: %d, max: %d\n", genome_evals, max_genomes);
  if (genome_evals > max_genomes) terminate();

  //   // If for some weird reason, terminates were already sent.
  //   if (genome_evals >= max_genomes && this->terminates_sent == 0) {
  //   }
}

template <Derived<ArchipelagoIO> IO>
void ArchipelagoMaster<IO>::terminate() {
  auto tmsg = make_unique<TerminateMsg>();
  io.send_msg_all((Msg *) tmsg.get(), children);
}

template <Derived<ArchipelagoIO> IO>
void ArchipelagoMaster<IO>::update_log() {
  Log::info("logging %d\n", best_genome == nullptr);
  if (!log_file->good()) {
    Log::warning("Something is wrong with the log file. Attempting to re-create the log file.\n");
    log_file = make_unique<ofstream>(log_file_path.c_str(), ios_base::out | ios_base::app);
    if (!log_file->good()) {
      Log::fatal("Failed to re-create the log file\n");
      terminate();
    }
  }

  std::chrono::time_point<std::chrono::system_clock> currentClock = std::chrono::system_clock::now();
  long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(currentClock - start).count();

  const RNN_Genome *best = best_genome.get();
  if (best == nullptr) return;
  (*log_file) << genome_evals << ", " << milliseconds << ", " << best->get_best_validation_mae() << ", "
              << best->get_best_validation_mse() << ", " << best->get_enabled_node_count() << ","
              << best->get_enabled_edge_count() << "," << best->get_enabled_recurrent_edge_count() << std::endl;
}

#endif
