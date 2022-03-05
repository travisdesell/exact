#include <deque>
using std::deque;

#include <memory>
using std::shared_ptr;
using std::unique_ptr;
using std::make_unique;

#include <utility>
using std::move;

#include <random>
using std::mt19937_64;
using std::random_device;

#include <vector>
using std::vector;

#include "common/archipelago_config.hxx"
#include "genome_operators.hxx"
#include "island.hxx"
#include "island_speciation_strategy.hxx"
#include "msg.hxx"
#include "rnn_genome.hxx"

struct ArchipelagoIO {
  virtual ~ArchipelagoIO();
  virtual void send_msg_to(Msg *, node_index_type) = 0;
  virtual void send_msg_all(Msg *, node_index_type) = 0;
  virtual pair<unique_ptr<Msg>, node_index_type> recieve_msg() = 0;
};

template <class Child, class Parent>
concept Derived = std::is_base_of<Parent, Child>::value;

template <Derived<ArchipelagoIO> IO>
class ArchipelagoNode {
 protected:
  const node_index_type node_id;
  ArchipelagoConfig &config;
  IO &io;

  mt19937_64 generator;
  const node_role role;
  vector<node_index_type> parents, children, neighbors;
  int terminate_count = 0;
  int terminates_sent = 0;
  // The number of genomes that have been processed since the last EvalAccountingMsg or GenomeShareMsg was sent.
  int unrecorded_genome_count = 0;
  int unrecorded_genome_threshold = 10;

 public:
  ArchipelagoNode(node_index_type node_id, ArchipelagoConfig &config, IO &io);
  virtual ~ArchipelagoNode();

  enum node_relationship { PARENT, NEIGHBOR, CHILD, INVALID, NONE };

  // Relationship this node has to another node, based on their
  virtual node_relationship relationship_with(node_index_type) {
    Log::info("It shouldnt be possible to call this function, but you managed to do so! Well done\n");
    exit(1);
  }

  void run() {
    // Keep running if all parents havent sent a terminate,
    // or if there are no parents (this is a master node) and we havent terminated the children yet.
    while (terminate_count < (int) parents.size() || (parents.size() == 0 && terminates_sent < children.size())) {
      auto [msg, src] = io.recieve_msg();

      if (TerminateMsg *tmsg = dynamic_cast<TerminateMsg *>(msg.get()); tmsg != nullptr) {
        terminate_count += 1;
        if (terminate_count == parents.size()) {
          terminates_sent += children.size();
          io.send_msg_all((Msg *) tmsg, children);
        }
      } else {
        process_msg(move(msg), src);
      }
    }
  }

  void account_genome_evals(int n=0) {
    unrecorded_genome_count += n;
    if (unrecorded_genome_count >= unrecorded_genome_threshold) {
      send_eval_accounting();
    }
  }

  void share_genome(shared_ptr<const RNN_Genome> shared) {
    unique_ptr<GenomeShareMsg> ea_msg = make_unique<GenomeShareMsg>(move(shared), false, unrecorded_genome_count);
    unrecorded_genome_count = 0;

    int propagate_index = uniform_int_distribution<int>(0, parents.size() - 1)(generator);
    for (int i = 0; i < (int) parents.size(); i++) {
      ea_msg->set_propagate(i == propagate_index);
      io.send_msg_to((Msg *) ea_msg.get(), parents[i]);
    }

    ea_msg->set_propagate(false);
    io.send_message_to((Msg *) ea_msg.get(), neighbors);
  }

  void send_eval_accounting() {
    int index = uniform_int_distribution<int>(0, parents.size() - 1)(generator);
    unique_ptr<EvalAccountingMsg> ea_msg = make_unique<EvalAccountingMsg>(unrecorded_genome_count);
    io.send_msg_to((Msg *) ea_msg.get(), parents[index]);
    unrecorded_genome_count = 0;
  }

  virtual void process_msg(unique_ptr<Msg> msg, node_index_type src) = 0;
};

template <Derived<ArchipelagoIO> IO>
class ArchipelagoWorker : public ArchipelagoNode<IO> {
  using ArchipelagoNode<IO>::io;
  using ArchipelagoNode<IO>::config;
  using ArchipelagoNode<IO>::parents;
  using ArchipelagoNode<IO>::children;
  using ArchipelagoNode<IO>::node_id;
  using ArchipelagoNode<IO>::neighbors;
  using ArchipelagoNode<IO>::unrecorded_genome_count;

  GenomeOperators &go;

 public:
  ArchipelagoWorker(node_index_type node_id, ArchipelagoConfig &config, IO &io, GenomeOperators &go);

  virtual typename ArchipelagoNode<IO>::node_relationship relationship_with(node_index_type other);
  virtual void process_msg(unique_ptr<Msg> msg, node_index_type src);
};

template <Derived<ArchipelagoIO> IO>
class ArchipelagoIslandCluster : public ArchipelagoNode<IO>, public IslandSpeciationStrategy {
  using ArchipelagoNode<IO>::io;
  using ArchipelagoNode<IO>::config;
  using ArchipelagoNode<IO>::parents;
  using ArchipelagoNode<IO>::children;
  using ArchipelagoNode<IO>::node_id;
  using ArchipelagoNode<IO>::neighbors;
  using ArchipelagoNode<IO>::unrecorded_genome_count;

  // Crossover using genomes recieved from other nodes.
  static constexpr inline double foreign_crossover_p = 0.25;

  // Crossover using genomes in the local cluster.
  static constexpr inline double local_crossover_p = 0.75;

  // Hope many results to record before sending an accounting message.
  static constexpr inline int UNRECORDED_GENOME_THRESHOLD = 10;

  uint32_t max_foreign_genomes;

  deque<shared_ptr<const RNN_Genome>> foreign_genomes;

  pair<uint32_t, uint32_t> n_parents_foreign_range;
  uint32_t get_random_n_parents_foreign();
  void get_n_random_foreign_genomes(uint32_t n, vector<shared_ptr<const RNN_Genome>> &genomes);

 protected:
  virtual unique_ptr<WorkMsg> generate_work_for_filled_island(Island &island);
  virtual typename ArchipelagoNode<IO>::node_relationship relationship_with(node_index_type other);
  
  virtual void process_msg(unique_ptr<Msg> msg, node_index_type src);
  void process_request(RequestMsg *, node_index_type src);
  void process_result(ResultMsg *, node_index_type src);
  void process_genome_share(GenomeShareMsg *, node_index_type src);

 public:
  ArchipelagoIslandCluster(node_index_type node_id, ArchipelagoConfig &config, IO &io, uint32_t number_of_islands,
                           uint32_t max_island_size, shared_ptr<const RNN_Genome> seed_genome,
                           pair<uint32_t, uint32_t> n_parents_foreign_range, GenomeOperators &go);
  virtual ~ArchipelagoIslandCluster();

};

template <Derived<ArchipelagoIO> IO>
class ArchipelagoManager : public ArchipelagoNode<IO> {
  using ArchipelagoNode<IO>::io;
  using ArchipelagoNode<IO>::config;
  using ArchipelagoNode<IO>::parents;
  using ArchipelagoNode<IO>::children;
  using ArchipelagoNode<IO>::node_id;
  using ArchipelagoNode<IO>::neighbors;
  using ArchipelagoNode<IO>::unrecorded_genome_count;

  uint32_t max_genomes;
  vector<shared_ptr<const RNN_Genome>> genomes;

 protected:
  virtual typename ArchipelagoNode<IO>::node_relationship relationship_with(node_index_type other);
  virtual void process_msg(unique_ptr<Msg> msg, node_index_type src);

 public:
  ArchipelagoManager(node_index_type, ArchipelagoConfig &, IO &, uint32_t);

};

template <Derived<ArchipelagoIO> IO>
class ArchipelagoMaster : public ArchipelagoNode<IO> {
  using ArchipelagoNode<IO>::io;
  using ArchipelagoNode<IO>::config;
  using ArchipelagoNode<IO>::parents;
  using ArchipelagoNode<IO>::children;
  using ArchipelagoNode<IO>::node_id;
  using ArchipelagoNode<IO>::neighbors;
  using ArchipelagoNode<IO>::unrecorded_genome_count;

  uint32_t max_genomes;
  uint32_t genome_evals;
  shared_ptr<const RNN_Genome> best_genome;
  string log_file_path;
  unique_ptr<ofstream> log_file;
  std::chrono::time_point<std::chrono::system_clock> start;

  void terminate();
  void update_log();

 protected:
  virtual typename ArchipelagoNode<IO>::node_relationship relationship_with(node_index_type other);
  virtual void process_msg(unique_ptr<Msg> msg, node_index_type src);

 public:
  ArchipelagoMaster(node_index_type, ArchipelagoConfig &, IO &, string log_file_location, uint32_t max_genomes);
};
