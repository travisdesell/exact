#ifndef MSG_HXX
#define MSG_HXX

#include <memory>
#include <utility>
#include <variant>

#include "genome_operators.hxx"
#include "rnn_genome.hxx"

class Msg {
 public:
  enum msg_ty : uint8_t {
    // Unit of work to be sent to a worker
    WORK = 1,
    // Result after a genome has been trained
    RESULT = 2,
    // Worker sends this to request a genome
    REQUEST = 3,
    // Used to properly initialize the innovation number counts, and can be used for
    // other things in the future.
    MPI_INIT = 4,
    // Sent after the proper # of genomes have been generated
    TERMINATE = 5,
    // Sharing a genome to other regions
    GENOME_SHARE = 6,
    // Used to send information about the number of genomes generated to the
    // master.
    // Should be propagated upwards to the master.
    EVAL_ACCOUNTING = 7,
  };

  static unique_ptr<Msg> read_from_stream(istream &bin_istream);
  static unique_ptr<Msg> read_from_array(const char *array, int32_t length);

  Msg();
  Msg(istream &bin_istream);
  virtual ~Msg() {}

  virtual void write_to_stream(ostream &bin_ostream) = 0;
  virtual int32_t get_msg_ty() const = 0;
};

class WorkMsg : public Msg {
 private:
  enum { unique = 0, shared = 1 };

  struct mu_args {
    typedef unique_ptr<RNN_Genome> unique;
    typedef shared_ptr<const RNN_Genome> shared;
    // Using a union to avoid unecessary destructor calls on gs, since
    // shared_ptr operations are relatively expensive.
    std::variant<unique, shared> g;
    uint32_t n_mutations;
  };
  struct co_args {
    typedef vector<unique_ptr<RNN_Genome>> unique;
    typedef vector<shared_ptr<const RNN_Genome>> shared;
    std::variant<unique, shared> parents;
  };

  std::variant<co_args, mu_args> args;
  int32_t group_id = -1;
  int32_t genome_number = -1;
  bool is_shared;

 public:
  enum : uint8_t { crossover = 0, mutation = 1 } work_type;

  WorkMsg(shared_ptr<const RNN_Genome> g,
          uint32_t n_mutations);                           // Mutation constructor
  WorkMsg(vector<shared_ptr<const RNN_Genome>> &parents);  // Crossover constructor
  WorkMsg(shared_ptr<const RNN_Genome> g);                 // Train constructor; represented as mu_args w/ 0 mutations.
  WorkMsg(istream &bin_istream);
  virtual ~WorkMsg() = default;

  virtual void write_to_stream(ostream &bin_ostream);
  virtual int32_t get_msg_ty() const;

  void set_group_id(int32_t gid);
  void set_genome_number(int32_t gn);
  unique_ptr<RNN_Genome> get_genome(GenomeOperators &operators);
};

class TerminateMsg : public Msg {
 public:
  TerminateMsg();
  TerminateMsg(istream &bin_istream);

  virtual void write_to_stream(ostream &bin_ostream);
  virtual int32_t get_msg_ty() const;
};

class EvalAccountingMsg : public Msg {
 protected:
  uint32_t n_evals;

 public:
  EvalAccountingMsg(uint32_t n_evals);
  EvalAccountingMsg(istream &bin_istream);

  virtual void write_to_stream(ostream &bin_ostream);
  virtual int32_t get_msg_ty() const;
};

class GenomeShareMsg : public EvalAccountingMsg {
 private:
  shared_ptr<const RNN_Genome> genome;
  bool propagate;

 public:
  GenomeShareMsg(RNN_Genome *g, bool propagate, uint32_t n_evals = 0);
  GenomeShareMsg(istream &bin_istream);

  virtual void write_to_stream(ostream &bin_ostream);
  virtual int32_t get_msg_ty() const;
};

class ResultMsg : public Msg {
 private:
  unique_ptr<RNN_Genome> genome;

 public:
  ResultMsg(unique_ptr<RNN_Genome> &g);
  ResultMsg(RNN_Genome *g);
  ResultMsg(istream &bin_istream);

  virtual void write_to_stream(ostream &bin_ostream);
  virtual int32_t get_msg_ty() const;

  unique_ptr<RNN_Genome> get_genome();
};

class RequestMsg : public Msg {
 public:
  RequestMsg();
  RequestMsg(istream &bin_istream);

  virtual void write_to_stream(ostream &bin_ostream);
  virtual int32_t get_msg_ty() const;
};

class MPIInitMsg : public Msg {
 private:
  node_inon nin;
  edge_inon ein;

 public:
  MPIInitMsg(node_inon _nin, edge_inon _ein);
  MPIInitMsg(istream &bin_istream);

  virtual void write_to_stream(ostream &bin_ostream);
  virtual int32_t get_msg_ty() const;
  void run(size_t n_workers, size_t worker_id);
};

#endif
