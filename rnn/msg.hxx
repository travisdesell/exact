#ifndef MSG_HXX
#define MSG_HXX

#include <memory>
#include <variant>

#include "genome_operators.hxx"
#include "rnn_genome.hxx"

class Msg {
   public:
    enum msg_ty : uint8_t {
        WORK,
        TERMINATE,
        GENOME_SHARE,
        GENOME_RESULT,
        GENOME_REQUEST,
        EVAL_ACCOUNTING,
    };

    static Msg* read_from_stream(istream &bin_istream);
    static Msg* read_from_array(const char *array, int32_t length);

    Msg();
    Msg(istream &bin_istream);
    virtual ~Msg() = 0;

    virtual void write_to_stream(ostream &bin_ostream) = 0;
    virtual int32_t get_msg_ty() const = 0;
};

class WorkMsg : public Msg {
   private:
    struct mu_args {
        shared_ptr<const RNN_Genome> g;
        uint32_t n_mutations;
    };
    struct co_args {
        vector<shared_ptr<const RNN_Genome>> parents;
    };

    std::variant<co_args, mu_args> args;
    int32_t group_id;
    int32_t genome_number;

   public:
    enum : uint8_t { crossover = 0, mutation = 1 } work_type;
   
    WorkMsg(RNN_Genome *g, uint32_t n_mutations); // Mutation constructor
    WorkMsg(vector<shared_ptr<const RNN_Genome>> parents); // Crossover constructor
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
   private:
    uint32_t n_evals;

   public:
    EvalAccountingMsg();
    EvalAccountingMsg(istream &bin_istream);

    virtual void write_to_stream(ostream &bin_ostream);
    virtual int32_t get_msg_ty() const;
};

class GenomeShareMsg : public EvalAccountingMsg {
   private:
    shared_ptr<const RNN_Genome> genome;
    bool propagate;

   public:
    GenomeShareMsg();
    GenomeShareMsg(istream &bin_istream);

    virtual void write_to_stream(ostream &bin_ostream);
    virtual int32_t get_msg_ty() const;
};

class GenomeResultMsg : public Msg {
   private:
    unique_ptr<RNN_Genome> genome;

   public:
    GenomeResultMsg();
    GenomeResultMsg(istream &bin_istream);

    virtual void write_to_stream(ostream &bin_ostream);
    virtual int32_t get_msg_ty() const;
};

class GenomeRequestMsg : public Msg {
   public:
    GenomeRequestMsg();
    GenomeRequestMsg(istream &bin_istream);

    virtual void write_to_stream(ostream &bin_ostream);
    virtual int32_t get_msg_ty() const;
};

#endif
