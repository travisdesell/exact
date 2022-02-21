#include "msg.hxx"

Msg::Msg() {}

// unique_ptr<RNN_Genome> Msg::get_genome(Msg *work, GenomeOperators &operators) {
//     RNN_Genome *result = work->get_genome(operators);
//     if (result)
//         operators.finalize_genome(result);
//     return result;
// }

unique_ptr<Msg> Msg::read_from_array(const char *array, int32_t length) {
    struct membuf : std::streambuf {
        membuf(const char *begin, const char *end) { this->setg((char *)begin, (char *)begin, (char *)end); }
    };

    membuf mb(array, array + length);
    istream is(&mb);
    return read_from_stream(is);
}

unique_ptr<Msg> Msg::read_from_stream(istream &bin_istream) {
    int msg_ty = bin_istream.peek();

    Msg *m = NULL;
    switch (msg_ty) {
        case msg_ty::WORK:
            m = (Msg *)new WorkMsg(bin_istream);
            break;
        case msg_ty::TERMINATE:
            m = (Msg *)new TerminateMsg(bin_istream);
            break;
        case msg_ty::GENOME_SHARE:
            m = (Msg *)new GenomeShareMsg(bin_istream);
            break;
        case msg_ty::RESULT:
            m = (Msg *)new ResultMsg(bin_istream);
            break;
        case msg_ty::REQUEST:
            m = (Msg *)new RequestMsg(bin_istream);
            break;
        case msg_ty::EVAL_ACCOUNTING:
            m = (Msg *)new EvalAccountingMsg(bin_istream);
            break;
        default:
            Log::fatal("encountered unrecognized msg_ty %d\n", msg_ty);
            exit(1);
    }

    // to satisfy a warning
    return unique_ptr<Msg>(m);
}

WorkMsg::WorkMsg(shared_ptr<const RNN_Genome> g, uint32_t n_mutations)
    : args(mu_args{.g = move(g), .n_mutations = n_mutations}), work_type(mutation), is_shared(true) {}
WorkMsg::WorkMsg(vector<shared_ptr<const RNN_Genome>> &parents)
    : work_type(crossover), args(co_args{.parents = move(parents)}), is_shared(true) {}
WorkMsg::WorkMsg(shared_ptr<const RNN_Genome> g)
    : work_type(mutation), args(mu_args{.g = move(g), .n_mutations = 0}), is_shared(true) {}
WorkMsg::WorkMsg(istream &bin_istream) : is_shared(false) {
    int class_id = bin_istream.get();
    if (class_id != get_msg_ty()) {
        Log::fatal("Read wrong message type for WorkMsg\n");
        exit(1);
    }

    bin_istream.read((char *)&group_id, sizeof(uint32_t));
    bin_istream.read((char *)&genome_number, sizeof(int32_t));

    work_type = bin_istream.get() ? mutation : crossover;

    switch (work_type) {
        case crossover: {
            uint32_t n;
            bin_istream.read((char *)&n, sizeof(uint32_t));

            vector<unique_ptr<RNN_Genome>> parents(n);
            for (uint32_t i = 0; i < n; i++) parents.push_back(make_unique<RNN_Genome>(bin_istream));
            args = co_args{.parents = move(parents)};

            break;
        }
        case mutation: {
            uint32_t n_mutations;

            bin_istream.read((char *)&n_mutations, sizeof(uint32_t));
            unique_ptr<RNN_Genome> genome = make_unique<RNN_Genome>(bin_istream);

            args = mu_args{.g = move(genome), .n_mutations = n_mutations};

            break;
        }
    }
}

void WorkMsg::write_to_stream(ostream &bin_ostream) {
    bin_ostream.put(get_msg_ty());

    bin_ostream.write((char *)&group_id, sizeof(int32_t));
    bin_ostream.write((char *)&genome_number, sizeof(int32_t));

    bin_ostream.put(work_type);
    switch (work_type) {
        case crossover: {
            auto &parents = get<shared>(get<crossover>(args).parents);
            uint32_t n = parents.size();

            bin_ostream.write((char *)&n, sizeof(uint32_t));

            for (uint32_t i = 0; i < n; i++) parents[i]->write_to_stream(bin_ostream);

            break;
        }
        case mutation: {
            mu_args &margs = get<mutation>(args);
            bin_ostream.write((char *)&margs.n_mutations, sizeof(uint32_t));
            get<shared>(margs.g).get()->write_to_stream(bin_ostream);

            break;
        }
    }
}

int32_t WorkMsg::get_msg_ty() const { return Msg::WORK; }

void WorkMsg::set_group_id(int32_t gid) { this->group_id = gid; }
void WorkMsg::set_genome_number(int32_t gn) { this->genome_number = gn; }

thread_local vector<const RNN_Genome *> _work_msg_get_genome_genomes(32);
unique_ptr<RNN_Genome> WorkMsg::get_genome(GenomeOperators &operators) {
    
    RNN_Genome* g = nullptr;
    // Generates new genome and deletes parent genome(s)
    if (work_type == crossover) {
        auto &cargs = get<crossover>(args);
        _work_msg_get_genome_genomes.clear();
        if (cargs.parents.index() == shared) {
            auto &parents = get<shared>(cargs.parents);
            for (int i = 0; i < parents.size(); i++) _work_msg_get_genome_genomes.push_back(parents[i].get());
        } else {
            auto &parents = get<unique>(cargs.parents);
            for (int i = 0; i < parents.size(); i++) _work_msg_get_genome_genomes.push_back(parents[i].get());
        }
        g = operators.ncrossover(_work_msg_get_genome_genomes);
    } else if (work_type == mutation) {
        auto &margs = get<mutation>(args);
        RNN_Genome* parent =
            is_shared ? get<shared>(margs.g)->copy() : get<unique>(margs.g).release();
        operators.mutate(parent, margs.n_mutations);
        g = parent;
    } else {
        // Unreachable
        exit(1);
    }

    unique_ptr<RNN_Genome> gu(g);
    gu->set_generation_id(genome_number);
    gu->set_group_id(group_id);
    return gu;
}

TerminateMsg::TerminateMsg() {}
TerminateMsg::TerminateMsg(istream &bin_istream) {
    int msg_ty = bin_istream.get();

    if (msg_ty != get_msg_ty()) {
        Log::fatal("Read wrong message type for TerminateMsg\n");
        exit(1);
    }
}

void TerminateMsg::write_to_stream(ostream &bin_ostream) { bin_ostream.put(get_msg_ty()); }

int32_t TerminateMsg::Msg::get_msg_ty() const { return Msg::TERMINATE; }

EvalAccountingMsg::EvalAccountingMsg(uint32_t n) : n_evals(n) {}
EvalAccountingMsg::EvalAccountingMsg(istream &bin_istream) {
    int msg_ty = bin_istream.get();

    if (msg_ty != get_msg_ty()) {
        Log::fatal("Read wrong message type for EvalAccountingMsg\n");
        exit(1);
    }

    bin_istream.read((char *)&n_evals, sizeof(uint32_t));
}

void EvalAccountingMsg::write_to_stream(ostream &bin_ostream) {
    bin_ostream.put(get_msg_ty());
    bin_ostream.write((char *)&n_evals, sizeof(uint32_t));
}

int32_t EvalAccountingMsg::get_msg_ty() const { return Msg::EVAL_ACCOUNTING; }

GenomeShareMsg::GenomeShareMsg(RNN_Genome *g, bool propagate, uint32_t n_evals)
    : EvalAccountingMsg(n_evals), genome(g), propagate(propagate) {}
GenomeShareMsg::GenomeShareMsg(istream &bin_istream) : EvalAccountingMsg(0) {
    int msg_ty = bin_istream.get();

    if (msg_ty != get_msg_ty()) {
        Log::fatal("Read wrong message type for GenomeShareMsg\n");
        exit(1);
    }

    bin_istream.read((char *)&propagate, sizeof(bool));
    bin_istream.read((char *)&n_evals, sizeof(uint32_t));
    genome = make_shared<RNN_Genome>(bin_istream);
}

void GenomeShareMsg::write_to_stream(ostream &bin_ostream) {
    bin_ostream.put(get_msg_ty());
    bin_ostream.write((char *)&propagate, sizeof(bool));
    bin_ostream.write((char *)&n_evals, sizeof(uint32_t));
    genome->write_to_stream(bin_ostream);
}

int32_t GenomeShareMsg::get_msg_ty() const { return Msg::GENOME_SHARE; }

ResultMsg::ResultMsg(unique_ptr<RNN_Genome> &g) : genome(move(g)) {}
ResultMsg::ResultMsg(RNN_Genome *g) : genome(g) {}
ResultMsg::ResultMsg(istream &bin_istream) {
    int msg_ty = bin_istream.get();

    if (msg_ty != get_msg_ty()) {
        Log::fatal("Read wrong message type for ResultMsg\n");
        exit(1);
    }

    genome = make_unique<RNN_Genome>(bin_istream);
}

void ResultMsg::write_to_stream(ostream &bin_ostream) {
    bin_ostream.put(get_msg_ty());
    genome->write_to_stream(bin_ostream);
}

int32_t ResultMsg::get_msg_ty() const { return Msg::RESULT; }

unique_ptr<RNN_Genome> ResultMsg::get_genome() { return move(genome); }

RequestMsg::RequestMsg() {}
RequestMsg::RequestMsg(istream &bin_istream) {
    int msg_ty = bin_istream.get();

    if (msg_ty != get_msg_ty()) {
        Log::fatal("Read wrong message type for RequestMsg\n");
        exit(1);
    }
}

void RequestMsg::write_to_stream(ostream &bin_ostream) { bin_ostream.put(get_msg_ty()); }

int32_t RequestMsg::get_msg_ty() const { return Msg::REQUEST; }

int32_t TerminateMsg::get_msg_ty() const { return Msg::TERMINATE; }
