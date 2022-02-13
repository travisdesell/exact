#include "msg.hxx"

Msg::Msg() {}

// unique_ptr<RNN_Genome> Msg::get_genome(Msg *work, GenomeOperators &operators) {
//     RNN_Genome *result = work->get_genome(operators);
//     if (result)
//         operators.finalize_genome(result);
//     return result;
// }

Msg *Msg::read_from_array(const char *array, int32_t length) {
    struct membuf : std::streambuf {
        membuf(const char *begin, const char *end) { this->setg((char *)begin, (char *)begin, (char *)end); }
    };

    membuf mb(array, array + length);
    istream is(&mb);
    return read_from_stream(is);
}

Msg *Msg::read_from_stream(istream &bin_istream) {
    int msg_ty = bin_istream.peek();

    switch (msg_ty) {
        case msg_ty::WORK:
            return (Msg *)new WorkMsg(bin_istream);
            break;
        case msg_ty::TERMINATE:
            return (Msg *)new TerminateMsg(bin_istream);
            break;
        case msg_ty::GENOME_SHARE:
            return (Msg *)new GenomeShareMsg(bin_istream);
            break;
        case msg_ty::GENOME_RESULT:
            return (Msg *)new GenomeResultMsg(bin_istream);
            break;
        case msg_ty::GENOME_REQUEST:
            return (Msg *)new GenomeRequestMsg(bin_istream);
            break;
        case msg_ty::EVAL_ACCOUNTING:
            return (Msg *)new EvalAccountingMsg(bin_istream);
            break;
        default:
            break;
    }

    Log::fatal("encountered unrecognized msg_ty %d\n", msg_ty);
    exit(1);

    // to satisfy a warning
    return NULL;
}

WorkMsg::WorkMsg(shared_ptr<const RNN_Genome> g, uint32_t n_mutations)
    : args(mu_args{.g = g, .n_mutations = n_mutations}), work_type(mutation) {}
WorkMsg::WorkMsg(vector<shared_ptr<const RNN_Genome>> parents) : work_type(crossover), args(co_args{.parents = parents}) {}
WorkMsg::WorkMsg(istream &bin_istream) {
    int class_id = bin_istream.get();
    if (class_id != get_msg_ty()) {
        Log::fatal("Read wrong message type for WorkMsg\n");
        exit(1);
    }

    bin_istream.read((char *) &group_id, sizeof(uint32_t));
    bin_istream.read((char *) &genome_number, sizeof(int32_t));

    work_type = bin_istream.get() ? mutation : crossover;

    if (work_type == crossover) {
        uint32_t n;
        bin_istream.read((char *)&n, sizeof(uint32_t));

        vector<shared_ptr<const RNN_Genome>> parents(n);
        for (uint32_t i = 0; i < n; i++) parents.push_back(make_unique<const RNN_Genome>(bin_istream));
        
        args = co_args{.parents = move(parents)};
    } else {
        uint32_t n_mutations;

        bin_istream.read((char *)&n_mutations, sizeof(uint32_t));
        unique_ptr<const RNN_Genome> genome = make_unique<const RNN_Genome>(bin_istream);

        args = mu_args{.g = genome, .n_mutations = n_mutations};
    }
}

void WorkMsg::write_to_stream(ostream &bin_ostream) {
    bin_ostream.put(get_msg_ty());

    bin_ostream.write((char *) &group_id, sizeof(int32_t));
    bin_ostream.write((char *) &genome_number, sizeof(int32_t));

    bin_ostream.put(work_type);

    if (work_type == crossover) {
        auto &parents = get<crossover>(args).parents;
        uint32_t n = parents.size();

        bin_ostream.write((char *)&n, sizeof(uint32_t));

        for (uint32_t i = 0; i < n; i++) parents[i]->write_to_stream(bin_ostream);
    } else {
        mu_args &margs = get<mutation>(args);
        bin_ostream.write((char *)&margs.n_mutations, sizeof(uint32_t));
        margs.g->write_to_stream(bin_ostream);
    }
}

int32_t WorkMsg::get_msg_ty() const {
    return Msg::WORK;
}

void WorkMsg::set_group_id(int32_t gid) { this->group_id = gid; }
void WorkMsg::set_genome_number(int32_t gn) { this->genome_number = gn; }

unique_ptr<RNN_Genome> WorkMsg::get_genome(GenomeOperators &operators) {
    hild;
    // Generates new genome and deletes parent genome(s)
    if (work_type == crossover) {
        co_args &cargs = get<crossover>(args);
        unique_ptr<RNN_Genome> c = operators.ncrossover(cargs.parents);
        return c;
    } else {
        mu_args &margs = get<mutation>(args);
        unique_ptr<RNN_Genome> c = operators.mutate(margs.g, margs.n_mutations);
        return c;
    }
}

TerminateMsg::TerminateMsg() {}
TerminateMsg::TerminateMsg(istream &bin_istream) {
    int msg_ty = bin_istream.get();

    if (msg_ty != get_msg_ty()) {
        Log::fatal("Read wrong message type for TerminateMsg\n");
        exit(1);
    }
}

void TerminateMsg::write_to_stream(ostream &bin_ostream) {
    bin_ostream.put(get_msg_ty());
}

int32_t TerminateMsg::get_msg_ty() const {
    return Msg::TERMINATE;
}
