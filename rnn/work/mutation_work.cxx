MutationWork::MutationWork(RNN_Genome *_genome, int32_t _n_mutations, int32_t _generation_id, int32_t _group_id) 
    : Work(_generation_id, _group_id), n_mutations(_n_mutations), genome(_genome) { }

MutationWork::MutationWork(istream &bin_istream) {
    int class_id = bin_istream.get();

    if (class_id != MutationWork::class_id) {
        Log::fatal("Read the wrong class_id for MutationWork");
        exit(0);
    }

    bin_istream.read((char *) &n_mutations, sizeof(int32_t));

    bin_istream.read((char *) &generation_id, sizeof(int32_t));
    bin_istream.read((char *) &group_id, sizeof(int32_t));

    genome = new RNN_Genome(bin_istream);
}

MutationWork::~MutationWork() {
    delete genome;
}

void MutationWork::write_to_stream(ostream &bin_ostream) {
    bin_ostream.put(MutationWork::class_id);

    bin_ostream.write((char *) &n_mutations, sizeof(int32_t));
    
    bin_ostream.write((char *) &generation_id, sizeof(int32_t));
    bin_ostream.write((char *) &group_id, sizeof(int32_t));
    
    genome->write_to_stream(bin_ostream);
}

RNN_Genome *MutationWork::get_genome(GenomeOperators &operators) {
    RNN_Genome *clone = NULL;   
    
    while (clone == NULL) {
        clone = genome->copy();

        if (n_mutations) 
            operators.mutate(clone, n_mutations);

        if (clone->outputs_unreachable()) {
            delete clone;
            clone = NULL;
        }
    }
    
    clone->set_group_id(group_id);
    clone->set_generation_id(generation_id);

    return clone;
}

int32_t MutationWork::get_class_id() { return MutationWork::class_id; }
