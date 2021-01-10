MutationWork::MutationWork(RNN_Genome *_genome, int32_t _n_mutations) 
    : genome(_genome), n_mutations(_n_mutations) { }

MutationWork::MutationWork(istream &bin_istream) {
    int class_id = bin_istream.get();
    if (class_id != MutationWork::class_id) {
        Log::fatal("Read the wrong class_id for MutationWork");
        exit(0);
    }

    bin_istream.read((char *) &n_mutations, sizeof(int32_t));
    genome = new RNN_Genome(bin_istream);
}

MutationWork::~MutationWork() {
    delete genome;
}

void MutationWork::write_to_stream(ostream &bin_ostream) {
    bin_ostream.put(MutationWork::class_id);
    bin_ostream.write((char *) &n_mutations);

    genome->write_to_stream(bin_ostream);
}

RNN_Genome *MutationWork::get_genome(GenomeOperators &operators) {
    return operators.mutate(this->genome, n_mutations);
}

int32_t MutationWork::get_class_id() { return MutationWork::class_id; }


