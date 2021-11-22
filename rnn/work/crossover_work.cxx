CrossoverWork::CrossoverWork(vector<RNN_Genome *> &parents, int32_t _generation_id, int32_t _group_id)
    : Work(_generation_id, _group_id), parents(move(parents)) { }

CrossoverWork::CrossoverWork(istream &bin_istream) {
    int class_id = bin_istream.get();

    if (class_id != CrossoverWork::class_id) {
        Log::fatal("Read the wrong class_id for CrossoverWork");
        exit(0);
    }

    bin_istream.read((char *) &generation_id, sizeof(int32_t));
    bin_istream.read((char *) &group_id, sizeof(int32_t));
    
    int32_t n = -1;
    bin_istream.read((char *) &n, sizeof(int32_t));

    vector<RNN_Genome *> parents(n);
    for (int i = 0; i < n; i++) {
        RNN_Genome *p = new RNN_Genome(bin_istream);
        parents.push_back(p);
    }
}

CrossoverWork::~CrossoverWork() {
    for (int i = 0; i < parents.size(); i++)
        delete parents[i];
}

void CrossoverWork::write_to_stream(ostream &bin_ostream) {
    bin_ostream.put(CrossoverWork::class_id);

    bin_ostream.write((char *) &generation_id, sizeof(int32_t));
    bin_ostream.write((char *) &group_id, sizeof(int32_t));

    int32_t n = (int32_t) parents.size();
    bin_ostream.write((char *) &n, sizeof(int32_t));

    for (int i = 0; i < n; i++)
        parents[i]->write_to_stream(bin_ostream);
}

RNN_Genome *CrossoverWork::get_genome(GenomeOperators &operators) {
    RNN_Genome *genome = operators.ncrossover(parents);
    genome->set_generation_id(generation_id);
    genome->set_group_id(group_id);
    return genome;
}

int32_t CrossoverWork::get_class_id() { return CrossoverWork::class_id; }
