CrossoverWork::CrossoverWork(RNN_Genome *_more_fit, RNN_Genome *_less_fit, int32_t _generation_id, int32_t _group_id)
    : Work(_generation_id, _group_id), more_fit(_more_fit), less_fit(_less_fit) { }

CrossoverWork::CrossoverWork(istream &bin_istream) {
    int class_id = bin_istream.get();
    
    if (class_id != CrossoverWork::class_id) {
        Log::fatal("Read the wrong class_id for CrossoverWork");
        exit(0);
    }

    bin_istream.read((char *) &generation_id, sizeof(int32_t));
    bin_istream.read((char *) &group_id, sizeof(int32_t));

    more_fit = new RNN_Genome(bin_istream);
    less_fit = new RNN_Genome(bin_istream);
}

CrossoverWork::~CrossoverWork() {
    delete more_fit;
    delete less_fit;
}

void CrossoverWork::write_to_stream(ostream &bin_ostream) {
    bin_ostream.put(CrossoverWork::class_id);
    
    bin_ostream.write((char *) &generation_id, sizeof(int32_t));
    bin_ostream.write((char *) &group_id, sizeof(int32_t));

    more_fit->write_to_stream(bin_ostream);
    less_fit->write_to_stream(bin_ostream);
}

RNN_Genome *CrossoverWork::get_genome(GenomeOperators &operators) {
    RNN_Genome *genome = operators.crossover(more_fit, less_fit);
    genome->set_generation_id(generation_id);
    genome->set_group_id(group_id);
    return genome;
}

int32_t CrossoverWork::get_class_id() { return CrossoverWork::class_id; }
