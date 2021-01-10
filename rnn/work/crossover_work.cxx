CrossoverWork::CrossoverWork(RNN_Genome *_more_fit, RNN_Genome *_less_fit)
    : more_fit(_more_fit), less_fit(_less_fit) { }

CrossoverWork::CrossoverWork(istream &bin_istream) {
    int class_id = bin_istream.get();
    if (class_id != CrossoverWork::class_id) {
        Log::fatal("Read the wrong class_id for CrossoverWork");
        exit(0);
    }

    more_fit = new RNN_Genome(bin_istream);
    less_fit = new RNN_Genome(bin_istream);
}

CrossoverWork::~CrossoverWork(istream &bin_istream) {
    delete more_fit;
    delete less_fit;
}

void CrossoverWork::write_to_stream(ostream &bin_ostream) {
    bin_ostream.put(CrossoverWork::class_id);
    
    more_fit->write_to_stream(bin_ostream);
    less_fit->write_to_stream(bin_ostream);
}

RNN_Genome *CrossoverWork::get_genome(GenomeOperatorss &operators) {
    return operatorss.crossover(more_fit, less_fit);
}

int32_t CrossoverWork::get_class_id() { return CrossoverWork::class_id; }
