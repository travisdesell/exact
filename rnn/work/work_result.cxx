WorkResult::WorkResult(RNN_Genome *_genome) 
    : Work(_genome->get_generation_id(), _genome->get_group_id()), genome(_genome) { }

WorkResult::WorkResult(istream &bin_istream) {
    int class_id = bin_istream.get();

    if (class_id != WorkResult::class_id) {
        Log::fatal("Read the wrong class_id for WorkResult");
        exit(0);
    }

    bin_istream.read((char *) &this->generation_id, sizeof(int32_t));
    bin_istream.read((char *) &this->group_id, sizeof(int32_t));

    genome = new RNN_Genome(bin_istream);
}

WorkResult::~WorkResult() {
    delete genome;
}

void WorkResult::write_to_stream(ostream &bin_ostream) {
    bin_ostream.put(WorkResult::class_id);

    bin_ostream.write((char *) &generation_id, sizeof(int32_t));
    bin_ostream.write((char *) &group_id, sizeof(int32_t));
    
    genome->write_to_stream(bin_ostream);
}

RNN_Genome *WorkResult::get_genome(GenomeOperators &operators) {
    return this->genome->copy();
}

int32_t WorkResult::get_class_id() { return WorkResult::class_id; }
