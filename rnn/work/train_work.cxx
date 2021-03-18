TrainWork::TrainWork(RNN_Genome *_genome, int32_t _generation_id, int32_t _group_id) 
    : Work(_generation_id, _group_id), genome(_genome) { }

TrainWork::TrainWork(istream &bin_istream) {
    int class_id = bin_istream.get();

    if (class_id != TrainWork::class_id) {
        Log::fatal("Read the wrong class_id for TrainWork");
        exit(0);
    }

    bin_istream.read((char *) &generation_id, sizeof(int32_t));
    bin_istream.read((char *) &group_id, sizeof(int32_t));

    genome = new RNN_Genome(bin_istream);
}

TrainWork::~TrainWork() {
    delete genome;
}

void TrainWork::write_to_stream(ostream &bin_ostream) {
    bin_ostream.put(TrainWork::class_id);

    bin_ostream.write((char *) &generation_id, sizeof(int32_t));
    bin_ostream.write((char *) &group_id, sizeof(int32_t));
    
    genome->write_to_stream(bin_ostream);
}

RNN_Genome *TrainWork::get_genome(GenomeOperators &operators) {
    RNN_Genome *clone = genome->copy();
    
    clone->set_group_id(group_id);
    clone->set_generation_id(generation_id);

    return clone;
}

int32_t TrainWork::get_class_id() { return TrainWork::class_id; }
