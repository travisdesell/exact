InitializeWork::InitializeWork(int32_t node_ic, int32_t edge_ic) 
    : Work(-1, -1),
      node_innovation_count(node_ic), 
      edge_innovation_count(edge_ic) { }

InitializeWork::InitializeWork(istream &bin_istream) {
    int class_id = bin_istream.get();
    if (class_id != InitializeWork::class_id) {
        Log::fatal("Read the wrong class_id for InitializeWork");
        exit(0);
    }

    bin_istream.read((char *) &node_innovation_count, sizeof(int32_t));
    bin_istream.read((char *) &edge_innovation_count, sizeof(int32_t));
}

InitializeWork::~InitializeWork() {}

void InitializeWork::update_genome_operators(int32_t rank, GenomeOperators &operators) {
    operators.set_edge_innovation_count(edge_innovation_count + rank);
    operators.set_node_innovation_count(node_innovation_count + rank);
}

void InitializeWork::write_to_stream(ostream &bin_ostream) {
    bin_ostream.put(InitializeWork::class_id);
    bin_ostream.write((char *) &node_innovation_count, sizeof(int32_t));
    bin_ostream.write((char *) &edge_innovation_count, sizeof(int32_t));
}

RNN_Genome *InitializeWork::get_genome(GenomeOperators &operators) {
    return NULL;
}

int32_t InitializeWork::get_class_id() { return InitializeWork::class_id; }
