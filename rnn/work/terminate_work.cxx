TerminateWork::TerminateWork() {}
TerminateWork::TerminateWork(istream &bin_istream) {
    int class_id = bin_istream.get();
    if (class_id != TerminateWork::class_id) {
        Log::fatal("Read the wrong class_id for TerminateWork");
        exit(0);
    }
}

~TerminateWork::TerminateWork() {}

void TerminateWork::write_to_stream(ostream &bin_ostream) {
    bin_ostream.put(TerminateWork::class_id);
}

RNN_Genome *TerminateWork:get_genome(GenomeOperators &operators) {
    return NULL;
}

int32_t TerminateWork::get_class_id() { return TerminateWork::class_id; }
