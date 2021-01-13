#include "work.hxx"

Work *read_from_stream(istream &bin_istream) {
    int class_id = bin_istream.peek();

    if (class_id == MutationWork::class_id)
        return MutationWork(bin_istream);
    else if (class_id == CrossoverWork::class_id)
        return CrossoverWork(bin_istream);
    else if (class_id == TerminateWork::class_id)
        return TerminateWork(bin_istream);
    
    Log::fatal("encountered unrecognized work class_id %d", class_id);
    exit(0);

    // to satisfy a warning
    return NULL;
}

void set_generation_id(int32_t generation_id) { this->generation_id = generation_id; }

void set_group_id(int32_t group_id) { this->group_id = group_id; }

#include "mutation_work.cxx"
#include "crossover_work.cxx"
#include "terminate_work.cxx"
#include "initialize_work.cxx"
