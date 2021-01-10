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

#include "work/mutation_work.cxx"
#include "work/crossover_work.cxx"
#include "work/terminate_work.cxx"
