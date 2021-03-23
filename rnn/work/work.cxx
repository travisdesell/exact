#include "work.hxx"

Work::Work() : generation_id(-1), group_id(-1) { }
Work::Work(int32_t gen_id, int32_t grp_id) : generation_id(gen_id), group_id(grp_id) { }

RNN_Genome *Work::get_genome(Work *work, GenomeOperators &operators) {
    RNN_Genome *result = work->get_genome(operators);
    if (result)
        operators.finalize_genome(result);
    return result;
}

Work *Work::read_from_array(const char *array, int32_t length) {
    struct membuf : std::streambuf {
        membuf(const char* begin, const char* end) {
            this->setg((char *) begin, (char *) begin, (char *) end);
        }
    };

    membuf mb(array, array + length);
    istream is(&mb);
    return read_from_stream(is);
}

Work *Work::read_from_stream(istream &bin_istream) {
    int class_id = bin_istream.peek();

    if (class_id == MutationWork::class_id)
        return new MutationWork(bin_istream);
    else if (class_id == CrossoverWork::class_id)
        return new CrossoverWork(bin_istream);
    else if (class_id == TerminateWork::class_id)
        return new TerminateWork(bin_istream);
    else if (class_id == InitializeWork::class_id)
        return new InitializeWork(bin_istream);
    else if (class_id == WorkResult::class_id)
        return new WorkResult(bin_istream);
    else if (class_id == TrainWork::class_id)
        return new TrainWork(bin_istream);
    
    Log::fatal("encountered unrecognized work class_id %d", class_id);
    exit(0);

    // to satisfy a warning
    return NULL;
}

void Work::set_generation_id(int32_t generation_id) { this->generation_id = generation_id; }

void Work::set_group_id(int32_t group_id) { this->group_id = group_id; }

#include "mutation_work.cxx"
#include "crossover_work.cxx"
#include "terminate_work.cxx"
#include "initialize_work.cxx"
#include "work_result.cxx"
#include "train_work.cxx"
