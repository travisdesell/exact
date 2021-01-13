#ifndef WORK_HXX
#define WORK_HXX

#include "../rnn_genome.hxx"
#include "../genome_operators.hxx"

class Work {
    private:
        int32_t generation_id;
        int32_t group_id;

    public:
        static Work *read_from_stream(istream &bin_istream);
        
        Work(int32_t generation_id, int32_t generation_island);
        Work(istream &bin_istream);
        ~Work();

        virtual void write_to_stream(ostream &bin_ostream) = 0;
        virtual RNN_Genome *get_genome(GenomeOperators &operators) = 0;
        virtual int32_t get_class_id() = 0;
        void set_generation_id(int32_t generation_id);
        void set_group_id(int32_t group_id);
};

#include "mutation_work.hxx"
#include "crossover_work.hxx"
#include "terminate_work.hxx"
#include "initialize_work.hxx"

#endif
