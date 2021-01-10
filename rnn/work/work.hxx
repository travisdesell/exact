#ifndef WORK_HXX
#define WORK_HXX

#include "rnn_genome.hxx"
#include "genome_operators.hxx"

class Work {
    public:
        static Work *read_from_stream(istream &bin_istream);
        
        Work(istream &bin_istream) = 0;
        ~Work() = 0;

        virtual void write_to_stream(ostream &bin_ostream) = 0;
        virtual RNN_Genome *get_genome(GenomeOperators &operators) = 0;
        virtual int32_t get_class_id() = 0;
};

#include "work/mutation_work.hxx"
#include "work/crossover_work.hxx"
#include "work/terminate_work.hxx"

#endif
