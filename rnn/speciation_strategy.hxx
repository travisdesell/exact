#ifndef EXAMM_SPECIATION_STRATEGY_HXX
#define EXAMM_SPECIATION_STRATEGY_HXX


class SpeciationStrategy {
    
    public:
        //utility functions
        virtual double get_best_fitness() = 0;
        virtual double get_worst_fitness() = 0;

        virtual RNN_Genome* get_best_genome() = 0;
        virtual RNN_Genome* get_worst_genome() = 0;


        virtual void print_population() = 0;
        virtual void write_memory_log(string filename) = 0;


        //basic functionality insert/generate
        virtual bool insert_genome(RNN_Genome* genome) = 0;

        virtual RNN_Genome* generate_genome() = 0;
};


#endif

