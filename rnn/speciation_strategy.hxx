#ifndef EXAMM_SPECIATION_STRATEGY_HXX
#define EXAMM_SPECIATION_STRATEGY_HXX


class SpeciationStrategy {
    
    public:
        //utility functions
        RNN_Genome* get_best_genome();
        RNN_Genome* get_worst_genome();


        void print_population();
        void write_memory_log(string filename);


        //basic functionality insert/generate
        bool insert_genome(RNN_Genome* genome);

        RNN_Genome* generate_genome();
 }


#endif

