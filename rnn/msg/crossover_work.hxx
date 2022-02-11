#ifndef CROSSOVER_WORK_HXX
#define CROSSOVER_WORK_HXX 1
#include "work.hxx"

class CrossoverWork : public Work {
    private: 
        vector<RNN_Genome *> parents;
    
    public:
        static constexpr int32_t class_id = 1;
       
        CrossoverWork(vector<RNN_Genome *> parents, int32_t generation_id=-1, int32_t generation_island=-1);
        CrossoverWork(istream &bin_istream);
        ~CrossoverWork();

        void write_to_stream(ostream &bin_ostream);
        RNN_Genome *get_genome(GenomeOperators &operators);
        int32_t get_class_id();
};

#endif
