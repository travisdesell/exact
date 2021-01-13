class CrossoverWork : public Work {
    private:
        static constexpr int32_t class_id = 1;
        
        RNN_Genome *more_fit;
        RNN_Genome *less_fit;
        const int32_t island_id;
    
    public:
        CrossoverWork(RNN_Genome *more_fit, RNN_Genome *less_fit, int32_t generation_id=0, int32_t generation_island=0);
        CrossoverWork(istream &bin_istream);
        ~CrossoverWork();

        void write_to_stream(ostream &bin_ostream);
        RNN_Genome *get_genome(GenomeOperators &operators);
        int32_t get_class_id();
};
