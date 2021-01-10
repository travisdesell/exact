class CrossoverWork : public Work {
    private:
        static constexpr int32_t class_id = 1;
        
        RNN_Genome *more_fit;
        RNN_Genome *less_fit;
    
    public:
        CrossoverWork(RNN_Genome *more_fit, RNN_Genome *less_fit);
        CrossoverWork(istream &bin_istream);
        ~CrosoverWork();

        void write_to_stream(ostream &bin_ostream);
        RNN_Genome *get_genome(GenomeOperators &operators);
        int32_t get_class_id();
};
