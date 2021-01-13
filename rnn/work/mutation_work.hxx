class MutationWork : public Work {
    private:
        static constexpr int32_t class_id = 0;

        int32_t n_mutations;
        RNN_Genome *genome;

    public:
        /**
         * The supplied genome must not be null! 
         **/
        MutationWork(RNN_Genome *_genome, int32_t _n_mutations, int32_t generation_id=0, int32_t generation_island=0);
        MutationWork(istream &bin_istream);
        ~MutationWork();

        void write_to_stream(ostream &bin_ostream);
        RNN_Genome *get_genome(GenomeOperators &operators);
        int32_t get_class_id();
};
