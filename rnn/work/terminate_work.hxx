class TerminateWork : public Work {
    private:
        static constexpr int32_t class_id = 2;

    public:
        TerminateWork();
        TerminateWork(istream &bin_istream);
        ~TerminateWork();

        void write_to_stream(ostream &bin_ostream);
        RNN_Genome *get_genome(GenomeOperators &operators);
        int32_t get_class_id();
};
