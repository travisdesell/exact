class InitializeWork : public Work {
private:
    static constexpr int32_t class_id = 3;

    const int32_t node_innovation_count;
    const int32_t edge_innovation_count;

public:
    InitializeWork(int32_t node_ic, int32_t edge_ic);
    InitializeWork(istream &bin_istream);
    ~InitializeWork();

    void write_to_stream(ostream &bin_ostream);
    RNN_Genome *get_genome(GenomeOperators &operators);
    int32_t get_class_id();
};
