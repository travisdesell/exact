#ifndef EDGE_PHEROMONE_HXX
#define EDGE_PHEROMONE_HXX

#include <iostream>


class EDGE_Pheromone {
    private:
        int32_t edge_innovation_number;

        double edge_pheromone;

        int depth;

        int input_innovation_number;
        int output_innovation_number;

    public:

        EDGE_Pheromone(int32_t _edge_innovation_number, double _edge_pheromone, int _depth, int32_t _input_innovation_number, int32_t _output_innovation_number);
        int32_t get_edge_innovation_number();
        double get_edge_phermone();
        int get_depth();

        int32_t get_input_innovation_number();
        int32_t get_output_innovation_number();
        void set_edge_phermone(double p);

        friend class ACNNTO;
};


#endif
