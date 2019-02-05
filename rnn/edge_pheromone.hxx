#ifndef EDGE_PHEROMONE_HXX
#define EDGE_PHEROMONE_HXX

#include <iostream>


class EDGE_Pheromone {
    private:
        int32_t edge_innovation_number;

        double edge_pheromone;

        // bool enabled;
        // bool forward_reachable;
        // bool backward_reachable;

        int32_t input_innovation_number;
        int32_t output_innovation_number;

    public:

        EDGE_Pheromone(int32_t _edge_innovation_number, double _edge_pheromone
                                                        , int32_t _input_innovation_number, int32_t _output_innovation_number);
        int32_t get_edge_innovation_number();
        double get_edge_phermone();
        int32_t get_input_innovation_number();
        int32_t get_output_innovation_number();
        void set_edge_phermone(double p);

        friend class ANT_COLONY;
};


#endif
