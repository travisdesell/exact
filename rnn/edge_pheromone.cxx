#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include "edge_pheromone.hxx"

EDGE_Pheromone::EDGE_Pheromone(int32_t _edge_innovation_number, double _edge_pheromone, int _depth, int _input_innovation_number, int _output_innovation_number) : edge_innovation_number(_edge_innovation_number), edge_pheromone(_edge_pheromone), depth(_depth), input_innovation_number(_input_innovation_number), output_innovation_number(_output_innovation_number)  {
    edge_innovation_number    = _edge_innovation_number;

    edge_pheromone            = _edge_pheromone;

    depth                     = _depth;

    input_innovation_number   = _input_innovation_number;
    output_innovation_number  = _output_innovation_number;

}

int32_t EDGE_Pheromone::get_edge_innovation_number() {
  return edge_innovation_number;
}

double EDGE_Pheromone::get_edge_phermone(){
    return edge_pheromone;
}

int EDGE_Pheromone::get_depth(){
    return depth;
}



int32_t EDGE_Pheromone::get_input_innovation_number() {
  return input_innovation_number;
}

int32_t EDGE_Pheromone::get_output_innovation_number(){
  return output_innovation_number;
}

void EDGE_Pheromone::set_edge_phermone(double p) {
  edge_pheromone = p;
}
