#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include <vector>
using namespace std;

#include "edge_pheromone.hxx"
#include "node_pheromone.hxx"


class TT {
  private:
    NODE_Pheromones node_ph;
  public:
    TT(NODE_Pheromones _node_ph);
    double get_node_ph_element(int i);
    EDGE_Pheromone* get_node_ph_line(int i);
    ~TT();
};

TT::~TT(){
// delete node_ph.type_pheromones;
};

TT::TT(NODE_Pheromones _node_ph):node_ph(_node_ph){
};

double TT::get_node_ph_element(int i){
  return node_ph.type_pheromones[i] ;
};

EDGE_Pheromone* TT::get_node_ph_line(int i){
  return node_ph.pheromone_lines[i] ;
};


int main(){
  // EDGE_Pheromone edge_pheromone_1 (11, 1.0, 55, 66);
  // EDGE_Pheromone edge_pheromone_2 (12, 1.0, 88, 99);
  vector<EDGE_Pheromone*> edges;
  EDGE_Pheromone* edge1 = new EDGE_Pheromone(11, 1.0, 55, 66);
  EDGE_Pheromone* edge2 = new EDGE_Pheromone(12, 1.0, 88, 99);
  edges.push_back(edge1);
  edges.push_back(edge2);
  double xx[3] = {1.0, 2.0};
  NODE_Pheromones nn(xx, edges);
  TT tt(nn);
  cout<<"VALUE: "<<tt.get_node_ph_element(1)<<endl;
  cout<<"edge phermone ID: "<<tt.get_node_ph_line(1)->get_edge_innovation_number()<<endl;




  vector<int*> aa;
  int bb[] = {1, 2};
  for (int i=0; i<5; i++){
    int* a = new int[2];
    bb[0]+=1;
    bb[1]+=1;
    memcpy(a, bb, 2*sizeof(int) );
    aa.push_back(a);
    cout<<a[0]<<endl;
  }
  cout<<"aa[0]: "<<aa[0][0]<<" bb[0]: "<<bb[0]<<endl;
  cout<<"aa[4]: "<<aa[4][0]<<" bb[0]: "<<bb[0]<<endl;

  return 0;
}
