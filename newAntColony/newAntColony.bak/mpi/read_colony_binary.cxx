// Example program
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
using std::ofstream;

#include <map>

#include <sstream>
using std::ostringstream;

#include <string>
using std::string;
using std::to_string;

using namespace std;

int main(int argc, char** argv)
{
  map<int, int> node_types;
  node_types[1] = 3;
  node_types[2] = 4;
  // cout << "first element in map: " << node_types[1] << endl;



  // ofstream bin_outfile("ttest.bin", ios::out | ios::binary);
  // bin_outfile.write((char*)&node_types[1], sizeof(int));
  // bin_outfile.write((char*)&node_types[2], sizeof(int));
  // bin_outfile.close();

  ifstream bin_infile(argv[1], ios::in | ios::binary);
  // ifstream bin_infile("../build/mpi/marching_ants_experiment/10_0.5_0.5/colony_99.bin", ios::in | ios::binary);
  int n;
  int32_t m;
  double k = 0.0;
  bin_infile.read((char*)&m, sizeof(int32_t));
  cout << "Genome ID: " << m <<endl;
  bin_infile.read((char*)&m, sizeof(int32_t));
  cout << "Recurrent Depth " << m <<endl;


  bin_infile.read((char*)&n, sizeof(int));
  int number_of_nodes = n;
  cout << "Number of Nodes in Colony: " << n <<endl;

  for ( int i=0; i<number_of_nodes; i++) {
    bin_infile.read((char*)&m, sizeof(int32_t));
    cout << "Node ID: " << m <<endl;
    bin_infile.read((char*)&n, sizeof(int));
    cout << "Number of Node Types: " << n <<endl;
    int number_of_node_types = n;
    if (m!=-1){
      for ( int j=0; j<number_of_node_types; j++) {
        bin_infile.read((char*)&k, sizeof(double));
        cout << "Node Pheromone: " << k <<endl;
      }
    }

    bin_infile.read((char*)&n, sizeof(int));
    int number_of_pheromone_lines = n;
    for ( int l=0; l<number_of_pheromone_lines; l++) {
      bin_infile.read((char*)&m, sizeof(int32_t));
      cout << "Edge ID: " << m <<endl;
      bin_infile.read((char*)&k, sizeof(double));
      cout << "Edge Pheromone : " << k <<endl;
      bin_infile.read((char*)&n, sizeof(int));
      cout << "Edge Depth: " << n <<endl;
      bin_infile.read((char*)&m, sizeof(int32_t));
      cout << "Edge IN: " << m <<endl;
      bin_infile.read((char*)&m, sizeof(int32_t));
      cout << "Edge OUT: " << m <<endl;
    }
  }



  bin_infile.close();
  return 0;
}
