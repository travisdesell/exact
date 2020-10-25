import struct
from graphviz import Digraph
from sys import argv
import numpy as np
from os import listdir as ld
from subprocess import call


# normalizaion_parameters = ['Coal_Feeder_Rate', 'Conditioner_Inlet_Temp', 'Conditioner_Outlet_Temp', 'Main_Flm_Int', 'Primary_Air_Flow', 'Primary_Air_Split', 'Secondary_Air_Flow', 'Secondary_Air_Split', 'Supp_Oil_Flow', 'System_Secondary_Air_Flow_Total', 'Tertiary_Air_Split', 'Total_Comb_Air_Flow'}
def read_bin(filename, verbose):
    count=0
    Nodes    = list()
    Edges    = []
    RecEdges = []
    max_rec_pheromone = 0
    max_pheromone = 0
    with open(filename, "rb") as f:
        colony_id = struct.unpack("<i",f.read(4))[0]
        if verbose: print "Colony ID: {}".format(colony_id)
        recurrence_depth = struct.unpack("<i",f.read(4))[0]
        if verbose: print "Recurrent Depth: {}".format(recurrence_depth)
        no_nodes = struct.unpack("<i",f.read(4))[0]
        if verbose: print "Number of Nodes in Colony: {}".format(no_nodes)



        for D in range( no_nodes ):
            node_id = struct.unpack("<i",f.read(4))[0]
            if verbose: print "Node ID: {}".format(node_id)
            no_node_type = struct.unpack("<i",f.read(4))[0]
            if verbose: print "Number of Node Types: {}".format(no_node_type)
            if node_id!=-1:
                for x in range(no_node_type):
                    node_pheromone = struct.unpack("d",f.read(8))[0]
                    if verbose: print "Node Pheromone: {}".format(node_pheromone)

            Nodes.append( node_id )

            no_pheromone_lines = struct.unpack("<i",f.read(4))[0]
            for l in range(no_pheromone_lines):
                edge_id = struct.unpack("<i",f.read(4))[0]
                if verbose: print "Edge ID: {}".format(edge_id)
                edge_pheromone = struct.unpack("d",f.read(8))[0]
                if verbose: print "Edge Pheromone: {}".format(edge_pheromone)
                edge_depth = struct.unpack("<i",f.read(4))[0]
                if verbose: print "Edge Depth: {}".format(edge_depth)

                if max_pheromone<edge_pheromone and edge_depth==0:
                    max_pheromone = edge_pheromone
                if max_rec_pheromone<edge_pheromone and edge_depth!=0:
                    max_rec_pheromone = edge_pheromone

                edge_IN = struct.unpack("<i",f.read(4))[0]
                if verbose: print "Edge I/P Node: {}".format(edge_IN)
                edge_OUT = struct.unpack("<i",f.read(4))[0]
                if verbose: print "Edge O/P Node: {}".format(edge_OUT)

                if edge_depth==0:
                    Edges.append([edge_id, edge_IN, edge_OUT, edge_pheromone])
                else:
                    RecEdges.append([edge_id, edge_IN, edge_OUT, edge_pheromone, edge_depth])

    print "\tMaximum Pheromone: ", max_pheromone, " -  Maximum RecPheromone: ", max_rec_pheromone
    return [colony_id, recurrence_depth, np.array(Nodes), np.array(Edges), np.array(RecEdges), max(max_pheromone, max_rec_pheromone)]

def build_gv(colony_id, recurrence_depth, Nodes, Edges, RecEdges, max_pheromone ):
    dot = Digraph( comment='colony_{}'.format(colony_id) )

    d = 0
    # for d in range(recurrence_depth):
    for node_id in Nodes:
        node_ID = node_id
        if node_id==-1 :
            node_ID = "Start"
        dot.node( "{}_{}".format(node_id, d), str(node_ID), rank=str(d), style='filled', color='/dark28/{}'.format(d+1))
    dot.node( "{}_{}".format(str(-(node_id+1)), d), "Output", rank=str(d), style='filled', color='/dark28/{}'.format(d+1))



    # for d in range(recurrence_depth):
    for edge in Edges:
        edge_pheromone_ = int( edge[-1]/max_pheromone  * 99 )
        s = ""
        if edge_pheromone_<=1:
            s = "invis"
        out_node = int(edge[2])
        in_node = int(edge[1])
        # if in_node==-1 :
        #     node_id = "Start"
        # if node_id<-1:
        #     node_id = "Output_" + str(d)
        dot.edge( "{}_{}".format(in_node, d), "{}_{}".format(out_node, d) , color="grey{}".format(100-edge_pheromone_), style=s)


    for edge in RecEdges:
        edge_pheromone_ = int( edge[-2]/max_pheromone  * 99 )
        s = ""
        if edge_pheromone_<=1:
            s = "invis"
        dot.edge( "{}_{}".format(int(edge[1]), int(edge[-1])-1), "{}_{}".format(int(edge[2]), int(edge[-1])) , color="grey{}".format(edge_pheromone_), style=s)

    dot.render('colony_{}.gv'.format(colony_id), view=False)


def convert_to_png(file, resize):
    cmd = "convert -flatten {} -resize {}% {}.png".format(file, resize, file[:-4])
    call(cmd.split())

def convert_to_gif():
    cmd = "convert -layers OptimizePlus -delay 100 colony_?.gv.png -delay 100 colony_?[0123456789].gv.png -delay 100 colony_1?[0123456789].gv.png -delay 100 colony_200.gv.png colony.mp4"
    call(cmd.split())

def main():
    Reading_results = list()
    for i in range(1,201):
        file = 'colony_{}.bin'.format(i)
        print "FILE: {}  -  ".format(file),

        Reading_results.append( read_bin( file, False ) )
    max_pheromone = max([x[-1] for x in Reading_results])

    for i in range(0,200):
        file = 'colony_{}.bin'.format(i+1)
        print "FILE: {}  -  ".format(file)
        build_gv( Reading_results[i][0], Reading_results[i][1], Reading_results[i][2], Reading_results[i][3], Reading_results[i][4], max_pheromone )
        convert_to_png(file[:-4]+".gv.pdf", '100')
    # convert_to_gif()


if __name__ == "__main__":
    main()
