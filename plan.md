Goal is to create arbitrary hierarchical island topologies. 
The leaf node level of this graph would be islands, next would be groups of islands (archipelagos), next would be clusters of archipelagos, etc.
There are a few reasons to do this; the primary motivating reason is for scalability: EXAMM as it stands is ultimately bottlenecked by the master process as are all master-worker algos, and large numbers of cores are going to put more pressure on the master process to be fast.
The other reason is secondary but potentially interesting: using complex island topologies may have impacts on preserving genetic diversity.
EXAMM stands out from the crowd of island based evo algorithms because most of them are not master-worker based.
Usually, each island is either its own process or a leader of some kind that handles results from workers that only evaluate genomes from a single island.

This parallel design by EXAMM is largely related to the format of its genomes. EXAMM is an algorithm derivative of NEAT, the important facts being: genomes basically neural networks and each node and edge in the network has a unique number associated with it.
These unique numbers are very convenient for performing crossover on neural networks as it allows you to compare any two neural networks in linear time. Graph comparison is a very complex operation if there is no extra bookkeeping to give an identity to each node.
To keep things simple, EXAMM keeps track of a counter that gets incremented every time a new component is added to a neural network so there is no repeated innovation numbers.
This counter only exists in the main EXAMM process, and workers cannot access or modify it. So, the master process has to perform all mutations and crossovers.

A relatively simple approach can be used to circumvent this limitation.
Basically if you have N processes, each process is assigned a modular congruence class mod N to use for innovation numbers.
There are N congruence classes mod N, and they are 0 through N-1. Each process recieves one of these and then may only use innovation numbers that satisfy the formula x + aN where x is their congruence class, and A is any integer. There will be no overlap between the IN numbers used by separate processes this way.

Now that we have a way to guarentee that separate processes wont re-use innovation numbers, we can perform mutation and crossover on any process. This removes the constraint that causes the master process to limit scalability. Now, it would be entirely possible for separate islands to have their own groups of workers that only report to that island manager. This could increase scalability by a factor of the number of separate islands used, e.g. 2 islands would spread the work of the master among two island managers, effectively cutting the load in half. Of course, there will be some additional communication costs for separate islands to share genetic information and for the collection of results.

The more traditional parallel island strategy for separating populations isn't the only new opportunity. It is now possible to create arbitrary topologies, in terms of both island hierarchy and parallelization.
The strategy described here is meant to maximize flexability but also allow for extremely efficient design.

The population structure will be a tree-like mesh of some kind, and the foundation will be the leaf nodes. This is where the populations will reside. Each node acts like a little version of EXAMM running its own neural architecture search.
This little instance of EXAMM can have multiple islands and should certainly have many workers evaluating for it. This presents a problem though: how will these leaf nodes share genetic information with each other for crossover?
The solution employed here is for each leaf node to have a set of edges between other nodes, these nodes are the "neighbors" (not necessarily leaf nodes only). Every time a well performing genome (say, new-local best on that leaf node)  is found, it will send a copy to all of its neighbors. 
Neighboring nodes will then keep a FIFO queue of these genoems (this queue should have a max length) that will be drawn from in the event that a inter-region crossover should occur.

As mentioned before, leaf nodes are not the only nodes that will recieve genomes for crossover. One level above leaf nodes are nodes that act as more of managers without doing any actual evaluation of genomes themselves. The leaf nodes are the workers of these manager nodes in a sense. They can serve as a layer of separation between leaf nodes that aren't directly connected, allowing several entirely isolated groups of islands to share a small amount of genetic information with one another. They serve a second purpose as well, that is to log results. Manager nodes should keep track of the best genome they've seen and if they recieve a new best, they should send it up to its own manager. Leaf nodes should also send occasional messages about how many genomes they've evaluated, either on its own or along with the genomes they send.

The root node in this hierarchy should serve as a master. The master should recieve all best-genomes direct children see as well as keep a running total of the number of genomes that have been evaluated by all descendents. This will create a log of results that isn't as fine grained as vanilla EXAMM (normally with EXAMM, every evaluated genome gets an entry in the results loG), but in exchange for this reduced accuracy there will be a significant reduction in communcation costs while also not missing any new global best genomes.`


# Class Structure

ExammNode {
    typedef node_id int;
    
    static const int MAX_CROSSOVER_GENOMES = ...;

    // Send these children best-genomes that this node recieves.
    // Children should also be sending this node their best-found genomes
    list<node_id> children;
    // If this node recieves a new-best from a child, send it up to the parents as well.
    const list<node_id> parents;
    // These are nodes on the same level as this node. They will receive new-best genomes found in this node.
    const list<node_id> neighbors;
    // Genomes that will be shared with children and neighbors for crossover. This should have a max length
    // defined by the constant above.
    list<unique_ptr<genome>> genomes;
    // The number of genomes that have been evaluated  
    uint unrecorded_genome_evals;
    // The number of termination messages recieved. Once this is equal to the number of parents, the algorithm should terminate.
    uint terminate_count;

    // When sending to parents, only ONE parent should propagate the genome upwards
    void send_genome_to_parents(genome *) = send_genome_to(genome *, parents);
    void send_genome_to_neighbors(genome *) = send_genome_to(genome *, neighbors);
    void send_genome_to(genome *, destinations);
    void terminate();
    // Send to one parent.
    void send_genome_evals();

    unique_ptr<Message> recv_message();

    void run();
};

ExammLeaf : ExammNode {
    // Leaf nodes have no children (workers are sort of children, but in a different relationship hierarchy)
    children = [];

    // list of all workers that will be evaluating genomes for this leaf.
    list<node_id> workers;

    unique_ptr<SpeciationStrategy> ss;

    void run();
};

ExammMaster : ExammNode {
    
};

Msg {
    
    public:
        enum msg_ty {
            MSG, // This should never be instantiated on its own
            GENOME,
            GENOME_REQUEST,
            WORK,
            GENOME_RESULT,
            EVAL_ACCOUNTING,
            GENOME_SHARE,
        };
    virtual get_class_id() { return msg_ty::MSG; }
};
GenomeRequestMsg : Msg {};
WorkMsg : Msg {
    enum { crossover, mutation } work_type;
    
    union work_args {
        struct {
            unique_ptr<genome> g;
            uint32_t n_mutations;
        } mutation;
        struct {
            vector<unique_ptr<genome>> parents;
            uint32_t n_parents;
        } crossover;

        ~work_args() {}
    } args;

    // do the crossover / mutation
    unique_ptr<genome> get_genome()

    ~WorkMsg() {
        switch (work_type) {
            case crossover: args.crossover.parents.~vector(); break;
            case mutation:  args.mutation.g.~unique_ptr(); break;
        }
    }
};

// Used by worker to send results.
GenomeResultMsg : Msg {
    // information about where the genome should go
    unique_ptr<genome> g;
};

EvalAccountingMsg : Msg {
    uint32_t n_evals;
};

// Used by non-workers (leafs or nodes) to share genomes.
GenomeShareMsg : EvalAccountingMsg {
    // Only should contain the genome and possibly some tabulation results.
    unique_ptr<genome> g;
    bool propagate;
};
