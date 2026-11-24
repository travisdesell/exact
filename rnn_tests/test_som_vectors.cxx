#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
using std::string;
#include <vector>
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "examm/examm.hxx"
#include "examm/island_speciation_strategy.hxx"
#include "examm/som_unit.hxx"
#include "examm/som_vector_space.hxx"
#include "rnn/generate_nn.hxx"
#include "rnn/genome_property.hxx"
#include "rnn/rnn_genome.hxx"
#include "rnn/rnn_node_interface.hxx"
#include "weights/weight_rules.hxx"

static int32_t g_pass = 0;
static int32_t g_fail = 0;

static void check(bool condition, const string& label) {
    if (condition) {
        Log::info("PASS: %s\n", label.c_str());
        g_pass++;
    } else {
        Log::error("FAIL: %s\n", label.c_str());
        g_fail++;
    }
}

static string format_vector(const vector<double>& v) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); i++) {
        if (i > 0) {
            oss << ", ";
        }
        oss << v[i];
    }
    oss << "] (len=" << v.size() << ")";
    return oss.str();
}

static string format_innovations(const vector<int32_t>& v) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); i++) {
        if (i > 0) {
            oss << ", ";
        }
        oss << v[i];
    }
    oss << "]";
    return oss.str();
}

static void print_unit(const string& label, const SomUnit& unit) {
    Log::info("%s node_vector=%s\n", label.c_str(), format_vector(unit.node_vector).c_str());
    Log::info("%s edge_vector=%s\n", label.c_str(), format_vector(unit.edge_vector).c_str());
}

static void print_catalog(const SomVectorSpace& space) {
    Log::info(
        "catalog node_innovations=%s node_vector_size=%zu\n", format_innovations(space.get_node_innovations()).c_str(),
        space.get_node_vector_size()
    );
    Log::info(
        "catalog edge_innovations=%s edge_vector_size=%zu\n", format_innovations(space.get_edge_innovations()).c_str(),
        space.get_edge_vector_size()
    );
}

static void print_stage_header(const string& stage) {
    Log::info("\n========== %s ==========\n", stage.c_str());
}

static void report_expand(bool grew, const string& context) {
    if (grew) {
        Log::info(
            "NEW INNOVATION(S) ADDED during %s — SomUnit vectors were expanded with trailing zeros\n", context.c_str()
        );
    } else {
        Log::info("No new innovation numbers during %s — SomUnit vector sizes unchanged\n", context.c_str());
    }
}

static void report_compress(bool removed, const string& context) {
    if (removed) {
        Log::info("VECTOR COMPRESSED during %s — inactive innovations removed from catalog and units\n", context.c_str());
    } else {
        Log::info("No compression during %s — all catalogued innovations still active in some genome\n", context.c_str());
    }
}


static void test_som_unit_methods() {
    print_stage_header("SomUnit method unit tests");

    SomUnit empty;
    check(empty.node_vector.empty() && empty.edge_vector.empty(), "default SomUnit starts empty");

    SomUnit sized(3, 2);
    check(sized.node_vector.size() == 3 && sized.edge_vector.size() == 2, "SomUnit(node_size, edge_size) sizes");
    check(sized.node_vector[0] == 0.0 && sized.edge_vector[1] == 0.0, "SomUnit constructor zeros contents");

    sized.resize(4, 1);
    check(sized.node_vector.size() == 4 && sized.edge_vector.size() == 1, "resize changes lengths");
    check(sized.node_vector[3] == 0.0 && sized.edge_vector[0] == 0.0, "resize fills with zeros");

    sized.node_vector[0] = 1.5;
    sized.append_node_zeros(2);
    check(sized.node_vector.size() == 6, "append_node_zeros grows node_vector");
    check(sized.node_vector[0] == 1.5, "append_node_zeros preserves existing values");
    check(sized.node_vector[4] == 0.0 && sized.node_vector[5] == 0.0, "append_node_zeros appends zeros");

    sized.edge_vector[0] = -2.0;
    sized.append_edge_zeros(3);
    check(sized.edge_vector.size() == 4, "append_edge_zeros grows edge_vector");
    check(sized.edge_vector[0] == -2.0, "append_edge_zeros preserves existing values");
    check(sized.edge_vector[1] == 0.0 && sized.edge_vector[3] == 0.0, "append_edge_zeros appends zeros");

    // node_vector: [1.5, 0, 0, 0, 0, 0] — erase 2 weights starting at offset 1
    sized.erase_node_slice(1, 2);
    check(sized.node_vector.size() == 4, "erase_node_slice shortens node_vector");
    check(sized.node_vector[0] == 1.5, "erase_node_slice keeps prefix");

    // edge_vector: [-2, 0, 0, 0] — erase one at offset 0
    sized.erase_edge_slice(0, 1);
    check(sized.edge_vector.size() == 3, "erase_edge_slice shortens edge_vector");
    check(sized.edge_vector[0] == 0.0, "erase_edge_slice removes requested slice");

    sized.erase_node_slice(0, 0);
    check(sized.node_vector.size() == 4, "erase_node_slice with count 0 is a no-op");
    sized.erase_edge_slice(0, 0);
    check(sized.edge_vector.size() == 3, "erase_edge_slice with count 0 is a no-op");

    print_unit("SomUnit after method tests", sized);
}

static void test_euclidean_distance() {
    print_stage_header("SomVectorSpace::euclidean / distance unit tests");

    vector<double> a = {0.0, 0.0};
    vector<double> b = {3.0, 4.0};
    double d = SomVectorSpace::euclidean(a, b);
    check(std::fabs(d - 5.0) < 1e-9, "euclidean([0,0],[3,4]) == 5");

    vector<double> node_a = {1.0};
    vector<double> edge_a = {2.0};
    vector<double> node_b = {1.0};
    vector<double> edge_b = {2.0};
    check(SomVectorSpace::distance(node_a, edge_a, node_b, edge_b) == 0.0, "distance of identical vectors is 0");

    SomUnit unit;
    unit.node_vector = {0.0};
    unit.edge_vector = {0.0};
    double unit_d = SomVectorSpace::distance(unit, node_a, edge_a);
    check(std::fabs(unit_d - (1.0 + 2.0)) < 1e-9, "distance(SomUnit, genome vectors) sums node+edge L2");
}

static EXAMM* make_examm(RNN_Genome* seed_genome, WeightRules* weight_rules, GenomeProperty* genome_property) {
    vector<string> possible_node_types = {"simple"};

    IslandSpeciationStrategy* strategy = new IslandSpeciationStrategy(
        /*number_of_islands*/ 1,
        /*max_island_size*/ 2,
        /*mutation_rate*/ 1.0,
        /*intra*/ 0.0,
        /*inter*/ 0.0,
        seed_genome,
        /*island_ranking_method*/ "",
        /*repopulation_method*/ "",
        /*extinction_event_generation_number*/ 0,
        /*num_mutations*/ 1,
        /*islands_to_exterminate*/ 0,
        /*max_genomes*/ 100,
        /*repeat_extinction*/ false,
        /*start_filled*/ false,
        /*transfer_learning*/ false,
        /*transfer_learning_version*/ "",
        /*seed_stirs*/ 0,
        /*tl_epigenetic_weights*/ false,
        possible_node_types,
        /*is_harada_selection*/ 0,
        /*harada_selection_ratio*/ 0.0,
        /*is_sweet*/ 0
    );

    EXAMM* examm = new EXAMM(
        /*island_size*/ 2,
        /*number_islands*/ 1,
        /*max_genomes*/ 100,
        /*max_wallclock_seconds*/ 0,
        strategy,
        weight_rules,
        genome_property,
        /*output_directory*/ "",
        /*save_genome_option*/ "all_best_genomes",
        /*generate_op_log*/ false,
        /*generate_visualization_json*/ false,
        /*growth_phase_genomes*/ 0,
        /*reduction_phase_genomes*/ 0,
        /*genome_size_log*/ 0,
        /*is_harada_selection*/ 0,
        /*harada_selection_ratio*/ 0.0,
        /*is_sweet*/ 0
    );
    examm->set_possible_node_types(possible_node_types);
    return examm;
}

static void encode_pair(
    SomVectorSpace& space, RNN_Genome* g1, RNN_Genome* g2, vector<double>& n1, vector<double>& e1, vector<double>& n2,
    vector<double>& e2
) {
    space.encode_genome(g1, n1, e1);
    space.encode_genome(g2, n2, e2);
    Log::info("genome1 encoded node_vector=%s\n", format_vector(n1).c_str());
    Log::info("genome1 encoded edge_vector=%s\n", format_vector(e1).c_str());
    Log::info("genome2 encoded node_vector=%s\n", format_vector(n2).c_str());
    Log::info("genome2 encoded edge_vector=%s\n", format_vector(e2).c_str());
}

static void test_genome_scenario() {
    print_stage_header("Genome scenario: identical -> mutate -> disable");

    WeightRules* weight_rules = new WeightRules();
    GenomeProperty* genome_property = new GenomeProperty();
    genome_property->set_bp_iterations(1);

    // Small EXAMM-style feed-forward seed (inputs + hidden + outputs), randomly initialized.
    vector<string> inputs = {"in1"};
    vector<string> outputs = {"out1"};
    RNN_Genome* seed = create_ff(inputs, /*hidden_layers*/ 1, /*hidden_nodes*/ 2, outputs, /*max_rec_depth*/ 1, weight_rules);
    seed->initialize_randomly();
    seed->set_bp_iterations(1);

    Log::info(
        "seed genome: nodes=%d edges=%d rec_edges=%d weights=%d\n", seed->get_node_count(),
        seed->get_enabled_edge_count(), seed->get_enabled_recurrent_edge_count(), seed->get_number_weights()
    );

    EXAMM* examm = make_examm(seed, weight_rules, genome_property);

    // Mirror island initialization: start from the randomly initialized seed genome, then
    // copy so the pair is identical before one side is mutated with EXAMM::mutate.
    RNN_Genome* genome1 = seed->copy();
    genome1->assign_reachability();
    Log::info(
        "genome1 from randomly initialized seed: nodes=%d enabled_nodes=%d enabled_edges=%d\n",
        genome1->get_node_count(), genome1->get_enabled_node_count(), genome1->get_enabled_edge_count()
    );

    RNN_Genome* genome2 = genome1->copy();
    genome2->assign_reachability();
    Log::info("genome2 is an identical copy of genome1\n");

    SomVectorSpace space;
    vector<SomUnit> units(2);

    vector<double> n1, e1, n2, e2;

    // ----- Stage A: identical genomes -----
    print_stage_header("Stage A: two identical genomes");
    bool grew = space.expand_from_genome(genome1, units);
    report_expand(grew, "expand_from_genome(genome1)");
    check(grew, "first expand registers seed innovations");

    // Sentinel so later expands prove existing entries are preserved and only the tail is zero-padded.
    if (!units[0].node_vector.empty()) {
        units[0].node_vector[0] = 42.0;
    }
    if (!units[0].edge_vector.empty()) {
        units[0].edge_vector[0] = 7.0;
    }

    grew = space.expand_from_genome(genome2, units);
    report_expand(grew, "expand_from_genome(genome2) on identical twin");
    check(!grew, "identical twin adds no new innovations");

    print_catalog(space);
    print_unit("unit[0]", units[0]);
    print_unit("unit[1]", units[1]);
    check(units[0].node_vector.size() == space.get_node_vector_size(), "unit[0] node size matches catalog after expand");
    check(units[0].edge_vector.size() == space.get_edge_vector_size(), "unit[0] edge size matches catalog after expand");
    if (!units[0].node_vector.empty()) {
        check(units[0].node_vector[0] == 42.0, "expand preserves existing unit node weights");
    }

    encode_pair(space, genome1, genome2, n1, e1, n2, e2);
    double dist = SomVectorSpace::distance(n1, e1, n2, e2);
    Log::info("distance(genome1, genome2) = %f\n", dist);
    check(dist == 0.0, "identical genomes have distance 0");

    double unit_dist = SomVectorSpace::distance(units[1], n1, e1);
    Log::info("distance(unit[1]=zeros, genome1) = %f\n", unit_dist);
    check(unit_dist >= 0.0, "distance(unit, genome) is non-negative");

    // ----- Stage B: mutate genome2 -----
    print_stage_header("Stage B: mutate genome2 with EXAMM::mutate");
    size_t node_size_before = space.get_node_vector_size();
    size_t edge_size_before = space.get_edge_vector_size();
    size_t unit0_node_before = units[0].node_vector.size();

    // Retry from a healthy backup until mutation introduces new innovations (clone-only
    // attempts are possible because EXAMM samples ops stochastically).
    RNN_Genome* genome2_backup = genome2->copy();
    grew = false;
    for (int32_t attempt = 1; attempt <= 20 && !grew; attempt++) {
        delete genome2;
        genome2 = genome2_backup->copy();
        genome2->assign_reachability();
        Log::info("Mutation attempt %d/20\n", attempt);
        examm->mutate(/*max_mutations*/ 3, genome2);
        genome2->assign_reachability();
        Log::info(
            "After mutate: nodes=%d enabled_nodes=%d enabled_edges=%d\n", genome2->get_node_count(),
            genome2->get_enabled_node_count(), genome2->get_enabled_edge_count()
        );
        grew = space.expand_from_genome(genome2, units);
        report_expand(grew, "expand_from_genome(genome2) after mutation attempt");
    }
    delete genome2_backup;

    print_catalog(space);
    print_unit("unit[0] after mutate-expand", units[0]);
    print_unit("unit[1] after mutate-expand", units[1]);

    if (grew) {
        check(space.get_node_vector_size() + space.get_edge_vector_size() > node_size_before + edge_size_before,
              "catalog grew after mutation introduced new innovations");
        check(units[0].node_vector.size() >= unit0_node_before, "unit vectors grew or stayed when catalog expanded");
        if (!units[0].node_vector.empty()) {
            check(units[0].node_vector[0] == 42.0, "expand after mutation preserves existing unit node weights");
        }
        if (units[0].node_vector.size() > unit0_node_before) {
            bool trailing_zero = true;
            for (size_t i = unit0_node_before; i < units[0].node_vector.size(); i++) {
                if (units[0].node_vector[i] != 0.0) {
                    trailing_zero = false;
                }
            }
            check(trailing_zero, "new node weights appended to units are zeros");
        }
    } else {
        Log::info(
            "Mutation did not introduce new innovation numbers after retries (clone/enable/disable only)\n"
        );
        check(space.get_node_vector_size() == node_size_before && space.get_edge_vector_size() == edge_size_before,
              "catalog unchanged when mutation added no innovations");
    }

    encode_pair(space, genome1, genome2, n1, e1, n2, e2);
    dist = SomVectorSpace::distance(n1, e1, n2, e2);
    Log::info("distance(genome1, genome2) after mutation = %f\n", dist);
    if (dist == 0.0) {
        Log::info("NOTE: genomes still encode identically after mutation attempts\n");
    } else {
        check(dist > 0.0, "mutated genome differs in SOM encoding");
    }

    // Pick two innovations still enabled on BOTH genomes after mutation (mutation may
    // have already disabled some of the original hidden nodes on genome2).
    int32_t one_sided_innov = -1;
    int32_t shared_disable_innov = -1;
    for (int32_t innov : space.get_node_innovations()) {
        if (genome1->is_node_enabled(innov) && genome2->is_node_enabled(innov)) {
            if (one_sided_innov < 0) {
                one_sided_innov = innov;
            } else if (shared_disable_innov < 0) {
                shared_disable_innov = innov;
                break;
            }
        }
    }
    check(one_sided_innov >= 0, "found a node innovation enabled on both genomes (one-sided disable)");
    check(shared_disable_innov >= 0, "found a second node innovation enabled on both genomes (shared disable)");

    // ----- Stage C: disable one node only on genome1 -----
    print_stage_header("Stage C: disable a node only on genome1");
    check(genome1->set_node_enabled(one_sided_innov, false), "disabled node on genome1 only");
    Log::info(
        "Disabled node innovation %d on genome1 only (still enabled on genome2)\n", one_sided_innov
    );

    encode_pair(space, genome1, genome2, n1, e1, n2, e2);
    dist = SomVectorSpace::distance(n1, e1, n2, e2);
    Log::info("distance(genome1, genome2) after one-sided disable = %f\n", dist);
    check(dist > 0.0, "one-sided disable yields nonzero distance (zeros vs real weights)");

    vector<RNN_Genome*> living = {genome1, genome2};
    size_t node_innov_count_before_c = space.get_node_innovations().size();
    bool compressed = space.compress(living, units);
    report_compress(compressed, "compress after one-sided disable");
    // Compression may still remove OTHER innovations that mutation left disabled on both
    // genomes, but the one-sided innovation must remain because genome2 still enables it.
    bool one_sided_still_present = false;
    for (int32_t innov : space.get_node_innovations()) {
        if (innov == one_sided_innov) {
            one_sided_still_present = true;
        }
    }
    check(one_sided_still_present, "compress keeps innovation still enabled on genome2");
    Log::info(
        "catalog size %zu -> %zu after Stage C compress\n", node_innov_count_before_c,
        space.get_node_innovations().size()
    );
    print_catalog(space);

    // ----- Stage D: disable a shared node on BOTH genomes -----
    print_stage_header("Stage D: disable shared node on both genomes, then compress");
    check(genome1->set_node_enabled(shared_disable_innov, false), "disabled shared node on genome1");
    check(genome2->set_node_enabled(shared_disable_innov, false), "disabled shared node on genome2");
    Log::info(
        "Disabled shared node innovation %d on BOTH genomes (candidate for compression)\n", shared_disable_innov
    );

    size_t node_innov_count_before = space.get_node_innovations().size();
    encode_pair(space, genome1, genome2, n1, e1, n2, e2);
    dist = SomVectorSpace::distance(n1, e1, n2, e2);
    Log::info("distance(genome1, genome2) after shared disable = %f\n", dist);

    compressed = space.compress(living, units);
    report_compress(compressed, "compress after shared disable");
    print_catalog(space);
    print_unit("unit[0] after compress", units[0]);
    print_unit("unit[1] after compress", units[1]);

    check(compressed, "compress removes innovation disabled across all genomes");
    check(space.get_node_innovations().size() < node_innov_count_before, "node innovation catalog shrank");
    bool shared_still_present = false;
    for (int32_t innov : space.get_node_innovations()) {
        if (innov == shared_disable_innov) {
            shared_still_present = true;
        }
    }
    check(!shared_still_present, "shared disabled innovation removed from catalog");
    check(units[0].node_vector.size() == space.get_node_vector_size(), "units resized with compressed catalog");
    check(units[1].node_vector.size() == space.get_node_vector_size(), "all units match compressed node size");

    // Re-encode after compress; vectors should be shorter.
    encode_pair(space, genome1, genome2, n1, e1, n2, e2);
    check(n1.size() == space.get_node_vector_size(), "encoded genome1 node vector matches compressed size");
    dist = SomVectorSpace::distance(n1, e1, n2, e2);
    Log::info("distance(genome1, genome2) after compress = %f\n", dist);

    // one-sided disabled node should still be in catalog (enabled on genome2)
    one_sided_still_present = false;
    for (int32_t innov : space.get_node_innovations()) {
        if (innov == one_sided_innov) {
            one_sided_still_present = true;
        }
    }
    check(one_sided_still_present, "innovation disabled on only one genome is retained");

    delete genome1;
    delete genome2;
    delete seed;
    // EXAMM destructor deletes weight_rules and genome_property.
    delete examm;
}

int main(int argc, char** argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);

    // Ensure Log::initialize has the required flags even when run with no CLI args.
    if (!argument_exists(arguments, "--std_message_level")) {
        arguments.push_back("--std_message_level");
        arguments.push_back("INFO");
    }
    if (!argument_exists(arguments, "--file_message_level")) {
        arguments.push_back("--file_message_level");
        arguments.push_back("INFO");
    }
    if (!argument_exists(arguments, "--output_directory")) {
        arguments.push_back("--output_directory");
        arguments.push_back("som_vector_test_output");
    }

    Log::initialize(arguments);
    Log::set_id("test_som_vectors");

    test_som_unit_methods();
    test_euclidean_distance();
    test_genome_scenario();

    Log::info("\n========== SUMMARY: %d passed, %d failed ==========\n", g_pass, g_fail);
    Log::release_id("test_som_vectors");

    return g_fail == 0 ? 0 : 1;
}
