#!/bin/sh
# This is an example of running EXAMM multithread version on coal dataset
#
# The coal dataset is normalized
# To run datasets that's not normalized, make sure to add arguments:
#    --normalize min_max for Min Max normalization, or
#    --normalize avg_std_dev for Z-score normalization


#cd build

exp_name="/test_output/coal_mt"
#mkdir -p $exp_name
exp_root=$1
echo "Running base EXAMM code with coal dataset, results will be saved to: "${exp_root}
echo "###-------------------###"

n_threads=2
n_genomes=20000

n_samples=0
sample_length=0

run_examm_mt() {
    dir="${exp_root}/resn_${n_samples}s_${sample_length}l"
    dir_out="${dir}/std.out"
    mkdir $dir

    echo "Running EXAMM_MT-RESN with ${n_threads} threads, ${n_genomes} genomes and ${n_samples} samples of length ${sample_length}"
    ./multithreaded/examm_mt --number_threads $n_threads \
    --training_filenames ../datasets/2018_coal/burner_[0-9].csv --test_filenames \
    ../datasets/2018_coal/burner_1[0-1].csv \
    --time_offset 1 \
    --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int \
    --output_parameter_names Main_Flm_Int \
    --number_islands 10 \
    --population_size 10 \
    --max_genomes $n_genomes \
    --bp_iterations 5 \
    --output_directory $dir \
    --possible_node_types simple UGRNN MGU GRU delta LSTM \
    --std_message_level INFO \
    --file_message_level NONE \
    --use_resn \
    --resn_n_samples ${n_samples} \
    --resn_sample_length ${sample_length} \
    --output_genome_name resn_genome_n_${n_samples}_l_${sample_length}
        > $dir_out
}


n_samples=100
sample_length=800

run_examm_mt

#n_samples=200
#sample_length=400

#run_examm_mt

#n_samples=400
#sample_length=200

#run_examm_mt

#n_samples=800
#sample_length=100

#run_examm_mt
