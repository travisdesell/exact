cd $1
make -j32
cd ..

echo "Running base EXAMM code with coal dataset, results will be saved to: "$exp_name
echo "###-------------------###"
root="/home/josh/development/exact"
MAX_GENOMES=25000
results_dir="/home/josh/development/hom_results/cluster_test/"
mkdir -p $results_dir
mpirun -oversubscribe -H wua:64 --use-hwthread-cpus --mca opal_warn_on_missing_libcuda 0 -np 64 --verbose \
$root/$1/mpi/examm_archipelago_mpi --number_threads 64 \
--training_filenames $root/datasets/2018_coal/burner_[0].csv \
--test_filenames $root/datasets/2018_coal/burner_1[0].csv \
--time_offset 1 \
--input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int \
--output_parameter_names Main_Flm_Int \
--number_islands 3 \
--population_size 16 \
--min_intra_crossover_parents 2 \
--max_intra_crossover_parents 4 \
--min_inter_crossover_parents 2 \
--max_inter_crossover_parents 4 \
--min_mutations 1 \
--max_mutations 1 \
--max_genomes $MAX_GENOMES \
--bp_iterations 0 \
--output_directory $results_dir \
--possible_node_types simple UGRNN MGU GRU delta LSTM \
--std_message_level INFO \
--file_message_level INFO \
--archipelago_config scripts/archipelago/cluster_test.ac \
--node-size 32 --n-nodes 1
