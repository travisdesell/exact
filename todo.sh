cd build
MAX_GENOME=4000
SPECIATION_METHOD="island"
CHECK_NUM=1000
NUM_ISLANDS=10
POPULATION=10
CHECK_ISLAND_METHOD="EraseWorst"
BP_ITERATION=10

for folder in  0 1 2 3 4 5 6 7 8 9
do
    exp_name="../test_output/bestgenome/$folder"
    mkdir -p $exp_name
    echo "\tIteration: "$exp_name
    echo "\t###-------------------###"
    mpirun -np 4 ./mpi/examm_mpi --training_filenames ../datasets/2018_coal/burner_[0-9].csv --test_filenames ../datasets/2018_coal/burner_1[0-1].csv --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int --output_parameter_names Main_Flm_Int --number_islands $NUM_ISLANDS --population_size $POPULATION --max_genomes $MAX_GENOME --speciation_method $SPECIATION_METHOD --num_genomes_check_on_island $CHECK_NUM --check_on_island_method $CHECK_ISLAND_METHOD --repopulation_method "bestGenome" --repopulation_mutations 0 --bp_iterations $BP_ITERATION --output_directory $exp_name --possible_node_types simple UGRNN MGU GRU delta LSTM --std_message_level ERROR --file_message_level ERROR
done

for folder in  0 1 2 3 4 5 6 7 8 9
do
    exp_name="../test_output/bestgenome_2/$folder"
    mkdir -p $exp_name
    echo "\tIteration: "$exp_name
    echo "\t###-------------------###"
    mpirun -np 4 ./mpi/examm_mpi --training_filenames ../datasets/2018_coal/burner_[0-9].csv --test_filenames ../datasets/2018_coal/burner_1[0-1].csv --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int --output_parameter_names Main_Flm_Int --number_islands $NUM_ISLANDS --population_size $POPULATION --max_genomes $MAX_GENOME --speciation_method $SPECIATION_METHOD --num_genomes_check_on_island $CHECK_NUM --check_on_island_method $CHECK_ISLAND_METHOD --repopulation_method "bestGenome" --repopulation_mutations 2 --bp_iterations $BP_ITERATION --output_directory $exp_name --possible_node_types simple UGRNN MGU GRU delta LSTM --std_message_level ERROR --file_message_level ERROR
done
