cd build
# mpirun -np 4 ./mpi/examm_mpi --training_filenames ../datasets/2018_coal/burner_[0-9].csv --test_filenames ../datasets/2018_coal/burner_1[0-1].csv --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int --output_parameter_names Main_Flm_Int --number_islands 8 --population_size 10 --max_genomes 4000 --speciation_method "island" --num_genomes_check_on_island 1000 --check_on_island_method "EraseWorst" --bp_iterations 5 --output_directory "$HOME/Dropbox/test_output/base1" --possible_node_types simple UGRNN MGU GRU delta LSTM --std_message_level ERROR --file_message_level ERROR


for folder in  0 1 2 3 4 5 6 7 8 9
do
    exp_name="$HOME/Dropbox/test_output/Erastworst/$folder"
    mkdir -p $exp_name
    echo "\tIteration: "$exp_name
    echo "\t###-------------------###"
    mpirun -np 4 ./mpi/examm_mpi --training_filenames ../datasets/2018_coal/burner_[0-9].csv --test_filenames ../datasets/2018_coal/burner_1[0-1].csv --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int --output_parameter_names Main_Flm_Int --number_islands 8 --population_size 10 --max_genomes 4000 --speciation_method "island" --num_genomes_check_on_island 1000 --check_on_island_method "EraseWorst" --bp_iterations 10 --output_directory $exp_name --possible_node_types simple UGRNN MGU GRU delta LSTM --std_message_level ERROR --file_message_level ERROR
done
