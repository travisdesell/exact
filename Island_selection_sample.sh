cd build

# for folder in  0 1 2 3 4 5 6 7 8 9
# do
    # exp_name="../test_output/debug"
    # mkdir -p $exp_name
    # echo "\tIteration: "$exp_name
    # echo "\t###-------------------###"
    # mpirun -np 20 ./mpi/examm_mpi --training_filenames ../datasets/2018_coal/burner_[0-9].csv --test_filenames ../datasets/2018_coal/burner_1[0-1].csv --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int --output_parameter_names Main_Flm_Int --number_islands 10 --population_size 10 --max_genomes 4000 --speciation_method "island" --extinction_event_generation_number  0 --island_ranking_method "EraseWorst" --repopulation_method "bestGenome" --islands_to_exterminate 0 --repopulation_mutations 0 --bp_iterations 3 --output_directory $exp_name --possible_node_types simple UGRNN MGU GRU delta LSTM --std_message_level ERROR --file_message_level ERROR
# done

# for folder in  0 1 2 3 4 5 6 7 8 9
# do
    exp_name="../test_output/debug"
    mkdir -p $exp_name
    echo "\tIteration: "$exp_name
    echo "\t###-------------------###"
    mpirun -np 8 ./mpi/examm_mpi --training_filenames ../datasets/2018_coal/burner_[0-9].csv --test_filenames ../datasets/2018_coal/burner_1[0-1].csv --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int --output_parameter_names Main_Flm_Int --number_islands 6 --population_size 5 --max_genomes 500 --speciation_method "neat"  --species_threshold 3 --fitness_threshold 1 --neat_c1 1 --neat_c2 1 --neat_c3 0.4 --bp_iterations 2 --output_directory $exp_name --possible_node_types simple UGRNN MGU GRU delta LSTM --std_message_level ERROR --file_message_level NONE
# done

# This part is for coal data debugging
    # exp_name="../test_output/debug"
    # mkdir -p $exp_name
    # echo "\tIteration: "$exp_name
    # echo "\t###-------------------###"
    # mpirun -np 20 ./mpi/examm_mpi --training_filenames ../datasets/2018_coal/burner_[0-9].csv --test_filenames ../datasets/2018_coal/burner_1[0-1].csv --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int --output_parameter_names Main_Flm_Int --number_islands 6 --population_size 5 --max_genomes 500 --speciation_method "neat"  --species_threshold 3 --fitness_threshold 1 --neat_c1 1 --neat_c2 1 --neat_c3 0.4 --bp_iterations 2 --output_directory $exp_name --possible_node_types simple UGRNN MGU GRU delta LSTM --std_message_level ERROR --file_message_level NONE

# ## This part is for flight data debugging
    
#     flight="pa28"
    
#     if [ $flight = "c172" ]; then
#         echo "\tUsing c172 dateset"
#         INPUT_PARAMETERS="AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd"
#     elif [ $flight = "pa28" ]; then
#         echo "\tUsing pa28 dataset"
#         INPUT_PARAMETERS="AltAGL AltB AltGPS AltMSL E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd"
#     elif [ $flight="pa44" ]; then
#         echo "\tUsing pa44 dataset"
#         INPUT_PARAMETERS="AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_MAP E1_OilP E1_OilT E1_RPM E2_CHT1 E2_EGT1 E2_EGT2 E2_EGT3 E2_EGT4 E2_FFlow E2_OilP E2_OilT E2_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd"
#     else 
#         echo "\tERROR: wrong flight type!"
#         exit 1
#     fi 

#     # OUTPUT_PARAMETERS="E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4"
#     OUTPUT_PARAMETERS="Pitch Roll AltMSL IAS LatAc NormAc "
#     exp_name="../test_output/debug"
#     mkdir -p $exp_name
#     echo "\tIteration: "$exp_name
#     echo "\t###-------------------###"
#     mpirun -np 22 ./mpi/examm_mpi --training_filenames ../datasets/2019_ngafid_transfer/${flight}_file_[1-10].csv --test_filenames ../datasets/2019_ngafid_transfer/${flight}_file_[11-12].csv --time_offset 1 --input_parameter_names $INPUT_PARAMETERS --output_parameter_names $OUTPUT_PARAMETERS --normalize --number_islands 6 --population_size 5 --max_genomes 500 --speciation_method "island" --extinction_event_generation_number 400 --island_ranking_method "EraseWorst" --repopulation_method "bestGenome" --islands_to_exterminate 6 --repopulation_mutations 2 --bp_iterations 2 --output_directory $exp_name --possible_node_types simple UGRNN MGU GRU delta LSTM --std_message_level ERROR --file_message_level ERROR

# EXAMM="/home/zl7069/git/transfer/exact"


# # # REPOPULATION_METHOD = "bestGenome"
# exp_name="../tests_output/test/"
# mkdir -p $exp_name
# # file="./test_output/fitness_log.csv"
# # files="../test_output/debug"
# IFS="genome_"
# for ff in ../test_output/debug/*.bin
# do
#     # echo $ff
#     # read -a ADDR <<< "$ff"
#     # read -ra ADDR <<< "$ff"
#     # read -ra ADDR
#     addr=$(echo $ff | tr "genome_" "\n")
#     echo ${ADDR[1]}
# done
# # target="$exp_name/fitness_log.csv"
# # cp $file $target
