# Branch: master
#     required:
#         --time_offset: number of minutes to predict in the future
#         --number_islands: number of islands for island speciation strategy
#         --population_size: island capacity
#         --max_genomes: total number of genomes to generate
#         --bp_iterations: back propagation iterations
#         --output_directory: output directory for results
#         --std_message_level: can be INFO, DEBUG, WARNING, TRACE, ERROR, FATAL
#         --file_message_level: can be INFO, DEBUG, WARNING, TRACE, ERROR, FATAL
        
#     examm:
#         --learning_rate: learning rate for back propagation
#         --high_threshold: high threshold for back propagation
#         --low_threshold: low threshold for back propagation
#         --dropout_probability: dropout probability
#         --possible_node_types: possible node types, include simple node and memory cells
#         --min_recurrent_depth: minimal recurrent depth allowed for recurrent connections
#         --max_recurrent_depth: maximum recurrent depth allowed for recurrent connections
#         --write_time_series: write time series data to ".csv" file

#     time series:
#         choose one set of the arguments for input files:
    
#             --filenames
#             --training_indexes
#             --test_indexes
#                 or
#             --training_filenames
#             --test_filenames

#         choose one set of the arguments for parameter names:

#             --parameters
#                 or
#             --input_parameter_names
#             --output_parameter_names
#             --shift_parameter_names: parameters 'time_offset' in the future are used for prediction

#         --normalize: data normalize method, can be "min_max" or "avg_std_dev"

#     island speciation strategy:
#         --speciation_method: can be "island" or "neat", "island" if not specified
#         --extinction_event_generation_number: the number of genomes generated when the extinction event happens, 0 if not specified
#         --islands_to_exterminate: number of islands to exterminate in each extinction event, 0 if not specified
#         --island_ranking_method: island ranking method, so far only 'EraseWorst' is avaliable
#         --repopulation_method: ways to repopulated the extincted island, can be "bestGenome", "bestIsland", "bestParents", and "randomParents"
#         --repopulation_mutations: number of mutations applied to the genomes for repopulation 
#         --repeat_extinction: if allow repeat extinction on the same island, 'false' if not specified

#     transfer learning:
#         --transfer_learning_version: can be "v1" "v2" "v1+v2"
#         --genome_bin: the file path to the genome ".bin" file
#         --start_filled: the islands would start with 'Filled' status if enabled 
#         --epigenetic_weights: false if not specified

# Sample script for examm:
    out_dir="./test_output/"
    mkdir -p $out_dir
    mpirun -np 8 ./build/mpi/examm_mpi --training_filenames ./datasets/2018_coal/burner_[0-9].csv \
        --test_filenames ./datasets/2018_coal/burner_1[0-1].csv \
        --time_offset 1 \
        --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int \
        --output_parameter_names Main_Flm_Int \
        --number_islands 10 \
        --population_size 10 \
        --max_genomes 4000 \
        --speciation_method "island" \
        --bp_iterations 10 \
        --output_directory $out_dir \
        --possible_node_types simple UGRNN MGU GRU delta LSTM \
        --std_message_level INFO \
        --file_message_level INFO 

# Sample script for speciation repopulation:
    # out_dir="./test_output/"
    # mkdir -p $out_dir
    # mpirun -np 8 ./build/mpi/examm_mpi --training_filenames ./datasets/2018_coal/burner_[0-9].csv \
    #     --test_filenames ./datasets/2018_coal/burner_1[0-1].csv \
    #     --time_offset 1 \
    #     --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int \
    #     --output_parameter_names Main_Flm_Int \
    #     --number_islands 10 \
    #     --population_size 10 \
    #     --max_genomes 4000 \
    #     --speciation_method "island" \
    #     --extinction_event_generation_number 1500 \
    #     --island_ranking_method "EraseWorst" \
    #     --repopulation_method "bestGenome" \
    #     --islands_to_exterminate 1 \
    #     --repopulation_mutations 2 \
    #     --repeat_extinction \
    #     --bp_iterations 10 \
    #     --output_directory $out_dir \
    #     --possible_node_types simple UGRNN MGU GRU delta LSTM \
    #     --std_message_level INFO \
    #     --file_message_level INFO 

#Sample scripe for transfer learning:
    # out_dir="./test_output/"
    # genome="/path/to/genome/genome.bin"
    # mkdir -p $out_dir
    # mpirun -np 8 ./build/mpi/examm_mpi --training_filenames ./datasets/2019_ngafid_transfer/pa44_file_[1-9].csv \
    #     --test_filenames ./datasets/2019_ngafid_transfer/pa44_file_1[0-2].csv \
    #     --time_offset 1 \
    #     --input_parameter_names "AltAGL" "AltB" "AltGPS" "AltMSL" "BaroA" "E1_CHT1" "E1_EGT1" "E1_EGT2" "E1_EGT3" "E1_EGT4" "E1_FFlow" "E1_MAP" "E1_OilP" "E1_OilT" "E1_RPM" "E2_CHT1" "E2_EGT1" "E2_EGT2" "E2_EGT3" "E2_EGT4" "E2_FFlow" "E2_OilP" "E2_OilT" "E2_RPM" "FQtyL" "FQtyR" "GndSpd" "IAS" "LatAc" "NormAc" "OAT" "Pitch" "Roll" "TAS" "VSpd" "VSpdG" "WndDr" "WndSpd" \
    #     --output_parameter_names "E1_CHT1" "E1_EGT1" "E1_EGT2" "E1_EGT3" "E1_EGT4" "E1_FFlow" "E1_MAP" "E1_OilP" "E1_OilT" "E1_RPM" "E2_CHT1" "E2_EGT1" "E2_EGT2" "E2_EGT3" "E2_EGT4" "E2_FFlow" "E2_OilP" "E2_OilT" "E2_RPM" \
    #     --number_islands 10 \
    #     --population_size 10 \
    #     --max_genomes 4000 \
    #     --bp_iterations 10 \
    #     --possible_node_types simple UGRNN MGU GRU delta LSTM \
    #     --normalize min_max \
    #     --output_directory $out_dir \
    #     --genome_bin $genome \
    #     --transfer_learning_version v1 \
    #     --epigenetic_weights \
    #     --std_message_level INFO \
    #     --file_message_level INFO 

# Sample script for weight initialization    
    # out_dir="./test_output/"
    # mkdir -p $out_dir
    # mpirun -np 8 ./build/mpi/examm_mpi --training_filenames ./datasets/2018_coal/burner_[0-9].csv \
    #     --test_filenames ./datasets/2018_coal/burner_1[0-1].csv \
    #     --time_offset 1 \
    #     --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int \
    #     --output_parameter_names Main_Flm_Int \
    #     --number_islands 5 \
    #     --population_size 5 \
    #     --max_genomes 2000 \
    #     --speciation_method "island" \
    #     --bp_iterations 10 \
    #     --weight_initialize xavier \
    #     --weight_inheritance xavier \
    #     --new_component_weight lamarckian \
    #     --output_directory $out_dir \
    #     --possible_node_types simple UGRNN MGU GRU delta LSTM \
    #     --std_message_level INFO \
    #     --file_message_level INFO 

