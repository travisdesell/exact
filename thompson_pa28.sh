num_cpus=32

for folder in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19; do
    flight="pa28"
    
    if [ $flight = "c172" ]; then
        echo "\tUsing c172 dateset"
        INPUT_PARAMETERS="AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd"
    elif [ $flight = "pa28" ]; then
        echo "\tUsing pa28 dataset"
        INPUT_PARAMETERS="AltAGL AltB AltGPS AltMSL E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd"
    elif [ $flight="pa44" ]; then
        echo "\tUsing pa44 dataset"
        INPUT_PARAMETERS="AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_MAP E1_OilP E1_OilT E1_RPM E2_CHT1 E2_EGT1 E2_EGT2 E2_EGT3 E2_EGT4 E2_FFlow E2_OilP E2_OilT E2_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd"
    else 
        echo "\tERROR: wrong flight type!"
        exit 1
    fi 

    # OUTPUT_PARAMETERS="E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4"
    OUTPUT_PARAMETERS="Pitch Roll AltMSL IAS LatAc NormAc "
    exp_name="thompson_results/$flight"
    mkdir -p $exp_name
    echo "\tIteration: "$exp_name
    echo "\t###-------------------###"
    mpirun --use-hwthread-cpus -np $num_cpus build/mpi/examm_mpi \
        --training_filenames datasets/2019_ngafid_transfer/${flight}_file_[1-10].csv \
        --test_filenames datasets/2019_ngafid_transfer/${flight}_file_[11-12].csv \
        --time_offset 1 \
        --input_parameter_names $INPUT_PARAMETERS \
        --output_parameter_names $OUTPUT_PARAMETERS \
        --normalize min_max \
        --number_islands 10 \
        --population_size 5 \
        --max_genomes 10000 \
        --speciation_method "island" \
        --extinction_event_generation_number 400 \
        --island_ranking_method "EraseWorst" \
        --repopulation_method "bestGenome" \
        --islands_to_exterminate 0 \
        --repopulation_mutations 2 \
        --bp_iterations 2 \
        --output_directory $exp_name \
        --possible_node_types simple UGRNN MGU GRU delta LSTM \
        --std_message_level ERROR \
        --file_message_level ERROR \
        --use_thompson_sampling 1
done
