#!/bin/sh
for folder in 4 5
do
  mkdir ./no_of_ants_experiment/rec_3/0$folder
  #for a in 5 10 20 40 80 160 320 640
  for a in 320 640
  do
    mpirun -np 13 acnnto_mpi --training_filenames ~/Dropbox/phd_work/microBeam/1537\ MTI-RIT\ Data/10_day_data/train_DELETE_ME.csv --test_filenames  ~/Dropbox/phd_work/microBeam/1537\ MTI-RIT\ Data/10_day_data/test_DELETE_ME.csv --number_threads 1 --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Oil_Flow Main_Flm_Int  --output_parameter_names Main_Flm_Int --population_size 15 --max_genomes 200 --bp_iterations 500 --normalize --output_directory ./no_of_ants_experiment/rec_3/0$folder --ants $a --hidden_layers_depth 3 --hidden_layer_nodes 12 --pheromone_decay_parameter 0.5 --pheromone_update_strength 0.5 --max_recurrent_depth 3
  done
done

#for d in 0.05 0.1 0.3 0.4 0.6 0.8 1.0
#do
#    mpirun -np 5 acnnto_mpi --training_filenames ~/Dropbox/phd_work/microBeam/1537\ MTI-RIT\ Data/10_day_data/train_DELETE_ME.csv --test_filenames  ~/Dropbox/phd_work/microBeam/1537\ MTI-RIT\ Data/10_day_data/test_DELETE_ME.csv --number_threads 1 --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Oil_Flow Main_Flm_Int  --output_parameter_names Main_Flm_Int --population_size 15 --max_genomes 200 --bp_iterations 500 --normalize --output_directory ./ --ants 150 --hidden_layers_depth 3 --hidden_layer_nodes 12 --pheromone_decay_parameter $d --pheromone_update_strength 0.5 --max_recurrent_depth 3
#done

#for u in 0.05 0.1 0.3 0.4 0.6 0.8 1.0
#do
#   mpirun -np 5 acnnto_mpi --training_filenames ~/Dropbox/phd_work/microBeam/1537\ MTI-RIT\ Data/10_day_data/train_DELETE_ME.csv --test_filenames  ~/Dropbox/phd_work/microBeam/1537\ MTI-RIT\ Data/10_day_data/test_DELETE_ME.csv --number_threads 1 --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Oil_Flow Main_Flm_Int  --output_parameter_names Main_Flm_Int --population_size 15 --max_genomes 200 --bp_iterations 500 --normalize --output_directory ./ --ants 150 --hidden_layers_depth 3 --hidden_layer_nodes 12 --pheromone_decay_parameter 0.5 --pheromone_update_strength $u --max_recurrent_depth 3
#done
