#!/bin/bash

# If you need any help, please email rc-help@rit.edu

#SBATCH -J fullExp
#SBATCH --nodes=16
#SBATCH –A acnntopo -p tier3
#SBATCH -o fullExp.output
#SBATCH -e fullExp.error
#SBATCH --mail-user abdelrahman.elsaid@ndus.edu
# notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-type=ALL

#SBATCH -t 240:00:00

##SBATCH --mem=22000
#SBATCH --mem-per-cpu=900

#
# Your job script goes below this line.
#

module load module_future
module load openmpi-1.10-x86_64
module load gcc/6.1.0


nodes=1024

command_run() {
    for A in 20 40 60 80 100 120 140 160 180 200
    do
        for fold in 0 1 2 3 4 5 6 7 8 9
        do
            work_dir="EXPIMENTS_NEW/$phi_folder/"$constPhi_folder"/$reward_type_folder/$normlization_folder/$ants_species_folder/$bias_folder/$jump/$experiment/$fold"
            mkdir -p $work_dir
#            time mpirun -np $nodes ~/NewAntColony/build/mpi/acnnto_mpi --training_filenames train_DELETE_ME.csv --test_filenames test_DELETE_ME.csv --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Oil_Flow Main_Flm_Int  --output_parameter_names Main_Flm_Int --population_size 40 --max_genomes 2000 --bp_iterations 40 --normalize --output_directory ./$work_dir --ants $A --hidden_layers_depth 3 --hidden_layer_nodes 12 --pheromone_decay_parameter $d --pheromone_update_strength $u --max_recurrent_depth 3 --reward_type $reward_type --norm $norm $weight_reg $bias_forward_paths $fitness_weight_update --const_phi $const_phi $ants_type $use_all_jumps $use_forward_social_ants $use_backward_social_ants
        done
    done
}

norm_jump_process() {
    reward_type="regularized"
        norm=""; weight_reg=""; reward_type_folder="Not_""$reward_type"
        oneAnt
        twoAnt
        for norm in "l1" "l2"
        do
            for w_r in 0.10 0.25 0.50 0.75 0.90
            do
                weight_reg="--weight_reg_parameter ""$w_r"
                reward_type_folder="$reward_type""_""$norm""_""$w_r"
                oneAnt
                twoAnt
            done
        done
    reward_type="constant"; norm=""; weight_reg=""; reward_type_folder="constantPherUpdate"
        oneAnt
        twoAnt
}

oneAnt() {
    use_forward_social_ants=""; use_backward_social_ants=""
    ants_type=""; use_all_jumps=""; ants_species_folder="oneAnt"; jump=""; experiment=""

    bias_forward_paths=""; bias_folder="noBias"
        command_run
    bias_forward_paths="--bias_forward_paths"; bias_folder="useBias"
        command_run
}

twoAnt() {
    ants_type="--use_two_ants_types"; ants_species_folder="twoAnt"; bias_folder=""
    bias_forward_paths=""
    Jumps_noJumps
}

Jumps_noJumps() {

    ##    ALL JUMP    ##
    use_all_jumps="--use_all_jumps"; jump="allJump"

    experiment="$jump""_explorer"                           ; use_forward_social_ants=""                            ; use_backward_social_ants=""
    command_run

    experiment="$jump""_explorer_social_forward"            ; use_forward_social_ants="--use_forward_social_ants"   ; use_backward_social_ants=""
    command_run

    experiment="$jump""_explorer_social_backward"           ; use_forward_social_ants=""                            ; use_backward_social_ants="--use_backward_social_ants"
    command_run

    experiment="$jump""_explorer_social_forward_backward"   ; use_forward_social_ants="--use_forward_social_ants"   ; use_backward_social_ants="--use_backward_social_ants"
    command_run



    ##    ONE JUMP    ##
    use_all_jumps=""; jump="oneJump"

    experiment="$jump""_explorer"                           ; use_forward_social_ants=""                            ; use_backward_social_ants=""
    command_run

    experiment="$jump""_explorer_social_forward"            ; use_forward_social_ants="--use_forward_social_ants"   ; use_backward_social_ants=""
    command_run

    experiment="$jump""_explorer_social_backward"           ; use_forward_social_ants=""                            ; use_backward_social_ants="--use_backward_social_ants"
    command_run

    experiment="$jump""_explorer_social_forward_backward"   ; use_forward_social_ants="--use_forward_social_ants"   ; use_backward_social_ants="--use_backward_social_ants"
    command_run
}

d=0.03; u=0.05

phi_folder="noPhi";constPhi_folder=""
fitness_weight_update=""; const_phi=0.0
    norm_jump_process

phi_folder="usePhi"; constPhi_folder=""
fitness_weight_update="--use_fitness_weight_update"; const_phi=0.0;
    norm_jump_process

phi_folder="constPhi"
fitness_weight_update=""
    for const_phi in 0.10 0.25 0.50 0.75 0.90
    do
        constPhi_folder="phi""$const_phi"
        norm_jump_process
    done



:<<'END'
END
