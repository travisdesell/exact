#!/bin/zsh
export INPUT_PARAMETERS='AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd'
export OUTPUT_PARAMETERS='E1_EGT1'

export offset=1
export k=1

push_job() {
  export maxt=$maxt
  export crystal=$crystal
  export bp=$bp
  export control=$control
  sbatch -J $control.maxt$maxt.cr$crystal.bp$bp ./experiment.zsh

}

export control="exp"
for maxt in 1.66 1.33 1.0; do
  for crystal in 64 128 256; do
    for bp in 4 8 16; do
      push_job
    done
  done
done

export control="control"
for bp in 4 8 16; do
  push_job
done
