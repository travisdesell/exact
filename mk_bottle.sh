for group in "test" "standard"; do
  for nn in 3 6 12 24; do
    for ni in 1 4 16 64; do
      echo $nn
      echo $ni
      
      file=${group}_nn${nn}_ni${ni}.sh
      
      cp bottle.sh $file
      cat $file | sed -e "s/<<NN>>/${nn}/g" | tee $file
      cat $file | sed -e "s/<<NI>>/${ni}/g" | tee $file
    done
  done
done
