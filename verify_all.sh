files=(basic_arithmetic linear_algebra main neural_networks operator_norms parsing robustness_certification)
for i in ${files[@]}
do
	echo Verifying $i.dfy...
	dafny verify $i.dfy
done
