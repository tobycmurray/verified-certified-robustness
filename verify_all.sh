files=(main string_utils basic_arithmetic lipschitz)
for i in ${files[@]}
do
	echo Verifying $i.dfy...
	dafny verify $i.dfy
done
