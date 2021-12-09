#!/bin/bash 
echo '000'
echo $1
filenames=$1
echo $filenames
echo '-----------'
for file in $(ls $1*/logging.log);do
	echo $file
	cat $file|grep best -C 6 | tail -n 12
done

# for((i=1;i<=10;i++));do
# 	echo $(expr $i \* 4);
# done 
