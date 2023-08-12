for model in MSBAM
do
    for LabelMode in Regression
    do
        for i in {1..23}
        do
        echo "Benchmark on subject: $i, start ..."
        CUDA_VISIBLE_DEVICES=0 python Dreamer_10cv.py --Subject $i --ModelChoice $model --LabelMode $LabelMode --LabelChoice A --ValMode random     
	#echo "Training on $token"
	    echo "Benchmark on subject: $i, finished"
        done
    done
done
