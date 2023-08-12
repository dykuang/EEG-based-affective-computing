for model in HST # EEGNet,MSBAM,HST
do
    for LabelMode in OneHot # OneHot_N, OneHot, Regression or Mixture
    do
        for i in {1..23}
        do
            for PT in T  # T or F
            do
            echo "Benchmark on subject: $i, start ..."
            CUDA_VISIBLE_DEVICES=0 python DREAMER_independent_1on1_formal_notransit.py --Train_Subject $i --Test_Subject $(expr $((i+1)) % 23 ) --ModelChoice $model --LabelMode $LabelMode --LabelChoice A --Pretrain $PT    
            #echo "Training on $token"
            echo "Benchmark on subject: $i, finished"
            done
        done
    done
done
