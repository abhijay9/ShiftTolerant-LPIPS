##### Sample command to run scripts
# For the vanilla version run the following:
# $ nohup bash n_pixel_shift_study/test_scripts/test.sh alex vanilla 0 64 50 > logs/eval_alex_vanilla.out &
# Then to check the results use: 
# $ tail -f logs/eval_alex_vanilla.out
# $ cat logs/eval_alex_vanilla.out | grep twoAFC
# $ cat logs/eval_alex_vanilla.out | grep r_rf
#
# For the shift_tolerant version run the following:
# $ nohup bash n_pixel_shift_study/test_scripts/test.sh alex shift_tolerant 0 64 50 > logs/eval_alex_shift_tolerant.out &

net=$1
variant=$2
gpuId=$3
load_size=$4
batch_size=$5

# 2AFC score
python test_shiftedDataset_model.py --use_gpu --gpu_ids "$gpuId" \
  --load_size "$load_size" --test_type twoAFC --batch_size "$batch_size" \
  --net "$net" --variant "$variant"

python n_pixel_shift_study/twoafc_calc.py --results_folder results/"$net"_"$variant"

# No Shift
python test_shiftedDataset_model.py --use_gpu --gpu_ids "$gpuId" \
  --load_size "$load_size" --test_type xshifted --num_pixels_shifted 0 \
  --net "$net" --variant "$variant" --batch_size "$batch_size"

# 1 Pix Shift
python test_shiftedDataset_model.py --use_gpu --gpu_ids "$gpuId" \
  --load_size "$load_size" --test_type xshifted --num_pixels_shifted 1 \
  --net "$net" --variant "$variant" --batch_size "$batch_size"

# 1 Pix Shift r_fr
python n_pixel_shift_study/rank_flips.py --shifted_0_results results/"$net"_$variant"/xshifted_n-0*" --shifted_n_results results/"$net"_$variant"/xshifted_n-1*" --type pair --model_name "$net"_"$variant"
cat n_pixel_shift_study/evaluations/rankFlips/"$net"_"$variant"_n-1_rrf.csv
python n_pixel_shift_study/rank_flip_rate.py --rrf_results n_pixel_shift_study/evaluations/rankFlips/"$net"_"$variant"_n-1_rrf.csv

# 2 Pix Shift
python test_shiftedDataset_model.py --use_gpu --gpu_ids "$gpuId" \
  --load_size "$load_size" --test_type xshifted --num_pixels_shifted 2 \
  --net "$net" --variant "$variant" --batch_size "$batch_size"

# 2 Pix Shift r_fr
python n_pixel_shift_study/rank_flips.py --shifted_0_results results/"$net"_$variant"/xshifted_n-0*" --shifted_n_results results/"$net"_$variant"/xshifted_n-2*" --type pair --model_name "$net"_"$variant"
cat n_pixel_shift_study/evaluations/rankFlips/"$net"_"$variant"_n-2_rrf.csv
python n_pixel_shift_study/rank_flip_rate.py --rrf_results n_pixel_shift_study/evaluations/rankFlips/"$net"_"$variant"_n-2_rrf.csv

# 3 Pix Shift
python test_shiftedDataset_model.py --use_gpu --gpu_ids "$gpuId" \
  --load_size "$load_size" --test_type xshifted --num_pixels_shifted 3 \
  --net "$net" --variant "$variant" --batch_size "$batch_size"

# 3 Pix Shift r_fr
python n_pixel_shift_study/rank_flips.py --shifted_0_results results/"$net"_$variant"/xshifted_n-0*" --shifted_n_results results/"$net"_$variant"/xshifted_n-3*" --type pair --model_name "$net"_"$variant"
cat n_pixel_shift_study/evaluations/rankFlips/"$net"_"$variant"_n-3_rrf.csv
python n_pixel_shift_study/rank_flip_rate.py --rrf_results n_pixel_shift_study/evaluations/rankFlips/"$net"_"$variant"_n-3_rrf.csv