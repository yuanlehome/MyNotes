cudaroot=/usr/local/cuda

${cudaroot}/bin/nvcc -gencode arch=compute_80,code=sm_80 -I${cudaroot}/include -L${cudaroot}/lib64 \
                 -I`pwd`/cutlass/include -I`pwd`/ -I${cudaroot}/ \
fast_gemv.cu -o fast_gemv

CUDA_VISIBLE_DEVICES=0 
 
${cudaroot}/bin/ncu --target-processes all --set full \
 -o CudaCoreWeightonly_1_5120_15360 -f \
 ./fast_gemv 1 5120 15360

${cudaroot}/bin/ncu --target-processes all --set full \
 -o CudaCoreWeightonly_1_5120_5120 -f \
 ./fast_gemv 1 5120 5120

${cudaroot}/bin/ncu --target-processes all --set full \
 -o CudaCoreWeightonly_1_5120_27648 -f \
 ./fast_gemv 1 5120 27648

${cudaroot}/bin/ncu --target-processes all --set full \
 -o CudaCoreWeightonly_1_13824_5120 -f \
 ./fast_gemv 1 13824 5120
