# ===-------------------------------------------------------------------------===
#                                   性能分析
# ===-------------------------------------------------------------------------===
# 
# nsight systems
# 
nsys nvprof -o [file_name] -f ./executable
# 
# dlprof
# 
dlprof --mode=tensorrt --force=true --reports=all --output_path=${model_dir}/dlprof_result [python .py]
dlprofviewer -b 0.0.0.0 -p 8110 [*/dlprof_dldb.sqlite]
# 
# nvprof
# 
# 查看线程束的活跃比例（每个周期活跃的线程束数量的平均值与一个 SM 支持的最大线程束数量的比）
nvprof --metrics achieved_occupancy ./executable
# 查看全局内存吞吐量（global memory load throughput）
nvprof --metrics gld_throughput ./executable
# 查看全局内存加载效率（Ratio of requested global memory load throughput to required global memory load throughput expressed as percentage）
nvprof --metrics gld_efficiency ./executable
# 查看每个线程束执行指令的平均数量
nvprof --metrics inst_per_warp ./executable
# 查看分支效率
nvprof --metrics branch_efficiency ./executable
# 查看分支/发散分支数量
nvprof --events branch,divergent_branch ./executable
# 查看全局/共享内存访问事务数量（load/store）
nvprof  --metrics shared_load_transactions,shared_store_transactions ./executable
nvprof  --metrics shared_load_transactions_per_request,shared_store_transactions_per_request ./executable
nvprof  --metrics gld_transactions ./executable
nvprof  --metrics gld_transactions_per_request ./executable
nvprof  --metrics gst_transactions ./executable
nvprof  --metrics gst_transactions_per_request ./executable

# 
# nvcc
# 
# 打印核函数使用的寄存器数量
nvcc --ptxas-options=-v
# 关闭/启用 L1 Cache
nvcc -Xptxas -dlcm=cg/ca
