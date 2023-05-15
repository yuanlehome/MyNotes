# 性能分析
# nsight systems
nsys nvprof -o f32df -f ./build/EmbEwLnNet_test &>log
# dlprof
dlprof --mode=tensorrt --force=true --reports=all --output_path=${model_dir}/dlprof_result [python .py]

dlprofviewer -b 0.0.0.0 -p 8110 [*/dlprof_dldb.sqlite]

# nvprof 命令
# 查看线程束的活跃比例
nvprof --metrics achieved_occupancy ./executable
# 查看全局内存吞吐量（global memory load throughput）
nvprof --metrics gld_throughput ./executable
# 查看全局内存加载效率（Ratio of requested global memory load throughput to required global memory load throughput expressed as percentage）
nvprof --metrics gld_efficiency ./executable
# 查看每个线程束执行指令的平均数量
nvprof --metrics inst_per_warp ./executable
