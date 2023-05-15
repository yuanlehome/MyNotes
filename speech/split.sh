#!/bin/bash

# input.wav 为 wav 格式的输入音频文件
# -segment_time 20 将文件分割为 20 秒一个
# -ac 2 双声道
# -ar 16000 转换为 16khz 的采样率

ffmpeg -i input.wav -f segment -segment_time 20 -c copy -ac 2 -ar 16000 input/input_%02d.wav
