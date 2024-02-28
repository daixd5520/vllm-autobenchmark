# vllm LLaMA2-70b offline推理测试

## 1. 设置max_batch_size的方法

`cd /path/to/vllm-autobenchmark/vllm/engine/`

`vim arg_utils.py`

将里面的max_num_seqs从256改成 想设置的max_batch_size（在 line 30）

## 2. 进行offline推理测试

`cd /path/to/vllm-autobenchmark/benchmarks/`

`python benchmark_throughput.py --backend vllm --input-len 2048 --output-len 128 --model /root/Llama-2-70b-chat-hf -tp 8 --num-prompts 3000`

### 参数说明

- input-len 输入长度，代码会根据这个数自动生成特定长度的sequence

- output-len 限制输出的最大长度

- model 模型文件所在路径

- tp 将模型切到几张卡上

- num-prompts 此次测试进行几条sequence，如果小于max_num_seqs的话，那么max_batch_size就会是num-prompts

### 如果想修改gpu显存利用率

vllm会根据总可用显存进行batch的动态调整，所以可以修改gpu_memory_utilization

`cd /path/to/vllm-autobenchmark/vllm/engine/`

`vim arg_utils.py`

修改gpu_memory_utilization（在 line 28）
