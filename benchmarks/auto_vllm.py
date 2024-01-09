# vllm自动化测试脚本
# cmd1:启动vllm服务
# cmd2:批量发送请求
# 用法：
# pip install -r requirements.txt
# pip install aiohttp
# pip install vllm
# 修改main中的models列表，放入模型
# 获取数据：wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
# 修改tensor_parallel_sizes 列表，比如想测4卡和8卡，就改成tensor_parallel_sizes = [4,8]
# python auto_vllm.py
# 等待，看logs文件夹里的log
import subprocess
import time
import torch

def get_gpu_model_and_count():
    """
    获取显卡模型名称和数量。

    :return: (显卡模型名称, 显卡数量)
    """
    # 查询显卡模型名称
    print("checking GPU...")
    result_model = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], stdout=subprocess.PIPE)
    gpu_model = result_model.stdout.decode('utf-8').split('\n')[0].strip().replace(" ", "")
    print("get gpu_model:{gpu}".format(gpu=gpu_model))

    # 查询显卡数量
    result_count = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv'], stdout=subprocess.PIPE)
    gpu_count = len(result_count.stdout.decode('utf-8').strip().split('\n')) - 1
    print("get gpu_count:{gpu_cnt}".format(gpu_cnt=gpu_count))

    return gpu_model, gpu_count


def run_llm_and_test(model_path, tensor_parallel_size, request_rate=10):
    """
    运行LLM并对其进行性能测试。

    :param model_path: LLM模型的路径。
    :param tensor_parallel_size: 并行处理的大小。
    :param request_rate: 请求速率。
    """
    gpu_model, _ = get_gpu_model_and_count()

    # 准备启动LLM的命令
    cmd1 = f"python3 -m vllm.entrypoints.api_server --model {model_path} --tensor-parallel-size {tensor_parallel_size} --trust-remote-code --dtype bfloat16 --gpu-memory-utilization 0.8 --enforce-eager"

    model_name = model_path.split("/")[-1]
    # 定义服务器日志文件的路径
    server_log_filename = f"./logs/server_{model_name}-{gpu_model}x{tensor_parallel_size}-req-{request_rate}.log"

    try:
        # 启动LLM服务器并保存输出到日志文件
        with open(server_log_filename, 'w') as server_log_file:
            print("about to run cmd1...")
            server_process = subprocess.Popen(cmd1, shell=True, stdout=server_log_file, stderr=server_log_file, text=True, bufsize=1, universal_newlines=True)

            # 检查LLM服务器的输出，等待"http://localhost:8000"或其他错误信息
            while True:
                with open(server_log_filename, 'r') as f:
                    content = f.read()
                    if "http://0.0.0.0:8000" in content:
                        break
                    # 如果检测到错误信息，抛出异常
                    if "Error" in content or "Exception" in content:
                        raise Exception(f"Error detected in cmd1 output. Check {server_log_filename} for details.")
                time.sleep(1)  # 等待1秒后再次检查

            # 定义进行性能测试的命令
            filename = f"./data/{model_name}-{gpu_model}x{tensor_parallel_size}-req-{request_rate}"
            cmd2 = f"python3 benchmark_serving.py --backend vllm --dataset ShareGPT_V3_unfiltered_cleaned_split.json --tokenizer {model_path} --request-rate {request_rate} --host 0.0.0.0 --trust-remote-code --port 8000"

            # 执行性能测试命令并保存输出到日志文件
            log_filename = f"./logs/{model_name}-{gpu_model}x{tensor_parallel_size}-req-{request_rate}.log"
            with open(log_filename, 'w') as log_file:
                print("about to run cmd2...")
                subprocess.run(cmd2, shell=True, stdout=log_file, stderr=log_file)

    except Exception as e:
        # 输出错误信息并终止LLM服务器
        print(f"Error encountered with model: {model_path} and tensor_parallel_size: {tensor_parallel_size}. Error details: {e}")
        server_process.terminate()
        server_process.wait()
        return  

    print("Executing pkill command...")
    server_process.terminate()
    server_process.wait()
    subprocess.run(['pkill', '-f', 'ray'])

if __name__ == "__main__":
    # 定义需要测试的LLM模型路径列表
    models = [
        "/llama_ft/models/Llama-2-70b-chat-hf/Llama2-70b-chat-hf"
    ]

    _, gpu_count = get_gpu_model_and_count()
    # 根据显卡数量定义tensor_parallel_sizes
    #tensor_parallel_sizes = list(range(4, gpu_count + 1))
    tensor_parallel_sizes = [8]

    # 对每个LLM模型进行测试
    for model in models:
        for tensor_parallel_size in tensor_parallel_sizes:
            run_llm_and_test(model, tensor_parallel_size, request_rate=1)
            print("wait for 60s...")
            torch.cuda.empty_cache()
            time.sleep(60)
            # run_llm_and_test(model, tensor_parallel_size, request_rate=2)
            # print("wait for 60s...")
            # torch.cuda.empty_cache()
            # time.sleep(60)
            run_llm_and_test(model, tensor_parallel_size, request_rate=4)
            print("wait for 60s...")
            torch.cuda.empty_cache()
            time.sleep(60)
            # run_llm_and_test(model, tensor_parallel_size, request_rate=8)
            # print("wait for 60s...")
            # torch.cuda.empty_cache()
            # time.sleep(60)
            # run_llm_and_test(model, tensor_parallel_size, request_rate=32)
            # print("wait for 60s...")
            # torch.cuda.empty_cache()
            # time.sleep(60)
            run_llm_and_test(model, tensor_parallel_size, request_rate='inf')
            print("wait for 60s...")
            torch.cuda.empty_cache()
            time.sleep(60)
