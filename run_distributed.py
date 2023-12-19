import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

#wandb.init(project="gpt-neo-distributed")

def print_perf_stats(latency_set, config, dtype_bytes, batch_size, warmup=3):
    latency_set = list(latency_set[warmup:])
    count = len(latency_set)

    if count > 0:
        avg = sum(latency_set) / count
        num_layers = getattr(config, "num_hidden_layers", None)
        num_parameters = num_layers * config.hidden_size ** 2 * 12  # 12 is an approximation

        avg_latency_ms = avg * 1000
        bandwidth_gb_s = 1/avg * num_parameters * dtype_bytes / 1e9
        flops_tflops_s = 1/avg * num_parameters * dtype_bytes * batch_size / 1e12
        print(f"batch size:{batch_size}")
        print(f"dtype:{dtype_bytes}")
        print(f"Avg Per Token Latency: {avg_latency_ms:.2f} ms")
        print(f"Avg BW: {bandwidth_gb_s:.2f} GB/s")
        print(f"Avg flops: {flops_tflops_s:.2f} TFlops/s")

def run_inference(rank, world_size, model_name, prompts, num_trials=200, max_new_tokens=1024):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model = model.half()
    model.eval()

    latency_set = []
    prompt = prompts[rank]
    start_tokenise_time = time.time()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    end_tokenise_time = time.time()
    tokenisation_time = end_tokenise_time - start_tokenise_time
    print(f"Tokenisation time:{tokenisation_time}")

    batch_size = input_ids.shape[0]
    print(f"batch size:{batch_size}")

    # Determine data type bytes
    dtype_bytes = torch.tensor(0).half().element_size()

    for _ in range(num_trials):
        start_time = time.time()
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
        end_time = time.time()

        latency = (end_time - start_time) / max_new_tokens
        latency_set.append(latency)

    if rank == 0:  # Print performance stats on rank 0
        print_perf_stats(latency_set, model.config, dtype_bytes, batch_size)

def main():
    model_name = "EleutherAI/gpt-neo-2.7B"
    prompts = ["First prompt that happens ", "Second prompt that happens ", "Third prompt is this", "Fourth prompt ends here"]
    world_size = 2

    mp.spawn(
        run_inference,
        args=(world_size, model_name, prompts),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
