import os
import time
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from transformers import AutoTokenizer, AutoModelForCausalLM

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def print_perf_stats(latency_set, config, dtype_bytes, batch_size, warmup=3):

    count = len(latency_set)

    if count > 0:
        avg = sum(latency_set) / count
        num_layers = getattr(config, "num_hidden_layers", None)
        num_parameters = num_layers * config.hidden_size ** 2 * 12  # 12 is an approximation

        avg_latency_ms = avg * 1000
        bandwidth_gb_s = 1/avg * num_parameters * dtype_bytes / 1e9
        flops_tflops_s = 1/avg * num_parameters * dtype_bytes * batch_size / 1e12
        print(f"batch size: {batch_size}")
        print(f"dtype bytes: {dtype_bytes}")
        print(f"Avg Per Token Latency: {avg_latency_ms:.2f} ms")
        print(f"Avg BW: {bandwidth_gb_s:.2f} GB/s")
        print(f"Avg flops: {flops_tflops_s:.2f} TFlops/s")

def batch_inference(rank, world_size, model_name, prompts, batch_size, max_length=50, warmup=3):
    setup(rank, world_size)
    

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device(f"cuda:{rank}")
    model = model.half()
    model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    start = rank * len(prompts) // world_size
    end = (rank + 1) * len(prompts) // world_size
    local_prompts = prompts[start:end]

    latency_set = []

    for _ in range(warmup):
        _ = model.generate(tokenizer.encode(local_prompts[0], return_tensors="pt").to(device), max_length=max_length)

    for i in range(0, len(local_prompts)-batch_size):
        print("creating batches")
        batch_prompts = local_prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length)
        input_ids = inputs['input_ids'].to(device)

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(input_ids, max_length=max_length + input_ids.shape[1], pad_token_id=tokenizer.eos_token_id)
            print("outputs")
        end_time = time.time()
        latency = (end_time - start_time) / len(batch_prompts)
        latency_set.append(latency)
        #print("printing outputs")
        #for output in outputs:
            #print(tokenizer.decode(output, skip_special_tokens=True))

    dtype_bytes = torch.tensor(0).half().element_size()
    print(f'{latency_set=}')
    print("about to print latency")
    print_perf_stats(latency_set, model.config, dtype_bytes, batch_size, warmup)
    #print("skipped latency")

    cleanup()

def main():
    model_name = "EleutherAI/gpt-neo-2.7B"
    prompts = [
    "Describe the landscape of Mars.",
    "Explain the theory of relativity.",
    "What is the recipe for a classic French omelet?",
    "Discuss the impact of social media on youth.",
    "Write a short story about a lost treasure.",
    "How does photosynthesis work?",
    "Describe the culture of Japan.",
    "Explain the importance of biodiversity.",
    "What are the health benefits of meditation?",
    "How do computers use binary code?",
    "Write a poem about the ocean.",
    "What is the history of the Olympic Games?",
    "Describe the process of making chocolate.",
    "What are the major causes of climate change?",
    "Explain the working of the human immune system.",
    "Describe a day in the life of an astronaut.",
    "How do bees make honey?",
    "What is quantum computing?",
    "Write a story set in a dystopian future.",
    "Explain the significance of the Renaissance period.",
    "How is glass made?",
    "What is artificial intelligence?",
    "Describe the Northern Lights.",
    "Explain the process of evolution.",
    "What are black holes?",
    "Write about a day in ancient Rome.",
    "How do airplanes fly?",
    "What is virtual reality?",
    "Describe the traditions of Chinese New Year.",
    "Explain the causes of economic recessions.",
    "What is blockchain technology?",
    "Write a detective mystery story.",
    "How does the stock market work?",
    "What are the benefits of renewable energy?",
    "Describe the architecture of the Roman Colosseum.",
    "Explain the principles of non-violent resistance.",
    "What is the importance of the Amazon rainforest?",
    "Write a fantasy story about dragons.",
    "How does the human brain process language?",
    "What is the significance of the Great Wall of China?",
    "Describe the life cycle of a butterfly.",
    "Explain the concept of time zones.",
    "What is the Internet of Things (IoT)?",
    "Write about a journey through space.",
    "How do volcanoes erupt?",
    "What is machine learning?",
    "Describe a traditional Indian wedding.",
    "Explain the significance of the Egyptian pyramids.",
    "What is the role of the United Nations?",
    "Write a suspense thriller plot.",
    "How is coffee produced?",
    "What is cryptocurrency?",
    "Describe the festival of Diwali.",
    "Explain the workings of a nuclear reactor.",
    "What are the wonders of the ancient world?",
    "Write a historical account of the Vikings.",
    "How do solar panels generate electricity?",
    "What is the human genome project?",
    "Describe the wildlife of the African savanna.",
    "Explain the basics of jazz music.",
    "What are the principles of sustainable living?",
    "Write a comedy script about a mistaken identity.",
    "How do satellites orbit the Earth?",
    "What is the significance of the Mona Lisa painting?",
    "Describe the process of brewing beer.",
    "Explain the significance of Shakespeare's works.",
    "What is the mechanism behind electric cars?",
    "Discuss the effects of global warming on polar regions.",
    "Write a short story about a time traveler.",
    "How does the human digestive system work?",
    "Describe the traditions of a Brazilian Carnival.",
    "Explain the basics of quantum physics.",
    "What are the nutritional benefits of a vegan diet?",
    "How do smartphones communicate with each other?",
    "Write a poem about a sunset in the mountains.",
    "What is the story behind the construction of the Eiffel Tower?",
    "Describe how to make homemade pasta.",
    "What are the causes and effects of deforestation?",
    "Explain how vaccinations protect against diseases.",
    "Describe the experience of scuba diving in the Great Barrier Reef.",
    "How is honey used in traditional medicine?",
    "What is the future of artificial intelligence in healthcare?",
    "Write a story about an encounter with aliens.",
    "Explain the cultural significance of the Indian festival Diwali.",
    "How is paper recycled?",
    "What challenges do astronauts face in space?",
    "Describe the phenomenon of the Aurora Borealis.",
    "Explain how wind turbines generate electricity.",
    "What are the mysteries of the Bermuda Triangle?",
    "Write a historical fiction set during the French Revolution.",
    "How do birds migrate thousands of miles?",
    "What is the concept behind smart homes?",
    "Describe the art of making sushi.",
    "Explain the economic impact of tourism.",
    "What is the science behind genetic engineering?",
    "Write a suspense story set in a haunted house.",
    "How does the Federal Reserve impact the economy?",
    "What is the role of renewable resources in energy production?",
    "Describe the Gothic architecture of Notre-Dame Cathedral.",
    "Explain the principles behind yoga and its benefits.",
    "What are the secrets of the deep ocean?",
    "Write an adventure story about exploring a jungle.",
    "How is language processed in the brain?",
    "What are the architectural marvels of ancient Egypt?",
    "Describe the life cycle of a frog.",
    "Explain the concept of different time dimensions.",
    "What is the potential of 5G technology?",
    "Write a narrative about a journey to the center of the Earth.",
    "How do earthquakes occur?",
    "What is the importance of data science?",
    "Describe the customs of a traditional Japanese tea ceremony.",
    "Explain the history behind the Great Pyramid of Giza.",
    "What is the function of the United Nations Security Council?",
    "Write a thriller involving an unsolved mystery.",
    "How is silk produced?",
    "What are the uses and benefits of blockchain in finance?",
    "Describe the celebration of Holi in India.",
    "Explain how hydroelectric power is generated.",
    "What are the Seven Wonders of the Modern World?",
    "Write about the life of the Vikings.",
    "How do photovoltaic cells work?",
    "What are the goals of the Human Brain Project?",
    "Describe a safari in the Serengeti National Park.",
    "Explain the basics of blues music.",
    "What is zero-waste living and its importance?",
    "Write a humorous story about a family reunion.",
    "How do GPS systems function?",
    "What is the historical significance of Leonardo da Vinci's works?"
]


    batch_size = 2
    world_size = 1

    processes = []
    for rank in range(world_size):
        p = Process(target=batch_inference, args=(rank, world_size, model_name, prompts, batch_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
