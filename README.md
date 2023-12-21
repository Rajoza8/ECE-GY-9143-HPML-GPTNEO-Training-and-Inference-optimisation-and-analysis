# ECE-GY-9143-HPML-GPTNEO-Training-and-Inference-optimisation-and-analysis

All the files outside the Training Folder are files for Inference of GPT-Neo and it's optimisations. Below are the instructions to run the files and generate results for INFERENCE(work by ro2151).

1. Run a working Singularity container with Sufficient overlay image.

2. Have all the inference files downloaded and saved in the folder 'deepspeed'

3. pip install -r requirements.txt (the one residing outside the Training folder)

4. If using conda then
5. conda create -c conda-forge -n deepspeed python=3.10
conda activate deepspeed
pip install -r requirements.txt

(If there's an error , please ensure you have CUDA properly installed as it auto detects Cuda devices.)



6.Now to run the Pytorch Distributed Data Parallel configuration of the model the command is :

python run_ddp.py (change the world_size = 1, 2  and batch_size= (1 to 32) in the main method of run_ddp.py file accordingly and then run the command)

7.To run the Hugging Face Baseline Model using deepspeed inference (with no optimisations) :

deepspeed --num_gpus  inference-test.py --model EleutherAI/gpt-neo-2.7B --dtype float16 --batch_size  --hf_baseline --test_performance

Please change the num_gpus (1,2) and batch_size (upto 64) and run this command . 

The output should be printed on the console as : 

generation time is {} sec

Avg Per Token Latency:     ms

Avg BW:   GB/s

Avg flops:      TFlops/s

More details are in the arguments.py file where the arguments are fetched from.


8.To test the model Inference with Cuda Graph optimisation , go to the line enable_cuda_graph = False line in the config passed to deepspeed_inference(enable_cuda_graph = True) . This will enable the optimisation using the default deepspeed launcher. Then run the command 

deepspeed --num_gpus  inference-test.py --model EleutherAI/gpt-neo-2.7B --dtype float16 --batch_size  --test_performance

Please change the num_gpus (1,2) and batch_size (upto 64) and run this command . 




9.To test the model Inference with deepspeed Transformer Kernel optimisation , just run this command:

deepspeed --num_gpus  inference-test.py --model EleutherAI/gpt-neo-2.7B --dtype float16 --batch_size --use_kernel  --test_performance

This will inject the kernel and run the optimised deepspeed launcher . This is where the prime reduction in latency happens. (Do not keep enable_cuda_graph=True in the inference_test.py file when running this command, as it is not supported by the deepspeed launcher. 

(Note giving num_gpus = 1,2 automatically applies Tensor parallelism (Model Parallelism and Pipeline) as it is already applied in The DSPipeline utility class in utils.py 

The latency results and the input and output prompts generated will be printed on the console. If you wish to use wandb , I have commented out the lines on code , you can uncomment and put your wandb credentials to start logging a wandb project.

The default max_tokens is 1024 , do not change that as all results are done using that . 






