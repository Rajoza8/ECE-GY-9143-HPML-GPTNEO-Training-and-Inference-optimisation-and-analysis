# Training and Evaluation of Megatron-LM with DeepSpeed

## Setup
In order to setup the python environement, you can use the anaconda or pip package manager. The following command will install all the required packages:

```bash
conda create --name <env> --file environment.yml
```


```bash
pip install -r requirements.txt
```

After installing the above packages, one also have to compile the LSH library:
```bash
git clone https://github.com/mattilyra/LSH
cd LSH
python setup.py install
```

Alongside,the above packages, one also need to compile the nvidia apex library for their particular GPU in order to use CUDA kernels. It can be done using 

```bash
cd apex;
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ --no-build-isolation
```

## Dataset
For training and analyzing the performance of the GPT model, we use the openwebtext dataset, by first downloading the files from [here](https://mega.nz/#F!EZZD0YwJ!9_PlEQzdMVLaNdKv_ICNVQ).

Then we remove the blacklisted URLs from the dataset using the following command:

```
python Training/Megatron-DeepSpeed/tools/openwebtext/blacklist_urls.py <path to the dowloaded deduplicated URLs> <filename for clean urls. e.g. clean_urls.txt>
```

After blacklisting the URLs, we download the content from the clean urls with [openwebtext's utilities](Training/Megatron-DeepSpeed/tools/openwebtext/openwebtext/download.py)

Then we merge the contents into one loose json file with 1 json per newline of the format `{'text': text, 'url': unique_url}`. It is important for the url to be unique.

### Preparing the data for GPT training:
1. Perform ftfy, english detection and remove documents with less than 128 tokens. This step can be sharded and run on shards.
```
python Training/Megatron-DeepSpeed/tools/openwebtext/cleanup_dataset.py <input data file> <output cleaned data filename>
```
Additional cleanup (e.g. remove documents less than 512 characters or dataset specific cleaning like stories, realnews datasets) can be done using `cleanup_fix_dataset.py`. More details can be found by running `python cleanup_fix_dataset.py --help`.
2. Using LSH, find possible duplicates and store then in a file for later processing. The code supports saving and loading fingerprints for recurrent deduplications, and is also multithreaded for faster processing. More details are can be found by `python find_duplicate.py --help`.
```
python Training/Megatron-DeepSpeed/tools/openwebtext/find_duplicates.py --inputs <pairlist list of input cleaned data files and keys, e.g. cc.json cc_id news.json news_id> --output <output possible duplicate urls filename>
```

3. Shuffle the dataset.
```
shuf <cleaned deduped data file> -o train_data.json
```

After shuffling the dataset, we have to convert the dataset into the binary format that is used by the GPT model. This can be done using the following command:

```bash
python Training/Megatron-DeepSpeed/tools/preprocess_data.py \
    --input <path to the input file> \
    --output-prefix <path to the output file> \
    --vocab <path to the vocab file> \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file <path to the merge file> \
    --append-eod \
    --workers 1 \
    --log-interval 100 \
    --only-source
```

with the following [vocab.json](https://huggingface.co/gpt2/blob/main/vocab.json) and [merges.txt](https://huggingface.co/gpt2/blob/main/merges.txt)

Final dataset and index has been stored at Training/Megatron-DeepSpeed/output_data_text_document with both bin and idx file. One can give the path mentioned above to the shell scripts used in training. 

## Training the model

For standard vanilla training, we can use the following command:

```bash
sh Training/Megatron-DeepSpeed/examples/pretrain_gpt.sh
```

In case of pytorch DDP, use

```bash
sh Training/Megatron-DeepSpeed/examples/pretrain_gpt_distributed.sh
```

For training with model parallelism by Megatron-LM, use

```bash
sh Training/Megatron-DeepSpeed/examples/pretrain_gpt_distributed_with_mp.sh
```

and for training with deepspeed, use 

```bash
sh /home/mohit/NYU/HPML/ECE-GY-9143-HPML-GPTNEO-Training-and-Inference-optimisation-and-analysis/Training/Megatron-DeepSpeed/examples_deepspeed/run_deepspeed_example.sh
```
