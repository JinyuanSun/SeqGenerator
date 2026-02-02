# SeqGenerator

Function-orientedEnabling Diverse Enzyme Design viathrough Function-Oriented Sequence-driven Diffusion Model

<p align = "center">
<img src="img/training-generating-process.png" width="95%" alt="" align=center />
</p>
<p align = "center">
The training and generating process of our protein sequence diffusion model.
</p>


## Setup:
The code is based on PyTorch and HuggingFace `transformers`.
```bash 
pip install -r requirements.txt 
```

## Datasets
Prepare datasets and put them under the `datasets` folder. Take `datasets/aspartese` as an example. 

### Prepare CSV datasets from FASTA
Training expects CSV files with a header of `cluster,src,trg,len` and rows that store a representative sequence
(`src`), a target sequence (`trg`), and the target length (`len`). The loader in
`diffuseq/text_datasets.py` uses the `cluster` column to group sequences and `trg`
as the training sequence.

**FASTA input requirements**
- Each FASTA record must include a cluster identifier and a sequence identifier.
- Use the following header format (pipe separated): `>cluster_id|sequence_id`.
- Example:
  ```
  >cluster_0001|seq_001
  MKT...
  ```

**Suggested preparation workflow**
1. **Split the FASTA file** into train/valid/test FASTA files based on your clustering
   or random split strategy.
2. **Create CSVs** with four columns:
   - `cluster`: cluster identifier (e.g., `cluster_0001`)
   - `src`: representative sequence for the cluster (choose one sequence per cluster)
   - `trg`: target sequence (every sequence in the split can be a target)
   - `len`: length of `trg` (integer)

3. **Generate rows** by pairing each target sequence with its cluster representative.
   For example, if a cluster has representative sequence `REPSEQ` and targets
   `SEQ_A`, `SEQ_B`, then create:
   ```
   cluster,src,trg,len
   cluster_0001,REPSEQ,SEQ_A,350
   cluster_0001,REPSEQ,SEQ_B,342
   ```

4. **Save the CSV files** under `datasets/<name>/` as:
   - `train.csv`
   - `valid.csv`
   - `test.csv`

5. **Validate** the CSVs:
   - Ensure there are no extra commas or spaces.
   - Confirm `len` matches the actual `trg` length.
   - Use UTF-8 encoding without quotes around the sequences.

Once prepared, point `--data_dir` to `datasets/<name>` when training.

## Training
```bash
cd scripts
bash train.sh
```
Arguments explanation:
- ```--max_len```: the maximum length of the natrual sequences
- ```--min_len```: the minimum length of the natrual sequences
- ```--dataset```: the name of datasets, just for notation
- ```--data_dir```: the path to the saved datasets folder, containing ```train.csv  valid.csv```
- ```--resume_checkpoint```: if not none, restore this checkpoint and continue training
- ```--model_path```: the path to the used pretrained ESM-2 model, here we use "esm2_t30_150M_UR50D" which can be download [here](https://github.com/facebookresearch/esm?tab=readme-ov-file) and put them to "diffusion_models/esm_orig"

You can override defaults via environment variables:
```bash
DATASET=aspartese \
DATA_DIR=../datasets/aspartese \
MAX_LEN=490 \
MIN_LEN=460 \
MODEL_PATH=../diffusion_models/esm_orig/esm2_t30_150M_UR50D.pt \
bash train.sh
```


## Generating
You need to modify the path to ```model_dir```, which is obtained in the training stage.
```bash
cd scripts
bash run_decode.sh
```
Arguments explanation:
- ```--model_dir```: the model obtained in the training stage, our trained model can be accessed [here](https://zenodo.org/records/10405049), for generating put the model and 'training_args.json' to this folder
- ```--seq_len_sample```: the generated sequence length is obtained by sampling the length of the natural sequences of this family
- ```--max_len```: the maximum length of the generated sequence
- ```--min_len```: the minimum length of the generated sequence
- ```--seq_num```: the number of sequences generated

You can also use environment variables:
```bash
MODEL_DIR=../diffusion_models/your_run/ema*.pt \
SEQ_LEN_SAMPLE=../datasets/aspartese/train.csv \
MAX_LEN=490 \
MIN_LEN=460 \
SEQ_NUM=500 \
bash run_decode.sh
```


## Acknowledgements
The code in this project is based on [DiffuSeq](https://github.com/Shark-NLP/DiffuSeq) and [ESM-2](https://github.com/facebookresearch/esm). Special thanks to the original authors for their contributions to the open-source community.
