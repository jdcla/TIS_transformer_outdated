# TIS transformer



<details><summary>Table of contents</summary>

- [About](#about)
- [Methodology](#methodology)
- [Benchmark](#benchmark)
- [Re-annotated Human Proteom](#human)
- [User guide](#userguide)
  - [Installation](#installation)
  - [Code](#code)

- [Citation](#citation)
</details>

## About <a name="about"></a>
The TIS transformer is created to annotate translation initiation sites on transcripts based on its nucleotide sequences. The repository holds the code and data to train the model on the human genome. The model presented in [this paper](https://www.google.com) can be directly used for the annotation of transcripts on humans.

## Methodology <a name="methodology"></a>
Annotations are performed by a machine learning model following a methodology similar to those defined for natural language modelling tasks. Ensemble annotations have been used to obtain transcripts and TIS annotations. The model processes the full transcript sequence to impute the presense of TIS at each position on the transcript. 

The model architecture is based on that of the [Performer](https://arxiv.org/abs/2009.14794), which allows the use of longer input sequences.

## Benchmark <a name="benchmark"></a>

The tool has been compared to similar approaches applying TIS imputation based on the transcript nucleotide sequence. More details about the benchmarking approach are listed in the [article](www.google.com). The scripts to obtain the scores for TISRover and TITER are deposited in `varia/`

<details>

**All positions**

| Method |  ROC AUC | PR AUC |
| - |  - | - |
| [TIS transformer]() |  99.99 | 85.63 | 

**Only ATG positions**

| Method |  ROC AUC | PR AUC |
| - |  - | - |
| [TIS transformer]() |  99.63 | 83.59 | 
| [TISRover](https://www.inderscience.com/info/inarticle.php?artid=94781) |  97.20 | 43.32 |
| [TITER](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5870772/) |  61.99 | 1.69 | 

</details>

## Re-annotation of the human proteome <a name="human"></a>

Using this method, the proteome of the complete human genome has been re-annotated by applying a cross-validation scheme. Annotations are performed on chromosomes not used to train the model. ([Biobix](), [Google drive]())

## User guide <a name="userguide"></a>

### Installation <a name="installation"></a>
To use the model, simply clone the repository in your working directory and install the necessary python (3.8.10) libraries.
<details><summary>Libraries</summary>

---
```
numpy 1.20.1
pytorch 1.8.1
pytorch-lightning 1.3.5
performer-pytorch 1.0.11

```
---

</details>

For linux users, the environment can be installed by running `conda env create -f environment.yml`

### Code <a name="code"></a>

The tool is run through `TIS_transformer.py`, which features three main functions: `pretrain`, `train` and `impute`. Models trained on the human genome are readily available, making only the `impute` function of interest to most users. 

#### pretrain

Conform with transformers trained for natural language processing objectives, a model can first be trained using self-supervised learning. Using a masked language modelling approach, the model is tasked to impute the classes of the masked input nucleotides. As such, a model is trained the 'semantics' of transcript sequences. The approach is similar to the one described by [Zaheer et al. ](https://arxiv.org/abs/2007.14062).

<details><summary>pretrain main arguments</summary>

---
```
python TIS_transformer.py pretrain -h

positional arguments:
    data_path             path to folder containing the data files
    val_set               file in data_path folder used for validation
    test_set              file in data_path folder used for testing
  --mask_frac float       fraction of inputs that are masked (default: 0.85)
  --rand_frac float       fraction of masked inputs that are randomized (default: 0.1)

# Example
    
python TIS_transformer.py pretrain 'data/GRCh38p13/' 'chr11.npy' 'chr4.npy' --max_epochs 70 --gpu 1 
```
---

</details>

#### train
Training the model using on a binary classification objective. The transcript sequence is processed as one. A label is assigned to every input nucleotide. Using the processed data a model can be trained to detect TISs. 

<details><summary>train main arguments</summary>

---
```
python TIS_transformer.py train -h
    
positional arguments:
  data_path             path to folder containing the data files
  val_set               file in data_path folder used for validation
  test_set              file in data_path folder used for testing
  --transfer_checkpoint Path to checkpoint pretrained model (default: None)
    
# Example
    
python TIS_transformer.py train 'data/GRCh38p13/' 'chr11.npy' 'chr4.npy' --max_epochs 70 --transfer_checkpoint lightning_logs/mlm_model/version_0/checkpoints/mlm_model.ckpt --name tis_chr4 --gpu 1 

```
---
    
</details>

#### impute

The impute function is used to obtain the predicted positions of TIS sites on a transcript. These predictions return probabilities for all nucleotide positions on the transcript, and are saved as numpy arrays. The code can handle the RNA sequence as input, or a list of transcripts given in a `.fa` or `.npy` format.

<details><summary>impute main arguments</summary>

---
```
python TIS_transformer.py impute -h

positional arguments:
  input               RNA sequence or path to `.npy` or `.fa` file
  transfer_checkpoint   path to checkpoint of trained model
  save_path           save file locations (default: results.npy)

# Example

python TIS_transformer.py impute AAAAACCTTTT lightning_logs/tis_chr4/version_0/checkpoints/tis_chr4.ckpt --gpu 1
    
python TIS_transformer.py impute example.fa lightning_logs/tis_chr4/version_0/checkpoints/tis_chr4.ckpt --gpu 1
    
python TIS_transformer.py impute data/GRCh38p13/chr4.npy lightning_logs/tis_chr4/version_0/checkpoints/tis_chr4.ckpt --gpu 1
```
---

</details>

### Other function flags

Other flags determine the architecture of the model or tweaking the training process

<details><summary>Dataloader flags</summary>

---
```
data loader arguments

  --max_seq_len int     maximum sequence length of transcripts (default: 25000)
  --num_workers int     number of data loader workers (default: 12)
  --max_transcripts_per_batch int
                        maximum amount of transcripts per batch (default: 400)
```
---

</details>

<details><summary>Model architecture flags</summary>

---
```
Model:
  Transformer arguments

  --transfer_checkpoint str
                        Path to checkpoint pretrained model (default: None)
  --lr float            learning rate (default: 0.001)
  --decay_rate float    linearly decays learning rate for every epoch (default: 0.95)
  --num_tokens int      number of unique input tokens (default: 7)
  --dim int             dimension of the hidden states (default: 30)
  --depth int           number of layers (default: 6)
  --heads int           number of attention heads in every layer (default: 6)
  --dim_head int        dimension of the attention head matrices (default: 16)
  --nb_features int     number of random features, if not set, will default to (d * log(d)),where d is the dimension
                        of each head (default: 80)
                        of each head (default: 80)
  --feature_redraw_interval int
                        how frequently to redraw the projection matrix (default: 100)
  --generalized_attention boolean
                        applies generalized attention functions (default: True)
  --kernel_fn boolean   generalized attention function to apply (if generalized attention) (default: ReLU())
  --reversible boolean  reversible layers, from Reformer paper (default: True)
  --ff_chunks int       chunk feedforward layer, from Reformer paper (default: 10)
  --use_scalenorm boolean
                        use scale norm, from 'Transformers without Tears' paper (default: False)
  --use_rezero boolean  use rezero, from 'Rezero is all you need' paper (default: False)
  --ff_glu boolean      use GLU variant for feedforward (default: True)
  --emb_dropout float   embedding dropout (default: 0.1)
  --ff_dropout float    feedforward dropout (default: 0.1)
  --attn_dropout float  post-attn dropout (default: 0.1)
  --local_attn_heads int
                        the amount of heads used for local attention (default: 4)
  --local_window_size int
                        window size of local attention (default: 256)
```
---

</details>

<details><summary>Pytorch lightning trainer flags</summary>

---
```
pl.Trainer:
  --logger [str_to_bool]
                        Logger (or iterable collection of loggers) for experiment tracking. A ``True`` value uses
                        the default ``TensorBoardLogger``. ``False`` will disable logging. (default: True)
  --checkpoint_callback [str_to_bool]
                        If ``True``, enable checkpointing. It will configure a default ModelCheckpoint callback if
                        there is no user-defined ModelCheckpoint in
                        :paramref:`~pytorch_lightning.trainer.trainer.Trainer.callbacks`. (default: True)
  --default_root_dir str
                        Default path for logs and weights when no logger/ckpt_callback passed. Default:
                        ``os.getcwd()``. Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'
                        (default: None)
  --gradient_clip_val float
                        0 means don't clip. (default: 0.0)
  --gradient_clip_algorithm str
                        'value' means clip_by_value, 'norm' means clip_by_norm. Default: 'norm' (default: norm)
  --process_position int
                        orders the progress bar when running multiple models on same machine. (default: 0)
  --num_nodes int       number of GPU nodes for distributed training. (default: 1)
  --num_processes int   number of processes for distributed training with distributed_backend="ddp_cpu" (default: 1)
  --gpus _gpus_allowed_type
                        number of gpus to train on (int) or which GPUs to train on (list or str) applied per node
                        (default: None)
  --auto_select_gpus [str_to_bool]
                        If enabled and `gpus` is an integer, pick available gpus automatically. This is especially
                        useful when GPUs are configured to be in "exclusive mode", such that only one process at a
                        time can access them. (default: False)
  --tpu_cores _gpus_allowed_type
                        How many TPU cores to train on (1 or 8) / Single TPU to train on [1] (default: None)
  --log_gpu_memory str  None, 'min_max', 'all'. Might slow performance (default: None)
  --progress_bar_refresh_rate int
                        How often to refresh progress bar (in steps). Value ``0`` disables progress bar. Ignored
                        when a custom progress bar is passed to :paramref:`~Trainer.callbacks`. Default: None, means
                        a suitable value will be chosen based on the environment (terminal, Google COLAB, etc.).
                        (default: None)
  --overfit_batches _int_or_float_type
                        Overfit a fraction of training data (float) or a set number of batches (int). (default: 0.0)
  --track_grad_norm float
                        -1 no tracking. Otherwise tracks that p-norm. May be set to 'inf' infinity-norm. (default:
                        -1)
  --check_val_every_n_epoch int
                        Check val every n train epochs. (default: 1)
  --fast_dev_run [str_to_bool_or_int]
                        runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es) of train, val and test to
                        find any bugs (ie: a sort of unit test). (default: False)
  --accumulate_grad_batches int
                        Accumulates grads every k batches or as set up in the dict. (default: 1)
  --max_epochs int      Stop training once this number of epochs is reached. Disabled by default (None). If both
                        max_epochs and max_steps are not specified, defaults to ``max_epochs`` = 1000. (default:
                        None)
  --min_epochs int      Force training for at least these many epochs. Disabled by default (None). If both
                        min_epochs and min_steps are not specified, defaults to ``min_epochs`` = 1. (default: None)
  --max_steps int       Stop training after this number of steps. Disabled by default (None). (default: None)
  --min_steps int       Force training for at least these number of steps. Disabled by default (None). (default:
                        None)
  --max_time str        Stop training after this amount of time has passed. Disabled by default (None). The time
                        duration can be specified in the format DD:HH:MM:SS (days, hours, minutes seconds), as a
                        :class:`datetime.timedelta`, or a dictionary with keys that will be passed to
                        :class:`datetime.timedelta`. (default: None)
  --limit_train_batches _int_or_float_type
                        How much of training dataset to check (float = fraction, int = num_batches) (default: 1.0)
  --limit_val_batches _int_or_float_type
                        How much of validation dataset to check (float = fraction, int = num_batches) (default: 1.0)
  --limit_test_batches _int_or_float_type
                        How much of test dataset to check (float = fraction, int = num_batches) (default: 1.0)
  --limit_predict_batches _int_or_float_type
                        How much of prediction dataset to check (float = fraction, int = num_batches) (default: 1.0)
  --val_check_interval _int_or_float_type
                        How often to check the validation set. Use float to check within a training epoch, use int
                        to check every n steps (batches). (default: 1.0)
  --flush_logs_every_n_steps int
                        How often to flush logs to disk (defaults to every 100 steps). (default: 100)
  --log_every_n_steps int
                        How often to log within steps (defaults to every 50 steps). (default: 50)
  --accelerator str     Previously known as distributed_backend (dp, ddp, ddp2, etc...). Can also take in an
                        accelerator object for custom hardware. (default: None)
  --sync_batchnorm [str_to_bool]
                        Synchronize batch norm layers between process groups/whole world. (default: False)
  --precision int       Double precision (64), full precision (32) or half precision (16). Can be used on CPU, GPU
                        or TPUs. (default: 32)
  --weights_summary str
                        Prints a summary of the weights when training begins. (default: top)
  --weights_save_path str
                        Where to save weights if specified. Will override default_root_dir for checkpoints only. Use
                        this if for whatever reason you need the checkpoints stored in a different place than the
                        logs written in `default_root_dir`. Can be remote file paths such as `s3://mybucket/path` or
                        'hdfs://path/' Defaults to `default_root_dir`. (default: None)
  --num_sanity_val_steps int
                        Sanity check runs n validation batches before starting the training routine. Set it to `-1`
                        to run all batches in all validation dataloaders. (default: 2)
  --truncated_bptt_steps int
                        Deprecated in v1.3 to be removed in 1.5. Please use
                        :paramref:`~pytorch_lightning.core.lightning.LightningModule.truncated_bptt_steps` instead.
                        (default: None)
  --resume_from_checkpoint str
                        Path/URL of the checkpoint from which training is resumed. If there is no checkpoint file at
                        the path, start from scratch. If resuming from mid-epoch checkpoint, training will start
                        from the beginning of the next epoch. (default: None)
  --profiler str        To profile individual steps during training and assist in identifying bottlenecks. (default:
                        None)
  --benchmark [str_to_bool]
                        If true enables cudnn.benchmark. (default: False)
  --deterministic [str_to_bool]
                        If true enables cudnn.deterministic. (default: False)
  --reload_dataloaders_every_epoch [str_to_bool]
                        Set to True to reload dataloaders every epoch. (default: False)
  --auto_lr_find [str_to_bool_or_str]
                        If set to True, will make trainer.tune() run a learning rate finder, trying to optimize
                        initial learning for faster convergence. trainer.tune() method will set the suggested
                        learning rate in self.lr or self.learning_rate in the LightningModule. To use a different
                        key set a string instead of True with the key name. (default: False)
  --replace_sampler_ddp [str_to_bool]
                        Explicitly enables or disables sampler replacement. If not specified this will toggled
                        automatically when DDP is used. By default it will add ``shuffle=True`` for train sampler
                        and ``shuffle=False`` for val/test sampler. If you want to customize it, you can set
                        ``replace_sampler_ddp=False`` and add your own distributed sampler. (default: True)
  --terminate_on_nan [str_to_bool]
                        If set to True, will terminate training (by raising a `ValueError`) at the end of each
                        training batch, if any of the parameters or the loss are NaN or +/-inf. (default: False)
  --auto_scale_batch_size [str_to_bool_or_str]
                        If set to True, will `initially` run a batch size finder trying to find the largest batch
                        size that fits into memory. The result will be stored in self.batch_size in the
                        LightningModule. Additionally, can be set to either `power` that estimates the batch size
                        through a power search or `binsearch` that estimates the batch size through a binary search.
                        (default: False)
  --prepare_data_per_node [str_to_bool]
                        If True, each LOCAL_RANK=0 will call prepare data. Otherwise only NODE_RANK=0, LOCAL_RANK=0
                        will prepare data (default: True)
  --plugins str         Plugins allow modification of core behavior like ddp and amp, and enable custom lightning
                        plugins. (default: None)
  --amp_backend str     The mixed precision backend to use ("native" or "apex") (default: native)
  --amp_level str       The optimization level to use (O1, O2, etc...). (default: O2)
  --distributed_backend str
                        deprecated. Please use 'accelerator' (default: None)
  --move_metrics_to_cpu [str_to_bool]
                        Whether to force internal logged metrics to be moved to cpu. This can save some gpu memory,
                        but can make training slower. Use with attention. (default: False)
  --multiple_trainloader_mode str
                        How to loop over the datasets when there are multiple train loaders. In 'max_size_cycle'
                        mode, the trainer ends one epoch when the largest dataset is traversed, and smaller datasets
                        reload when running out of their data. In 'min_size' mode, all the datasets reload when
                        reaching the minimum length of datasets. (default: max_size_cycle)
  --stochastic_weight_avg [str_to_bool]
                        Whether to use `Stochastic Weight Averaging (SWA)(default: False)
```
---

</details>

## Input data

To optimize and evaluate a model, data is processed into `.npy` format. For this study, The Ensemble (GRCh38p13, release 102) database was used to obtain the transcript sequences and annotations. These are stored in `data/GRCh38p13/`. Transcript data is separated by chromosome. The input data constitutes a ragged numpy array of transcripts (rows) where each entry holds the transcript ID and an array of input classes (columns 1) and corresponding target labels (column 2) for each nucleotide on the transcript. The input classes represent the different nucleotides as: `{A:0, T:1, C:2, G:3}`

<details>

---
```
>>> chr3 = np.load('chr3.npy', allow_pickle=True)
>>> chr3[0]
array([array([[3, 0],
              [3, 0],
              [2, 0],
              ...,
              [0, 0],
              [0, 0],
              [0, 0]], dtype=int16), 'ENST00000661675'], dtype=object)

```
---

</details>

## Output data

The model returns predictions for every nucleotide on the transcript, saved as a `.npy` file. For each transcript, the array lists the transcript label, input sequence and model outputs.

<details>

---
```
>>> results = np.load('results.npy', allow_pickle=True)
>>> results[0]
array(['>ENST00000410304',
       array([3, 1, 2, 1, 0, 2, 2, 3, 2, 2, 0, 1, 0, 2, 2, 0, 2, 1, 2, 1, 3, 0,
              0, 2, 0, 2, 3, 2, 2, 2, 0, 0, 1, 2, 1, 1, 3, 1, 2, 1, 3, 0, 1, 2,
              1, 2, 0, 3, 0, 0, 3, 2, 1, 0, 0, 3, 2, 0, 3, 3, 3, 1, 2, 0, 3, 3,
              2, 2, 1, 0, 3, 1, 1, 0, 3, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 3, 3, 0,
              1, 3, 3, 3, 3, 3, 0, 0, 0, 1, 3])                                ,
       array([2.3891837e-09, 7.0824785e-07, 8.3791534e-09, 4.3269135e-09,
              4.9220684e-08, 1.5315813e-10, 7.0196869e-08, 2.4103475e-10,
              4.5873511e-10, 1.4299616e-10, 6.1071654e-09, 1.9664975e-08,
              2.9255699e-07, 4.7719610e-08, 7.7600065e-10, 9.2305236e-10,
              3.3297397e-07, 3.5771163e-07, 4.1942007e-05, 4.5123262e-08,
              1.2450059e-09, 9.2165324e-11, 3.6457399e-09, 8.8559119e-08,
              9.2133210e-05, 1.7473910e-09, 4.0608841e-09, 2.9064828e-12,
              1.9478179e-08, 9.0584736e-12, 1.7068935e-05, 2.8910944e-07,
              3.5740332e-08, 3.3406838e-10, 5.7711222e-08, 5.0289093e-09,
              7.4243858e-12, 2.2184177e-09, 5.2881451e-06, 6.1195571e-10,
              1.4648888e-10, 1.4948037e-07, 2.3879443e-07, 1.6367457e-08,
              1.9375465e-08, 3.3595885e-08, 4.1618881e-10, 6.3614699e-12,
              4.1953702e-10, 1.3611480e-08, 2.0185058e-09, 8.1397658e-08,
              2.3339116e-07, 4.8850779e-08, 1.6549968e-12, 1.2499275e-11,
              8.3455109e-10, 1.5468280e-12, 3.5863316e-08, 1.2135585e-09,
              4.4234839e-14, 2.0041482e-11, 4.0546926e-09, 4.8796110e-12,
              3.4575018e-13, 5.0659910e-10, 3.2857072e-13, 2.3365734e-09,
              8.3198276e-10, 2.9397595e-10, 3.3731489e-08, 9.1637538e-11,
              1.0781720e-09, 1.0790679e-11, 4.8457072e-10, 4.6192927e-10,
              4.9371015e-12, 2.8158498e-13, 2.9590792e-09, 4.3507330e-07,
              5.7654831e-10, 2.4951474e-09, 4.6289192e-12, 1.5421598e-02,
              1.0270607e-11, 1.1841109e-09, 7.9038587e-10, 6.5511790e-10,
              6.0892291e-13, 1.6157842e-11, 6.9130129e-10, 4.5778301e-11,
              2.1682500e-03, 2.3315516e-09, 2.2578116e-11], dtype=float32)],
      dtype=object)

```
---

</details>

#

## Citation <a name="citation"></a>
       
