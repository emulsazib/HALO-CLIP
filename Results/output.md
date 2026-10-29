# Test 1 Flickr30k
## Load Dataset from Hugging Face 
> Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

> WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

README.md: 100% 2.23k/2.23k [00:00<00:00, 269kB/s]

- data/test-00000-of-00009.parquet: reconstructing file: 100%  471MB /  471MB, 43.0MB/s  
- data/test-00000-of-00009.parquet: downloading bytes:   471MB, 42.1MB/s  
- data/test-00001-of-00009.parquet: reconstructing file: 100%  475MB /  475MB, 44.9MB/s  
- data/test-00001-of-00009.parquet: downloading bytes:   475MB, 44.1MB/s  
- data/test-00002-of-00009.parquet: reconstructing file: 100%  486MB /  486MB, 20.4MB/s  
- data/test-00002-of-00009.parquet: downloading bytes:   485MB, 11.4MB/s  
- data/test-00003-of-00009.parquet: reconstructing file: 100%  472MB /  472MB, 44.9MB/s  
- data/test-00003-of-00009.parquet: downloading bytes:   472MB, 44.4MB/s  
- data/test-00004-of-00009.parquet: reconstructing file: 100%  490MB /  490MB, 46.5MB/s  
- data/test-00004-of-00009.parquet: downloading bytes:   489MB, 45.3MB/s  
- data/test-00005-of-00009.parquet: reconstructing file: 100%  500MB /  500MB, 45.1MB/s  
- data/test-00005-of-00009.parquet: downloading bytes:   496MB, 42.8MB/s  
- data/test-00006-of-00009.parquet: reconstructing file: 100%  528MB /  528MB, 50.0MB/s  
- data/test-00006-of-00009.parquet: downloading bytes:   522MB, 48.5MB/s  
- data/test-00007-of-00009.parquet: reconstructing file: 100%  511MB /  511MB, 48.5MB/s  
- data/test-00007-of-00009.parquet: downloading bytes:   506MB, 47.5MB/s  
- data/test-00008-of-00009.parquet: reconstructing file: 100%  477MB /  477MB, 45.0MB/s  
- data/test-00008-of-00009.parquet: downloading bytes:   477MB, 43.8MB s  

Generating test split: 100% 31783/31783 [00:10<00:00, 3690.12 examples/s]
Dataset({
    features: ['image', 'caption', 'sentids', 'img_id', 'filename'],
    num_rows: 31783
})
![Load Dataset](https://github.com/emulsazib/HALO-CLIP/blob/main/LLM2CLIP_Output/Results/load_dataset.png?raw=true)
train images: 30,783 (153,915 captions) | val images: 1,000 (5,000 captions)
• Two young guys with shaggy hair look at their hands while hanging out in the yard .
• Two young  White males are outside near many bushes .
• Two men in green shirts are standing in a yard .
• A man in a blue shirt standing in a garden .
• Two friends enjoy time spent together .

## Embed captions with the LLM2CLIP LLM (frozen, cached)
 config.json: 100% 1.02k/1.02k [00:00<00:00, 102kB/s]tokenizer_config.json: 100% 54.6k/54.6k [00:00<00:00, 5.74MB/s]tokenizer.json: 100% 9.09M/9.09M [00:00<00:00, 133MB/s]special_tokens_map.json: 100% 325/325 [00:00<00:00, 41.6kB/s]model.
 
- safetensors: reconstructing file: 100% 4.94GB / 4.94GB,  283MB/s  model.safetensors: downloading bytes:  4.15GB,  307MB/s  
 
- Loading weights: 100% 146/146 [00:00<00:00, 571.81it/s]LLM caption embeddings: 100% 1203/1203 [01:13<00:00, 24.50it/s]
 
- LLM caption embeddings: 100% 40/40 [00:02<00:00, 20.73it/s]Embedded 158,915 captions in 76s train_emb (153915, 2048) | val_emb (5000, 2048)

## Build the CLIP model (repo's eva_clip) and load weights from Hugging Face
LLM2CLIP-Llama3.2-1B-EVA02-L-14-336.pt:reconstructing file: 100%  750MB /  750MB, 65.6MB/s  LLM2CLIP-Llama3.2-1B-EVA02-L-14-336.pt: downloading bytes:   750MB, 61.4MB/s  Loaded all weights from LLM2CLIP-Llama3.2-1B-EVA02-L-14-336.pt
Llama3.2-1B-EVA02-L-14-336: 375M params, 375M trainable (vision 305M, adapter 70M)

## Training curves
![Training carve](https://github.com/emulsazib/HALO-CLIP/blob/main/LLM2CLIP_Output/Results/training_carve.png?raw=true)


- epoch 0 | i2t R@1/5/10 94.7/99.5/99.8 | t2i R@1/5/10 81.5/95.5/97.7 | mean R@1 88.12 (before training)
- ★ new best → /content/drive/MyDrive/llm2clip_runs/llm2clip_flickr30k/best_model.pt
- epoch 1/3: 100% 240/240 [11:05<00:00,  2.75s/it, i2t=0.94, loss=0.184, scale=99.9]epoch 1 | i2t R@1/5/10 93.2/99.3/99.8 | t2i R@1/5/10 81.7/96.1/98.0 | mean R@1 87.47 | train loss 0.161 | 11.2 min
- epoch 2/3: 100% 240/240 [11:04<00:00,  2.75s/it, i2t=0.98, loss=0.079, scale=99.9]epoch 2 | i2t R@1/5/10 95.3/99.5/99.7 | t2i R@1/5/10 83.5/96.7/98.2 | mean R@1 89.42 | train loss 0.138 | 11.2 min
- ★ new best → /content/drive/MyDrive/llm2clip_runs/llm2clip_flickr30k/best_model.pt
- epoch 3/3: 100% 240/240 [11:04<00:00,  2.75s/it, i2t=0.98, loss=0.064, scale=99.9]epoch 3 | i2t R@1/5/10 95.2/99.4/99.7 | t2i R@1/5/10 84.5/97.0/98.5 | mean R@1 89.87 | train loss 0.101 | 11.2 min
- ★ new best → /content/drive/MyDrive/llm2clip_runs/llm2clip_flickr30k/best_model.pt

**Best: epoch 3 | mean R@1 89.87 | rsum 574.3**

## Sanity check: retrieve captions for val images with the best weights
![Sanity Check](https://github.com/emulsazib/HALO-CLIP/blob/main/LLM2CLIP_Output/Results/sanity_check.png?raw=true)
--- 
# Test 2 COCO1k
/usr/local/lib/python3.13/dist-packages/timm/models/layers/__init__.py:49: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
Please 'pip install apex'
GPU: NVIDIA A100-SXM4-40GB (42 GB) | AMP: torch.bfloat16 | xformers: False (SDPA fallback otherwise)
Batch size 128 | visual LR 1e-05 | adapter LR 1e-05
Mounted at /content/drive
Outputs → /content/drive/MyDrive/llm2clip_runs/llm2clip_flickr30k
<torch._C.Generator at 0x7f49d3f75ff0>

 README.md: 100% 641/641 [00:00<00:00, 62.1kB/s]Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
(…)oco_2014_5k_test_image_text_retrieval.py: 100% 2.90k/2.90k [00:00<00:00, 354kB/s]TEST/test/0000.parquet: reconstructing file: 100%  510MB /  510MB, 22.9MB/s  TEST/test/0000.parquet: downloading bytes:   509MB, 21.4MB/s  TEST/test/0001.parquet: reconstructing file: 100%  311MB /  311MB, 25.6MB/s  TEST/test/0001.parquet: downloading bytes:   310MB, 22.1MB/s  Generating test split:  5000/0 [00:01<00:00, 4226.31 examples/s]Dataset({
    features: ['image', 'caption', 'sentids', 'imgid', 'split', 'cocoid', 'filename', 'filepath', 'tokens', 'sentid'],
    num_rows: 5000
})
![Load Dataset COCO1k]()
train images: 4,000 (20,000 captions) | val images: 1,000 (5,000 captions)
• A man with a red helmet on a small moped on a dirt road.
• Man riding a motor bike on a dirt road on the countryside.
• A man riding on the back of a motorcycle.
• A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains.
• A man in a red shirt and a red hat is on a motorcycle on a hill side.

 config.json: 100% 1.02k/1.02k [00:00<00:00, 128kB/s]tokenizer_config.json: 100% 54.6k/54.6k [00:00<00:00, 6.35MB/s]tokenizer.json: 100% 9.09M/9.09M [00:00<00:00, 27.4MB/s]special_tokens_map.json: 100% 325/325 [00:00<00:00, 43.6kB/s]model.safetensors: reconstructing file: 100% 4.94GB / 4.94GB,  264MB/s  model.safetensors: downloading bytes:  4.15GB,  325MB/s  Loading weights: 100% 146/146 [00:00<00:00, 555.92it/s]LLM caption embeddings: 100% 157/157 [00:10<00:00, 21.84it/s]LLM caption embeddings: 100% 40/40 [00:02<00:00, 20.29it/s]Embedded 25,000 captions in 12s
train_emb (20000, 2048) | val_emb (5000, 2048)

 LLM2CLIP-Llama3.2-1B-EVA02-L-14-336.pt: reconstructing file: 100%  750MB /  750MB, 67.8MB/s  LLM2CLIP-Llama3.2-1B-EVA02-L-14-336.pt: downloading bytes:   750MB, 64.3MB/s  Loaded all weights from LLM2CLIP-Llama3.2-1B-EVA02-L-14-336.pt
Llama3.2-1B-EVA02-L-14-336: 375M params, 375M trainable (vision 305M, adapter 70M)

Resumed after epoch 3 (best mean R@1 89.87)

Best: epoch 3 | mean R@1 89.87 | rsum 574.3

![Learning Carve]()

![Check]()

README.md: 100% | 754/754 [00:00<00:00, 91.7kB/s]
flickr_1k_test_image_text_retrieval.py: 100% | 2.79k/2.79k [00:00<00:00, 350kB/s]
TEST/test/0000.parquet: reconstructing file: 100% | 142MB / 142MB, 13.5MB/s
TEST/test/0000.parquet: downloading bytes: 141MB, 13.4MB/s
Generating test split: 1000/0 [00:00<00:00, 4260.38 examples/s]

Using the latest cached version of the dataset since nlphuji/mscoco_2014_5k_test_image_text_retrieval couldn't be found on the Hugging Face Hub
WARNING:datasets.load:Using the latest cached version of the dataset since nlphuji/mscoco_2014_5k_test_image_text_retrieval couldn't be found on the Hugging Face Hub

Flickr: 1,000 images, 5,000 captions (nlphuji/flickr_1k_test_image_text_retrieval)
COCO: 5,000 images, 25,010 captions (nlphuji/mscoco_2014_5k_test_image_text_retrieval)

Loading weights: 100% | 146/146 [00:00<00:00, 561.66it/s]
LLM caption embeddings: 100% | 40/40 [00:02<00:00, 20.40it/s]
LLM caption embeddings: 100% | 196/196 [00:11<00:00, 20.92it/s]
Loaded all weights from LLM2CLIP-Llama3.2-1B-EVA02-L-14-336.pt

/tmp/ipykernel_3303/671931697.py:119: UserWarning: Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected, since the `weights_only` argument was not explicitly passed to `torch.load`, forcing weights_only=False.
  model.load_state_dict(torch.load(BEST_PATH, map_location="cpu"))

# Test 3 ShareGPT4V
 Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
share-captioner_coco_lcs_sam_1246k_1107.(…): reconstructing file: 100% 1.49GB / 1.49GB,  104MB/s  share-captioner_coco_lcs_sam_1246k_1107.(…): downloading bytes:   553MB, 34.6MB/s  annotations: share-captioner_coco_lcs_sam_1246k_1107.json (1.49 GB)
read 591,486 entries | COCO collected 21,000/21,000
paper's SG4V-1K (first 1000 entries) by source: {'sam': 1000}
  images 1,000/21,000 | failed 0 | 82.3 img/s | 0:00:12
  images 2,000/21,000 | failed 0 | 89.7 img/s | 0:00:22
  images 3,000/21,000 | failed 0 | 92.8 img/s | 0:00:32
  images 4,000/21,000 | failed 0 | 92.0 img/s | 0:00:43
  images 5,000/21,000 | failed 0 | 93.7 img/s | 0:00:53
  images 6,000/21,000 | failed 0 | 94.8 img/s | 0:01:03
  images 7,000/21,000 | failed 0 | 94.0 img/s | 0:01:14
  images 8,000/21,000 | failed 0 | 94.9 img/s | 0:01:24
  images 9,000/21,000 | failed 0 | 95.6 img/s | 0:01:34
  images 10,000/21,000 | failed 0 | 95.1 img/s | 0:01:45
  images 11,000/21,000 | failed 0 | 95.4 img/s | 0:01:55
  images 12,000/21,000 | failed 0 | 95.8 img/s | 0:02:05
  images 13,000/21,000 | failed 0 | 95.4 img/s | 0:02:16
  images 14,000/21,000 | failed 0 | 95.6 img/s | 0:02:26
  images 15,000/21,000 | failed 0 | 96.0 img/s | 0:02:36
  images 16,000/21,000 | failed 0 | 95.9 img/s | 0:02:46
  images 17,000/21,000 | failed 0 | 95.9 img/s | 0:02:57
  images 18,000/21,000 | failed 0 | 96.2 img/s | 0:03:07
  images 19,000/21,000 | failed 0 | 96.4 img/s | 0:03:17
  images 20,000/21,000 | failed 0 | 96.2 img/s | 0:03:27
  images 21,000/21,000 | failed 0 | 96.4 img/s | 0:03:37
train images: 20,000 | val images: 1,000 (1 long caption each)
average caption length: 145 words
![Load Dataset]()

- The image captures a lively scene at a ski resort on a snowy day. The sky above is a clear blue, providing a beautiful contrast to the white snow below. The resort is bustling with people, all dressed in winter clothing and carrying ski equipment, ready to enjoy the day on the slopes.

- In the center of the image, a group of people can be seen walking towards a ski lift, their ski equipment in hand. The ski lift, a prominent feature in the image, is located on the right side of the frame. It stands tall against the backdrop of the snow-covered mountains, ready to transport skiers to the top of  ...

config.json: 100% 835/835 [00:00<00:00, 111kB/s]
tokenizer_config.json: 100% 51.0k/51.0k [00:00<00:00, 159kB/s]
tokenizer.json: 100% 9.09M/9.09M [00:00<00:00, 129MB/s]
special_tokens_map.json: 100% 335/335 [00:00<00:00, 41.8kB/s]
model.safetensors.index.json: 100% 22.2k/22.2k [00:00<00:00, 2.80MB/s]
Download complete:  12.9GB,  206MB/s  
Reconstruction complete: 100% 15.0GB / 15.0GB,  617MB/s  
Fetching 4 files: 100% 4/4 [00:42<00:00, 24.71s/it]
Loading weights: 100% 290/290 [00:00<00:00, 3685.09it/s]
LLM caption embeddings: 100% 329/329 [05:29<00:00,  1.84it/s]embedded 21,000 captions in 0:05:29
train_emb (20000, 4096) | val_emb (1000, 4096)

LLM2CLIP-EVA02-L-14-336.pt: reconstructing file: 100% 1.16GB / 1.16GB, 94.5MB/s  
LLM2CLIP-EVA02-L-14-336.pt: downloading bytes:  1.16GB, 80.3MB/s  Loaded all weights from LLM2CLIP-EVA02-L-14-336.pt
EVA02-CLIP-L-14-336: 579M params, 579M trainable (vision 305M, adapter 274M)
```
==============================================================================================================
epoch 0 (before training) | I2T R@1/5/10 98.8/99.9/100.0 | T2I R@1/5/10 98.7/99.8/99.9 | mean R@1 98.75
  * new best (mean R@1 98.75) -> /content/drive/MyDrive/llm2clip_runs/llm2clip_sg4v/best_model.pt
==============================================================================================================
[ep 1/3 | step   10/156 |   6%] loss 0.0028 | i2t_acc 1.00 t2i_acc 1.00 | lr 1.10e-06 | scale 100.0 | gnorm 1.46 | 42.8 img/s | mem 25.4GB | elapsed 0:00:29 | eta 0:22:51
[ep 1/3 | step   20/156 |  13%] loss 0.0033 | i2t_acc 1.00 t2i_acc 1.00 | lr 2.10e-06 | scale 100.0 | gnorm 0.12 | 46.1 img/s | mem 25.4GB | elapsed 0:00:57 | eta 0:21:32
[ep 1/3 | step   30/156 |  19%] loss 0.0072 | i2t_acc 1.00 t2i_acc 1.00 | lr 3.10e-06 | scale 100.0 | gnorm 3.80 | 46.1 img/s | mem 25.4GB | elapsed 0:01:25 | eta 0:20:47
[ep 1/3 | step   40/156 |  26%] loss 0.0013 | i2t_acc 1.00 t2i_acc 1.00 | lr 4.10e-06 | scale 100.0 | gnorm 0.25 | 46.1 img/s | mem 25.4GB | elapsed 0:01:53 | eta 0:20:11
[ep 1/3 | step   50/156 |  32%] loss 0.0067 | i2t_acc 1.00 t2i_acc 1.00 | lr 5.10e-06 | scale 100.0 | gnorm 0.01 | 46.1 img/s | mem 25.4GB | elapsed 0:02:20 | eta 0:19:38
[ep 1/3 | step   60/156 |  38%] loss 0.0046 | i2t_acc 1.00 t2i_acc 1.00 | lr 6.10e-06 | scale 100.0 | gnorm 3.75 | 46.1 img/s | mem 25.4GB | elapsed 0:02:48 | eta 0:19:07
[ep 1/3 | step   70/156 |  45%] loss 0.0067 | i2t_acc 1.00 t2i_acc 1.00 | lr 7.10e-06 | scale 100.0 | gnorm 0.24 | 46.1 img/s | mem 25.4GB | elapsed 0:03:16 | eta 0:18:36
[ep 1/3 | step   80/156 |  51%] loss 0.0095 | i2t_acc 1.00 t2i_acc 0.99 | lr 8.10e-06 | scale 100.0 | gnorm 2.93 | 46.1 img/s | mem 25.4GB | elapsed 0:03:44 | eta 0:18:07
[ep 1/3 | step   90/156 |  58%] loss 0.0125 | i2t_acc 0.99 t2i_acc 1.00 | lr 9.10e-06 | scale 100.0 | gnorm 0.10 | 46.1 img/s | mem 25.4GB | elapsed 0:04:11 | eta 0:17:38
[ep 1/3 | step  100/156 |  64%] loss 0.0158 | i2t_acc 1.00 t2i_acc 0.99 | lr 1.00e-05 | scale 100.0 | gnorm 4.71 | 46.1 img/s | mem 25.4GB | elapsed 0:04:39 | eta 0:17:09
[ep 1/3 | step  110/156 |  71%] loss 0.0198 | i2t_acc 1.00 t2i_acc 0.99 | lr 9.98e-06 | scale 100.0 | gnorm 4.33 | 46.1 img/s | mem 25.4GB | elapsed 0:05:07 | eta 0:16:40
[ep 1/3 | step  120/156 |  77%] loss 0.0139 | i2t_acc 1.00 t2i_acc 0.99 | lr 9.93e-06 | scale 100.0 | gnorm 2.84 | 46.1 img/s | mem 25.4GB | elapsed 0:05:35 | eta 0:16:12
[ep 1/3 | step  130/156 |  83%] loss 0.0223 | i2t_acc 1.00 t2i_acc 0.99 | lr 9.84e-06 | scale 100.0 | gnorm 6.16 | 46.1 img/s | mem 25.4GB | elapsed 0:06:02 | eta 0:15:43
[ep 1/3 | step  140/156 |  90%] loss 0.0164 | i2t_acc 1.00 t2i_acc 0.99 | lr 9.71e-06 | scale 100.0 | gnorm 0.19 | 46.1 img/s | mem 25.4GB | elapsed 0:06:30 | eta 0:15:15
[ep 1/3 | step  150/156 |  96%] loss 0.0137 | i2t_acc 1.00 t2i_acc 0.99 | lr 9.55e-06 | scale 100.0 | gnorm 2.11 | 46.1 img/s | mem 25.4GB | elapsed 0:06:58 | eta 0:14:47
[ep 1/3 | step  156/156 | 100%] loss 0.0161 | i2t_acc 1.00 t2i_acc 0.99 | lr 9.44e-06 | scale 100.0 | gnorm 0.75 | 46.1 img/s | mem 25.4GB | elapsed 0:07:15 | eta 0:14:30
--------------------------------------------------------------------------------------------------------------
epoch 1/3 done | I2T R@1/5/10 96.3/99.8/99.9 | T2I R@1/5/10 95.9/99.6/99.9 | mean R@1 96.10 | train loss 0.0107 | 7.4 min
--------------------------------------------------------------------------------------------------------------
[ep 2/3 | step   10/156 |   6%] loss 0.0164 | i2t_acc 1.00 t2i_acc 0.99 | lr 9.23e-06 | scale 100.0 | gnorm 0.17 | 44.2 img/s | mem 25.4GB | elapsed 0:08:10 | eta 0:14:52
[ep 2/3 | step   20/156 |  13%] loss 0.0140 | i2t_acc 1.00 t2i_acc 0.99 | lr 8.98e-06 | scale 99.9 | gnorm 6.50 | 46.1 img/s | mem 25.4GB | elapsed 0:08:38 | eta 0:14:19
[ep 2/3 | step   30/156 |  19%] loss 0.0063 | i2t_acc 1.00 t2i_acc 1.00 | lr 8.71e-06 | scale 99.9 | gnorm 4.62 | 46.1 img/s | mem 25.4GB | elapsed 0:09:05 | eta 0:13:47
[ep 2/3 | step   40/156 |  26%] loss 0.0189 | i2t_acc 1.00 t2i_acc 0.99 | lr 8.41e-06 | scale 99.9 | gnorm 5.71 | 46.1 img/s | mem 25.4GB | elapsed 0:09:33 | eta 0:13:16
[ep 2/3 | step   50/156 |  32%] loss 0.0122 | i2t_acc 1.00 t2i_acc 0.99 | lr 8.09e-06 | scale 99.9 | gnorm 2.93 | 46.1 img/s | mem 25.4GB | elapsed 0:10:01 | eta 0:12:44
[ep 2/3 | step   60/156 |  38%] loss 0.0094 | i2t_acc 1.00 t2i_acc 0.99 | lr 7.74e-06 | scale 99.9 | gnorm 6.23 | 46.1 img/s | mem 25.4GB | elapsed 0:10:29 | eta 0:12:13
[ep 2/3 | step   70/156 |  45%] loss 0.0157 | i2t_acc 1.00 t2i_acc 0.99 | lr 7.38e-06 | scale 99.9 | gnorm 1.82 | 46.1 img/s | mem 25.4GB | elapsed 0:10:56 | eta 0:11:43
[ep 2/3 | step   80/156 |  51%] loss 0.0215 | i2t_acc 1.00 t2i_acc 0.99 | lr 6.99e-06 | scale 99.9 | gnorm 4.12 | 46.1 img/s | mem 25.4GB | elapsed 0:11:24 | eta 0:11:13
[ep 2/3 | step   90/156 |  58%] loss 0.0071 | i2t_acc 1.00 t2i_acc 1.00 | lr 6.59e-06 | scale 99.9 | gnorm 5.97 | 46.1 img/s | mem 25.4GB | elapsed 0:11:52 | eta 0:10:42
[ep 2/3 | step  100/156 |  64%] loss 0.0072 | i2t_acc 1.00 t2i_acc 1.00 | lr 6.18e-06 | scale 99.9 | gnorm 1.03 | 46.1 img/s | mem 25.4GB | elapsed 0:12:20 | eta 0:10:12
[ep 2/3 | step  110/156 |  71%] loss 0.0098 | i2t_acc 1.00 t2i_acc 0.99 | lr 5.77e-06 | scale 99.9 | gnorm 4.44 | 46.1 img/s | mem 25.4GB | elapsed 0:12:47 | eta 0:09:43
[ep 2/3 | step  120/156 |  77%] loss 0.0120 | i2t_acc 1.00 t2i_acc 1.00 | lr 5.34e-06 | scale 99.9 | gnorm 5.83 | 46.1 img/s | mem 25.4GB | elapsed 0:13:15 | eta 0:09:13
[ep 2/3 | step  130/156 |  83%] loss 0.0108 | i2t_acc 1.00 t2i_acc 0.99 | lr 4.91e-06 | scale 99.9 | gnorm 3.22 | 46.1 img/s | mem 25.4GB | elapsed 0:13:43 | eta 0:08:43
[ep 2/3 | step  140/156 |  90%] loss 0.0055 | i2t_acc 1.00 t2i_acc 1.00 | lr 4.49e-06 | scale 99.9 | gnorm 0.41 | 46.1 img/s | mem 25.4GB | elapsed 0:14:11 | eta 0:08:14
[ep 2/3 | step  150/156 |  96%] loss 0.0056 | i2t_acc 1.00 t2i_acc 1.00 | lr 4.07e-06 | scale 99.9 | gnorm 0.10 | 46.1 img/s | mem 25.4GB | elapsed 0:14:38 | eta 0:07:45
[ep 2/3 | step  156/156 | 100%] loss 0.0184 | i2t_acc 1.00 t2i_acc 0.99 | lr 3.82e-06 | scale 99.9 | gnorm 3.82 | 46.1 img/s | mem 25.4GB | elapsed 0:14:55 | eta 0:07:27
--------------------------------------------------------------------------------------------------------------
epoch 2/3 done | I2T R@1/5/10 98.4/99.9/99.9 | T2I R@1/5/10 96.5/99.8/99.9 | mean R@1 97.45 | train loss 0.0118 | 7.4 min
--------------------------------------------------------------------------------------------------------------
[ep 3/3 | step   10/156 |   6%] loss 0.0070 | i2t_acc 1.00 t2i_acc 0.99 | lr 3.41e-06 | scale 99.9 | gnorm 0.52 | 44.2 img/s | mem 25.4GB | elapsed 0:15:50 | eta 0:07:11
[ep 3/3 | step   20/156 |  13%] loss 0.0065 | i2t_acc 1.00 t2i_acc 1.00 | lr 3.01e-06 | scale 99.9 | gnorm 0.72 | 46.1 img/s | mem 25.4GB | elapsed 0:16:18 | eta 0:06:40
[ep 3/3 | step   30/156 |  19%] loss 0.0167 | i2t_acc 1.00 t2i_acc 0.99 | lr 2.62e-06 | scale 99.9 | gnorm 2.53 | 46.1 img/s | mem 25.4GB | elapsed 0:19:22 | eta 0:07:08
[ep 3/3 | step   40/156 |  26%] loss 0.0066 | i2t_acc 1.00 t2i_acc 1.00 | lr 2.26e-06 | scale 99.9 | gnorm 2.28 | 46.1 img/s | mem 25.4GB | elapsed 0:19:49 | eta 0:06:32
[ep 3/3 | step   50/156 |  32%] loss 0.0076 | i2t_acc 1.00 t2i_acc 1.00 | lr 1.91e-06 | scale 99.9 | gnorm 1.55 | 46.1 img/s | mem 25.4GB | elapsed 0:20:17 | eta 0:05:56
[ep 3/3 | step   60/156 |  38%] loss 0.0079 | i2t_acc 1.00 t2i_acc 1.00 | lr 1.59e-06 | scale 99.9 | gnorm 0.07 | 46.1 img/s | mem 25.4GB | elapsed 0:20:45 | eta 0:05:21
[ep 3/3 | step   70/156 |  45%] loss 0.0089 | i2t_acc 1.00 t2i_acc 1.00 | lr 1.29e-06 | scale 99.9 | gnorm 2.31 | 46.1 img/s | mem 25.4GB | elapsed 0:21:13 | eta 0:04:46
[ep 3/3 | step   80/156 |  51%] loss 0.0043 | i2t_acc 1.00 t2i_acc 1.00 | lr 1.02e-06 | scale 99.9 | gnorm 3.46 | 46.1 img/s | mem 25.4GB | elapsed 0:21:41 | eta 0:04:12
[ep 3/3 | step   90/156 |  58%] loss 0.0012 | i2t_acc 1.00 t2i_acc 1.00 | lr 7.73e-07 | scale 99.9 | gnorm 0.09 | 46.1 img/s | mem 25.4GB | elapsed 0:22:08 | eta 0:03:38
[ep 3/3 | step  100/156 |  64%] loss 0.0057 | i2t_acc 1.00 t2i_acc 1.00 | lr 5.61e-07 | scale 99.9 | gnorm 0.29 | 46.1 img/s | mem 25.4GB | elapsed 0:22:36 | eta 0:03:04
[ep 3/3 | step  110/156 |  71%] loss 0.0039 | i2t_acc 1.00 t2i_acc 1.00 | lr 3.81e-07 | scale 99.9 | gnorm 0.86 | 46.1 img/s | mem 25.4GB | elapsed 0:23:04 | eta 0:02:30
[ep 3/3 | step  120/156 |  77%] loss 0.0108 | i2t_acc 1.00 t2i_acc 0.99 | lr 2.34e-07 | scale 99.9 | gnorm 3.35 | 46.1 img/s | mem 25.4GB | elapsed 0:23:32 | eta 0:01:57
[ep 3/3 | step  130/156 |  83%] loss 0.0035 | i2t_acc 1.00 t2i_acc 1.00 | lr 1.23e-07 | scale 99.9 | gnorm 0.82 | 46.1 img/s | mem 25.4GB | elapsed 0:23:59 | eta 0:01:24
[ep 3/3 | step  140/156 |  90%] loss 0.0055 | i2t_acc 1.00 t2i_acc 1.00 | lr 4.66e-08 | scale 99.9 | gnorm 3.83 | 46.1 img/s | mem 25.4GB | elapsed 0:24:27 | eta 0:00:51
[ep 3/3 | step  150/156 |  96%] loss 0.0012 | i2t_acc 1.00 t2i_acc 1.00 | lr 6.56e-09 | scale 99.9 | gnorm 0.08 | 46.1 img/s | mem 25.4GB | elapsed 0:24:55 | eta 0:00:19
[ep 3/3 | step  156/156 | 100%] loss 0.0019 | i2t_acc 1.00 t2i_acc 1.00 | lr 0.00e+00 | scale 99.9 | gnorm 0.42 | 46.1 img/s | mem 25.4GB | elapsed 0:25:11 | eta 0:00:00
--------------------------------------------------------------------------------------------------------------
epoch 3/3 done | I2T R@1/5/10 98.0/99.9/99.9 | T2I R@1/5/10 98.2/99.9/99.9 | mean R@1 98.10 | train loss 0.0063 | 10.0 min
--------------------------------------------------------------------------------------------------------------

Best: epoch 0 | mean R@1 98.75 | rsum 597.1
```


## 🏆 Training Summary & Best Model

*   **Best Epoch:** 0 (Before Training)
*   **Mean R@1:** 98.75
*   **Rsum:** 597.1
*   **Saved Path:** `/content/drive/MyDrive/llm2clip_runs/llm2clip_sg4v/best_model.pt`

## 📊 Epoch Evaluations

| Epoch | I2T R@1 / 5 / 10 | T2I R@1 / 5 / 10 | Mean R@1 | Train Loss | Duration |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0 (Pre-train)** | 98.8 / 99.9 / 100.0 | 98.7 / 99.8 / 99.9 | 98.75 | - | - |
| **1/3** | 96.3 / 99.8 / 99.9 | 95.9 / 99.6 / 99.9 | 96.10 | 0.0107 | 7.4 min |
| **2/3** | 98.4 / 99.9 / 99.9 | 96.5 / 99.8 / 99.9 | 97.45 | 0.0118 | 7.4 min |
| **3/3** | 98.0 / 99.9 / 99.9 | 98.2 / 99.9 / 99.9 | 98.10 | 0.0063 | 10.0 min |

---

## 📈 Step-by-Step Training Logs

### Epoch 1/3
| Step | Progress | Loss | I2T Acc | T2I Acc | LR | Scale | GNorm | Img/s | Mem | Elapsed | ETA |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 10/156 | 6% | 0.0028 | 1.00 | 1.00 | 1.10e-06 | 100.0 | 1.46 | 42.8 | 25.4GB | 0:00:29 | 0:22:51 |
| 20/156 | 13% | 0.0033 | 1.00 | 1.00 | 2.10e-06 | 100.0 | 0.12 | 46.1 | 25.4GB | 0:00:57 | 0:21:32 |
| 30/156 | 19% | 0.0072 | 1.00 | 1.00 | 3.10e-06 | 100.0 | 3.80 | 46.1 | 25.4GB | 0:01:25 | 0:20:47 |
| 40/156 | 26% | 0.0013 | 1.00 | 1.00 | 4.10e-06 | 100.0 | 0.25 | 46.1 | 25.4GB | 0:01:53 | 0:20:11 |
| 50/156 | 32% | 0.0067 | 1.00 | 1.00 | 5.10e-06 | 100.0 | 0.01 | 46.1 | 25.4GB | 0:02:20 | 0:19:38 |
| 60/156 | 38% | 0.0046 | 1.00 | 1.00 | 6.10e-06 | 100.0 | 3.75 | 46.1 | 25.4GB | 0:02:48 | 0:19:07 |
| 70/156 | 45% | 0.0067 | 1.00 | 1.00 | 7.10e-06 | 100.0 | 0.24 | 46.1 | 25.4GB | 0:03:16 | 0:18:36 |
| 80/156 | 51% | 0.0095 | 1.00 | 0.99 | 8.10e-06 | 100.0 | 2.93 | 46.1 | 25.4GB | 0:03:44 | 0:18:07 |
| 90/156 | 58% | 0.0125 | 0.99 | 1.00 | 9.10e-06 | 100.0 | 0.10 | 46.1 | 25.4GB | 0:04:11 | 0:17:38 |
| 100/156 | 64% | 0.0158 | 1.00 | 0.99 | 1.00e-05 | 100.0 | 4.71 | 46.1 | 25.4GB | 0:04:39 | 0:17:09 |
| 110/156 | 71% | 0.0198 | 1.00 | 0.99 | 9.98e-06 | 100.0 | 4.33 | 46.1 | 25.4GB | 0:05:07 | 0:16:40 |
| 120/156 | 77% | 0.0139 | 1.00 | 0.99 | 9.93e-06 | 100.0 | 2.84 | 46.1 | 25.4GB | 0:05:35 | 0:16:12 |
| 130/156 | 83% | 0.0223 | 1.00 | 0.99 | 9.84e-06 | 100.0 | 6.16 | 46.1 | 25.4GB | 0:06:02 | 0:15:43 |
| 140/156 | 90% | 0.0164 | 1.00 | 0.99 | 9.71e-06 | 100.0 | 0.19 | 46.1 | 25.4GB | 0:06:30 | 0:15:15 |
| 150/156 | 96% | 0.0137 | 1.00 | 0.99 | 9.55e-06 | 100.0 | 2.11 | 46.1 | 25.4GB | 0:06:58 | 0:14:47 |
| 156/156 | 100% | 0.0161 | 1.00 | 0.99 | 9.44e-06 | 100.0 | 0.75 | 46.1 | 25.4GB | 0:07:15 | 0:14:30 |

### Epoch 2/3
| Step | Progress | Loss | I2T Acc | T2I Acc | LR | Scale | GNorm | Img/s | Mem | Elapsed | ETA |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 10/156 | 6% | 0.0164 | 1.00 | 0.99 | 9.23e-06 | 100.0 | 0.17 | 44.2 | 25.4GB | 0:08:10 | 0:14:52 |
| 20/156 | 13% | 0.0140 | 1.00 | 0.99 | 8.98e-06 | 99.9 | 6.50 | 46.1 | 25.4GB | 0:08:38 | 0:14:19 |
| 30/156 | 19% | 0.0063 | 1.00 | 1.00 | 8.71e-06 | 99.9 | 4.62 | 46.1 | 25.4GB | 0:09:05 | 0:13:47 |
| 40/156 | 26% | 0.0189 | 1.00 | 0.99 | 8.41e-06 | 99.9 | 5.71 | 46.1 | 25.4GB | 0:09:33 | 0:13:16 |
| 50/156 | 32% | 0.0122 | 1.00 | 0.99 | 8.09e-06 | 99.9 | 2.93 | 46.1 | 25.4GB | 0:10:01 | 0:12:44 |
| 60/156 | 38% | 0.0094 | 1.00 | 0.99 | 7.74e-06 | 99.9 | 6.23 | 46.1 | 25.4GB | 0:10:29 | 0:12:13 |
| 70/156 | 45% | 0.0157 | 1.00 | 0.99 | 7.38e-06 | 99.9 | 1.82 | 46.1 | 25.4GB | 0:10:56 | 0:11:43 |
| 80/156 | 51% | 0.0215 | 1.00 | 0.99 | 6.99e-06 | 99.9 | 4.12 | 46.1 | 25.4GB | 0:11:24 | 0:11:13 |
| 90/156 | 58% | 0.0071 | 1.00 | 1.00 | 6.59e-06 | 99.9 | 5.97 | 46.1 | 25.4GB | 0:11:52 | 0:10:42 |
| 100/156 | 64% | 0.0072 | 1.00 | 1.00 | 6.18e-06 | 99.9 | 1.03 | 46.1 | 25.4GB | 0:12:20 | 0:10:12 |
| 110/156 | 71% | 0.0098 | 1.00 | 0.99 | 5.77e-06 | 99.9 | 4.44 | 46.1 | 25.4GB | 0:12:47 | 0:09:43 |
| 120/156 | 77% | 0.0120 | 1.00 | 1.00 | 5.34e-06 | 99.9 | 5.83 | 46.1 | 25.4GB | 0:13:15 | 0:09:13 |
| 130/156 | 83% | 0.0108 | 1.00 | 0.99 | 4.91e-06 | 99.9 | 3.22 | 46.1 | 25.4GB | 0:13:43 | 0:08:43 |
| 140/156 | 90% | 0.0055 | 1.00 | 1.00 | 4.49e-06 | 99.9 | 0.41 | 46.1 | 25.4GB | 0:14:11 | 0:08:14 |
| 150/156 | 96% | 0.0056 | 1.00 | 1.00 | 4.07e-06 | 99.9 | 0.10 | 46.1 | 25.4GB | 0:14:38 | 0:07:45 |
| 156/156 | 100% | 0.0184 | 1.00 | 0.99 | 3.82e-06 | 99.9 | 3.82 | 46.1 | 25.4GB | 0:14:55 | 0:07:27 |

### Epoch 3/3
| Step | Progress | Loss | I2T Acc | T2I Acc | LR | Scale | GNorm | Img/s | Mem | Elapsed | ETA |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 10/156 | 6% | 0.0070 | 1.00 | 0.99 | 3.41e-06 | 99.9 | 0.52 | 44.2 | 25.4GB | 0:15:50 | 0:07:11 |
| 20/156 | 13% | 0.0065 | 1.00 | 1.00 | 3.01e-06 | 99.9 | 0.72 | 46.1 | 25.4GB | 0:16:18 | 0:06:40 |
| 30/156 | 19% | 0.0167 | 1.00 | 0.99 | 2.62e-06 | 99.9 | 2.53 | 46.1 | 25.4GB | 0:19:22 | 0:07:08 |
| 40/156 | 26% | 0.0066 | 1.00 | 1.00 | 2.26e-06 | 99.9 | 2.28 | 46.1 | 25.4GB | 0:19:49 | 0:06:32 |
| 50/156 | 32% | 0.0076 | 1.00 | 1.00 | 1.91e-06 | 99.9 | 1.55 | 46.1 | 25.4GB | 0:20:17 | 0:05:56 |
| 60/156 | 38% | 0.0079 | 1.00 | 1.00 | 1.59e-06 | 99.9 | 0.07 | 46.1 | 25.4GB | 0:20:45 | 0:05:21 |
| 70/156 | 45% | 0.0089 | 1.00 | 1.00 | 1.29e-06 | 99.9 | 2.31 | 46.1 | 25.4GB | 0:21:13 | 0:04:46 |
| 80/156 | 51% | 0.0043 | 1.00 | 1.00 | 1.02e-06 | 99.9 | 3.46 | 46.1 | 25.4GB | 0:21:41 | 0:04:12 |
| 90/156 | 58% | 0.0012 | 1.00 | 1.00 | 7.73e-07 | 99.9 | 0.09 | 46.1 | 25.4GB | 0:22:08 | 0:03:38 |
| 100/156 | 64% | 0.0057 | 1.00 | 1.00 | 5.61e-07 | 99.9 | 0.29 | 46.1 | 25.4GB | 0:22:36 | 0:03:04 |
| 110/156 | 71% | 0.0039 | 1.00 | 1.00 | 3.81e-07 | 99.9 | 0.86 | 46.1 | 25.4GB | 0:23:04 | 0:02:30 |
| 120/156 | 77% | 0.0108 | 1.00 | 0.99 | 2.34e-07 | 99.9 | 3.35 | 46.1 | 25.4GB | 0:23:32 | 0:01:57 |
| 130/156 | 83% | 0.0035 | 1.00 | 1.00 | 1.23e-07 | 99.9 | 0.82 | 46.1 | 25.4GB | 0:23:59 | 0:01:24 |
| 140/156 | 90% | 0.0055 | 1.00 | 1.00 | 4.66e-08 | 99.9 | 3.83 | 46.1 | 25.4GB | 0:24:27 | 0:00:51 |
| 150/156 | 96% | 0.0012 | 1.00 | 1.00 | 6.56e-09 | 99.9 | 0.08 | 46.1 | 25.4GB | 0:24:55 | 0:00:19 |
| 156/156 | 100% | 0.0019 | 1.00 | 1.00 | 0.00e+00 | 99.9 | 0.42 | 46.1 | 25.4GB | 0:25:11 | 0:00:00 |

## Learning Carve
![Learning carve]()

![Check]()

## Compare with the paper
```
 
| Method | SG4V I2T R@1 | SG4V T2I R@1 |
|---|---:|---:|
| Paper: CLIP ViT-L/14-336 | 86.2 | 84.0 |
| Paper: CLIP ViT-L/14-336 +LLM2CLIP-15M | 98.1 | 98.4 |
| Paper: CLIP ViT-L/14-336 +LLM2CLIP-60M | 98.9 | 99.1 |
| Paper: EVA02 ViT-L/14-336 | 91.5 | 89.4 |
| Paper: EVA02 ViT-L/14-336 +LLM2CLIP-15M | 98.8 | 99.2 |
| Paper: EVA02 ViT-L/14-336 +LLM2CLIP-60M | 98.8 | 99.2 |
| Paper: SigLIP ViT-L-384 | 90.1 | 90.4 |
| **Ours: start (llm2clip, no fine-tune)** | **98.8** | **98.7** |
| *Δ vs paper EVA02 +LLM2CLIP-60M* | +0.0 | -0.5 |

R@1 (%) on SG4V validation = first 1000 COCO entries of `share-captioner_coco_lcs_sam_1246k_1107.json`. The paper's SG4V-1K is the first 1000 entries of the same file (0/1000 are COCO; the rest are LCS/SAM images not downloaded here).  
Text encoder: `microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned`; the paper's rows use its Llama-3-8B text encoder.  
Training: 20,000 COCO images (validation images excluded).



Method
SG4V I2T R@1
SG4V T2I R@1



Paper: CLIP ViT-L/14-336
86.2
84.0


Paper: CLIP ViT-L/14-336 +LLM2CLIP-15M
98.1
98.4


Paper: CLIP ViT-L/14-336 +LLM2CLIP-60M
98.9
99.1


Paper: EVA02 ViT-L/14-336
91.5
89.4


Paper: EVA02 ViT-L/14-336 +LLM2CLIP-15M
98.8
99.2


Paper: EVA02 ViT-L/14-336 +LLM2CLIP-60M
98.8
99.2


Paper: SigLIP ViT-L-384
90.1
90.4


Ours: start (llm2clip, no fine-tune)
98.8
98.7


Δ vs paper EVA02 +LLM2CLIP-60M
+0.0
-0.5


R@1 (%) on SG4V validation = first 1000 COCO entries of share-captioner_coco_lcs_sam_1246k_1107.json. The paper's SG4V-1K is the first 1000 entries of the same file (0/1000 are COCO; the rest are LCS/SAM images not downloaded here).Text encoder: microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned; the paper's rows use its Llama-3-8B text encoder.Training: 20,000 COCO images (validation images excluded).952
```
| Method | SG4V I2T R@1 | SG4V T2I R@1 |
| :--- | :---: | :---: |
| Paper: CLIP ViT-L/14-336 | 86.2 | 84.0 |
| Paper: CLIP ViT-L/14-336 +LLM2CLIP-15M | 98.1 | 98.4 |
| Paper: CLIP ViT-L/14-336 +LLM2CLIP-60M | 98.9 | 99.1 |
| Paper: EVA02 ViT-L/14-336 | 91.5 | 89.4 |
| Paper: EVA02 ViT-L/14-336 +LLM2CLIP-15M | 98.8 | 99.2 |
| Paper: EVA02 ViT-L/14-336 +LLM2CLIP-60M | 98.8 | 99.2 |
| Paper: SigLIP ViT-L-384 | 90.1 | 90.4 |
| **Ours: start (llm2clip, no fine-tune)** | **98.8** | **98.7** |
| *Δ vs paper EVA02 +LLM2CLIP-60M* | +0.0 | -0.5 |

R@1 (%) on SG4V validation = first 1000 COCO entries of `share-captioner_coco_lcs_sam_1246k_1107.json`. The paper's SG4V-1K is the first 1000 entries of the same file (0/1000 are COCO; the rest are LCS/SAM images not downloaded here).  
Text encoder: `microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned`; the paper's rows use its Llama-3-8B text encoder.  
Training: 20,000 COCO images (validation images excluded).