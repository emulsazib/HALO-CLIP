# Train 1
## Load Dataset from Hugging Face 
 Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
README.md: 100% 2.23k/2.23k [00:00<00:00, 269kB/s]data/test-00000-of-00009.parquet: reconstructing file: 100%  471MB /  471MB, 43.0MB/s  data/test-00000-of-00009.parquet: downloading bytes:   471MB, 42.1MB/s  data/test-00001-of-00009.parquet: reconstructing file: 100%  475MB /  475MB, 44.9MB/s  data/test-00001-of-00009.parquet: downloading bytes:   475MB, 44.1MB/s  data/test-00002-of-00009.parquet: reconstructing file: 100%  486MB /  486MB, 20.4MB/s  data/test-00002-of-00009.parquet: downloading bytes:   485MB, 11.4MB/s  data/test-00003-of-00009.parquet: reconstructing file: 100%  472MB /  472MB, 44.9MB/s  data/test-00003-of-00009.parquet: downloading bytes:   472MB, 44.4MB/s  data/test-00004-of-00009.parquet: reconstructing file: 100%  490MB /  490MB, 46.5MB/s  data/test-00004-of-00009.parquet: downloading bytes:   489MB, 45.3MB/s  data/test-00005-of-00009.parquet: reconstructing file: 100%  500MB /  500MB, 45.1MB/s  data/test-00005-of-00009.parquet: downloading bytes:   496MB, 42.8MB/s  data/test-00006-of-00009.parquet: reconstructing file: 100%  528MB /  528MB, 50.0MB/s  data/test-00006-of-00009.parquet: downloading bytes:   522MB, 48.5MB/s  data/test-00007-of-00009.parquet: reconstructing file: 100%  511MB /  511MB, 48.5MB/s  data/test-00007-of-00009.parquet: downloading bytes:   506MB, 47.5MB/s  data/test-00008-of-00009.parquet: reconstructing file: 100%  477MB /  477MB, 45.0MB/s  data/test-00008-of-00009.parquet: downloading bytes:   477MB, 43.8MB s  
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
 config.json: 100% 1.02k/1.02k [00:00<00:00, 102kB/s]tokenizer_config.json: 100% 54.6k/54.6k [00:00<00:00, 5.74MB/s]tokenizer.json: 100% 9.09M/9.09M [00:00<00:00, 133MB/s]special_tokens_map.json: 100% 325/325 [00:00<00:00, 41.6kB/s]model.safetensors: reconstructing file: 100% 4.94GB / 4.94GB,  283MB/s  model.safetensors: downloading bytes:  4.15GB,  307MB/s  Loading weights: 100% 146/146 [00:00<00:00, 571.81it/s]LLM caption embeddings: 100% 1203/1203 [01:13<00:00, 24.50it/s]LLM caption embeddings: 100% 40/40 [00:02<00:00, 20.73it/s]Embedded 158,915 captions in 76s
train_emb (153915, 2048) | val_emb (5000, 2048)

## Build the CLIP model (repo's eva_clip) and load weights from Hugging Face
 LLM2CLIP-Llama3.2-1B-EVA02-L-14-336.pt: reconstructing file: 100%  750MB /  750MB, 65.6MB/s  LLM2CLIP-Llama3.2-1B-EVA02-L-14-336.pt: downloading bytes:   750MB, 61.4MB/s  Loaded all weights from LLM2CLIP-Llama3.2-1B-EVA02-L-14-336.pt
Llama3.2-1B-EVA02-L-14-336: 375M params, 375M trainable (vision 305M, adapter 70M)

## Training curves
![Training carve](https://github.com/emulsazib/HALO-CLIP/blob/main/LLM2CLIP_Output/Results/training_carve.png?raw=true)


epoch 0 | i2t R@1/5/10 94.7/99.5/99.8 | t2i R@1/5/10 81.5/95.5/97.7 | mean R@1 88.12 (before training)
- ★ new best → /content/drive/MyDrive/llm2clip_runs/llm2clip_flickr30k/best_model.pt
epoch 1/3: 100% 240/240 [11:05<00:00,  2.75s/it, i2t=0.94, loss=0.184, scale=99.9]epoch 1 | i2t R@1/5/10 93.2/99.3/99.8 | t2i R@1/5/10 81.7/96.1/98.0 | mean R@1 87.47 | train loss 0.161 | 11.2 min
epoch 2/3: 100% 240/240 [11:04<00:00,  2.75s/it, i2t=0.98, loss=0.079, scale=99.9]epoch 2 | i2t R@1/5/10 95.3/99.5/99.7 | t2i R@1/5/10 83.5/96.7/98.2 | mean R@1 89.42 | train loss 0.138 | 11.2 min
- ★ new best → /content/drive/MyDrive/llm2clip_runs/llm2clip_flickr30k/best_model.pt
epoch 3/3: 100% 240/240 [11:04<00:00,  2.75s/it, i2t=0.98, loss=0.064, scale=99.9]epoch 3 | i2t R@1/5/10 95.2/99.4/99.7 | t2i R@1/5/10 84.5/97.0/98.5 | mean R@1 89.87 | train loss 0.101 | 11.2 min
- ★ new best → /content/drive/MyDrive/llm2clip_runs/llm2clip_flickr30k/best_model.pt

**Best: epoch 3 | mean R@1 89.87 | rsum 574.3**

## Sanity check: retrieve captions for val images with the best weights
![Sanity Check](https://github.com/emulsazib/HALO-CLIP/blob/main/LLM2CLIP_Output/Results/sanity_check.png?raw=true)
--- 
# Train 2