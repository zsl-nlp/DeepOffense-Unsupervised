# DeepOffense-Unsupervised


## 1.代码结构

    │  arbic.py
    │  danish.py
    │  english.py
    │  greek.py
    │  img.png
    │  model.png
    │  model.py
    │  readme
    │  README.md
    │  run.sh
    │  turkish.py
    │
    └─UnsupervisedMT
        │  .gitignore
        │  CODE_OF_CONDUCT.md
        │  CONTRIBUTING.md
        │  illustration.png
        │  LICENSE
        │  README.md
        │
        ├─NMT
        │  │  get_data_deen.sh
        │  │  get_data_enfr.sh
        │  │  main.py
        │  │  preprocess.py
        │  │
        │  └─src
        │      │  adam_inverse_sqrt_with_warmup.py
        │      │  evaluator.py
        │      │  fairseq_utils.py
        │      │  gumbel.py
        │      │  logger.py
        │      │  multiprocessing_event_loop.py
        │      │  sequence_generator.py
        │      │  test.py
        │      │  trainer.py
        │      │  utils.py
        │      │  __init__.py
        │      │
        │      ├─data
        │      │      dataset.py
        │      │      dictionary.py
        │      │      loader.py
        │      │      __init__.py
        │      │
        │      ├─model
        │      │      attention.py
        │      │      discriminator.py
        │      │      lm.py
        │      │      pretrain_embeddings.py
        │      │      seq2seq.py
        │      │      transformer.py
        │      │      __init__.py
        │      │
        │      └─modules
        │              label_smoothed_cross_entropy.py
        │              layer_norm.py
        │              multihead_attention.py
        │              sinusoidal_positional_embedding.py
        │              __init__.py
        │
        └─PBSMT
            │  create-phrase-table.py
            │  run.sh
            │
            └─src
                    dictionary.py
                    loader.py
                    utils.py
                    __init__.py
                    `

## 2.依赖库
    
    bert4keras==0.9.8
     h5py==2.10.0 
     Keras==2.3.1
     tensorflow-gpu==1.14.0
     tqdm==4.54.1
   

## 3.运行

   
    run run.sh
    
=======
