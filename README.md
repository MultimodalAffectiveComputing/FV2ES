# FV2ES: A Fully End2End Multimodal System for Fast Yet Effective Video Emotion Recognition

![](https://img.shields.io/badge/python-3.6+-blue.svg)

[[Paper]](https://www.aclweb.org/anthology/2021.naacl-main.417/)

**FV2ES: A Fully End2End Multimodal System for Fast Yet Effective Video Emotion Recognition**, by **Qinglan Wei**, Xuling Huang, Yuan Zhang.

## Paper Abstract

> In the latest social networks, more and more people prefer to express their emotions in videos through text, speech, and rich facial expressions. Multimodal video emotion analysis techniques can help understand users’ inner world automatically based on human expressions and gestures in images, tones in voices, and recognized natural language. However, in the existing research, the acoustic modality has long been in a marginal position as compared to visual and textual modalities. That is, it tends to be more difficult to improve the contribution of the acoustic modality for the whole multimodal emotion recognition task. Besides, although better performance can be obtained by introducing common deep learning methods, the complex structures of these training models always result in low inference efficiency, especially when exposed to high-resolution and long-length videos. Moreover, the lack of a fully end-to-end multimodal video emotion recognition system hinders its application. In this paper, we designed a fully multimodal video-to-emotion system (named FV2ES) for fast yet effective recognition inference, whose benefits are threefold: (1) The adoption of the hierarchical attention method upon the sound spectra breaks through the limited contribution of the acoustic modality, and outperforms the existing models’ performance on both IEMOCAP and CMU-MOSEI datasets; (2) the introduction of the idea of multi-scale for visual extraction while single-branch for inference brings higher efficiency and maintains the prediction accuracy at the same time; (3) the further integration of data pre-processing into the aligned multimodal learning model allows the significant reduction of computational costs and storage space.

If you work is inspired by our paper or code, please cite it, thanks!

<pre>
@inproceedings{dai-etal-2021-multimodal,
    title = "FV2ES: A Fully End2End Multimodal System for Fast Yet Effective Video Emotion Recognition",
    author = "Qinglan Wei  and
      Xuling Huang  and
      Yuan Zhang",
    abstract = "In the latest social networks, more and more people prefer to express their emotions in videos through text, speech, and rich facial expressions. Multimodal video emotion analysis techniques can help understand users’ inner world automatically based on human expressions and gestures in images, tones in voices, and recognized natural language. However, in the existing research, the acoustic modality has long been in a marginal position as compared to visual and textual modalities. That is, it tends to be more difficult to improve the contribution of the acoustic modality for the whole multimodal emotion recognition task. Besides, although better performance can be obtained by introducing common deep learning methods, the complex structures of these training models always result in low inference efficiency, especially when exposed to high-resolution and long-length videos. Moreover, the lack of a fully end-to-end multimodal video emotion recognition system hinders its application. In this paper, we designed a fully multimodal video-to-emotion system (named FV2ES) for fast yet effective recognition inference, whose benefits are threefold: (1) The adoption of the hierarchical attention method upon the sound spectra breaks through the limited contribution of the acoustic modality, and outperforms the existing models’ performance on both IEMOCAP and CMU-MOSEI datasets; (2) the introduction of the idea of multi-scale for visual extraction while single-branch for inference brings higher efficiency and maintains the prediction accuracy at the same time; (3) the further integration of data pre-processing into the aligned multimodal learning model allows the significant reduction of computational costs and storage space.",
}
</pre>

## Dataset

As mentioned in our paper, there are two public datasets used in our experiments, including the IEMOCAP and the CMU-MOSEI datasets.  
The IEMOCAP dataset consists of multimodal data of three modalities of video, audio, and text transcription. We select six main categories from the original emotions: anger, happiness, excitement, sadness, frustration, and neutral. And to create a new split for the dataset, we randomly assign 70%, 10%, and 20% of the data to the training, validation, and test sets respectively.  
The CMU-MOSEI dataset also consists of multimodal data of three modalities of vision, audio, and text. Six kinds of labels including happiness, sadness, anger, fear, disgust, and surprise were annotated for the videos. And the dataset contains 250 topics, 3837 videos, 23453 sentences, 1000 narrators, and the total duration reaches 65 hours.  

The raw data can be downloaded from [CMU-MOSEI](http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI.zip) (~120GB) and [IEMOCAP](https://hkustconnect-my.sharepoint.com/:u:/g/personal/wdaiai_connect_ust_hk/EdZoawxqQ01Ej38NpflFZPEB4zYR9RxIPcaAPcFU77qFgQ?e=7efIl0) (~16.5GB). However, for the IEMOCAP, you need to [request for a permission](https://sail.usc.edu/iemocap/iemocap_release.htm) from the original author, then you can be given the passcode to download.

## Preparation

### Dataset

To run our code directly, you can download the processed data from [here](https://hkustconnect-my.sharepoint.com/:u:/g/personal/wdaiai_connect_ust_hk/EbEzVnCduqVNuT_LvuRApVQBagraPx7nGqIEHgdnGPjN7g?e=zeMtct) (88.6G). Unzip it and the tree structure of the data direcotry looks like this:

```
./data
- IEMOCAP_HCF_FEATURES
- IEMOCAP_RAW_PROCESSED
- IEMOCAP_SPLIT
- MOSEI_RAW_PROCESSED
- MOSEI_HCF_FEATURES
- MOSEI_SPLIT
```

### Environment

* Python 3.7.6
* PyTorch 1.8.0
* torchaudio 0.8.0
* torchvision 0.9.0
* transformers 4.17.0
* facenet-pytorch 2.5.2

## Command examples for running

### Train the V2EM

```console
python main.py -lr=4.5e-6 -ep=30 -mod=tav -bs=2 --img-interval=500 --early-stop=6 --loss=bce --cuda=0 --model=mme2e --num-emotions=6 --trans-dim=64 --trans-nlayers=4 --trans-nheads=4 --text-lr-factor=10 --text-model-size=base --text-max-len=100 
```

## CLI

```
usage: main.py [-h] -bs BATCH_SIZE -lr LEARNING_RATE [-wd WEIGHT_DECAY] -ep
               EPOCHS [-es EARLY_STOP] [-cu CUDA] [-cl CLIP] [-sc] [-se SEED]
               [--loss LOSS] [--optim OPTIM] [--text-lr-factor TEXT_LR_FACTOR]
               [-mo MODEL] [--text-model-size TEXT_MODEL_SIZE]
               [--fusion FUSION] [--feature-dim FEATURE_DIM] [-hfcs HFC_SIZES [HFC_SIZES ...]]
               [--trans-dim TRANS_DIM] [--trans-nlayers TRANS_NLAYERS]
               [--trans-nheads TRANS_NHEADS] [-aft AUDIO_FEATURE_TYPE]
               [--num-emotions NUM_EMOTIONS] [--img-interval IMG_INTERVAL]
               [--hand-crafted] [--text-max-len TEXT_MAX_LEN]
               [--datapath DATAPATH] [--dataset DATASET] [-mod MODALITIES]
               [--valid] [--test] [--ckpt CKPT] [--ckpt-mod CKPT_MOD]
               [-dr DROPOUT] [-nl NUM_LAYERS] [-hs HIDDEN_SIZE] [-bi] [--gru]

FV2ES: A Fully End2End Multimodal System for Fast Yet Effective Video Emotion Recognition

optional arguments:
  -h, --help            show this help message and exit
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        Learning rate
  -wd WEIGHT_DECAY, --weight-decay WEIGHT_DECAY
                        Weight decay
  -ep EPOCHS, --epochs EPOCHS
                        Number of epochs
  -es EARLY_STOP, --early-stop EARLY_STOP
                        Early stop
  -cu CUDA, --cuda CUDA
                        Cude device number
  -cl CLIP, --clip CLIP
                        Use clip to gradients
  -sc, --scheduler      Use scheduler to optimizer
  -se SEED, --seed SEED
                        Random seed
  --loss LOSS           loss function
  --optim OPTIM         optimizer function: adam/sgd
  --text-lr-factor TEXT_LR_FACTOR
                        Factor the learning rate of text model
  -mo MODEL, --model MODEL
                        Which model
  --text-model-size TEXT_MODEL_SIZE
                        Size of the pre-trained text model
  --fusion FUSION       How to fuse modalities
  --feature-dim FEATURE_DIM
                        Dimension of features outputed by each modality model
  -hfcs HFC_SIZES [HFC_SIZES ...], --hfc-sizes HFC_SIZES [HFC_SIZES ...]
                        Hand crafted feature sizes
  --trans-dim TRANS_DIM
                        Dimension of the transformer after CNN
  --trans-nlayers TRANS_NLAYERS
                        Number of layers of the transformer after CNN
  --trans-nheads TRANS_NHEADS
                        Number of heads of the transformer after CNN
  -aft AUDIO_FEATURE_TYPE, --audio-feature-type AUDIO_FEATURE_TYPE
                        Hand crafted audio feature types
  --num-emotions NUM_EMOTIONS
                        Number of emotions in data
  --img-interval IMG_INTERVAL
                        Interval to sample image frames
  --hand-crafted        Use hand crafted features
  --text-max-len TEXT_MAX_LEN
                        Max length of text after tokenization
  --datapath DATAPATH   Path of data
  --dataset DATASET     Use which dataset
  -mod MODALITIES, --modalities MODALITIES
                        what modalities to use
  --valid               Only run validation
  --test                Only run test
  --ckpt CKPT           Path of checkpoint
  --ckpt-mod CKPT_MOD   Load which modality of the checkpoint
  -dr DROPOUT, --dropout DROPOUT
                        dropout
  -nl NUM_LAYERS, --num-layers NUM_LAYERS
                        num of layers of LSTM
  -hs HIDDEN_SIZE, --hidden-size HIDDEN_SIZE
                        hidden vector size of LSTM
  -bi, --bidirectional  Use Bi-LSTM
  --gru                 Use GRU rather than LSTM
```
