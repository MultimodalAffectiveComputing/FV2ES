import os
import glob
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data.dataset import Dataset
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from scipy.io import wavfile

def read_video(file_name):
    vidcap = cv2.VideoCapture(file_name)

    # Read FPS
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver)  < 3 :
        fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
    else :
        fps = vidcap.get(cv2.CAP_PROP_FPS)

    # Read image data
    success, image = vidcap.read()
    images = []
    while success:
        images.append(image)
        success, image = vidcap.read()
    return np.stack(images), fps

def parse_evaluation_transcript(eval_lines, transcript_lines):
    metadata = {}

    # Parse Evaluation
    for line in eval_lines:
        if line.startswith('['):
            tokens = line.strip().split('\t')
            time_tokens = tokens[0][1:-1].split(' ')
            start_time, end_time = float(time_tokens[0]), float(time_tokens[2])
            uttr_id, label = tokens[1], tokens[2]
            if label=='ang'or label=='exc'or label=='fru'or label=='hap'or label=='neu'or label=='sad':## if label in 5 emotions
                metadata[uttr_id] = {'start_time': start_time, 'end_time': end_time, 'label': label}

    # Parse Transcript
    trans = []
    for line in transcript_lines:
        tokens = line.split(':')
        uttr_id = tokens[0].split(' ')[0]
        if '_' not in uttr_id:
            continue
        text = tokens[-1].strip()
        try:
            metadata[uttr_id]['text'] = text
        except KeyError:
            print(f'with unsure label: {uttr_id}')#show clips with unsure labels
    return metadata

def retrieve_audio(signal, sr, start_time, end_time):
    start_idx = int(sr * start_time)
    end_idx = int(sr * end_time)
    audio_segment = signal[:,start_idx:end_idx]
    return audio_segment, sr

def retrieve_video(frames, fps, start_time, end_time):
    start_idx = int(fps * start_time)
    end_idx = int(fps * end_time)
    images = frames[start_idx:end_idx,:,:,:]
    return images, fps

def dump_image_audio(uttr_id, audio_segment, sr, img_segment, img_segment_L, img_segment_R, fps, out_path='./', grayscale=False):
    out_path = f'{out_path}/{"_".join(uttr_id.split("_")[:2])}'
    if not os.path.exists(f'./{out_path}/{uttr_id}'):
        os.makedirs(f'./{out_path}/{uttr_id}')
    wavfile.write(f'./{out_path}/{uttr_id}/audio.wav', sr, audio_segment)
    wavfile.write(f'./{out_path}/{uttr_id}/audio_L.wav', sr, audio_segment[:,0])
    wavfile.write(f'./{out_path}/{uttr_id}/audio_R.wav', sr, audio_segment[:,1])
    for i in range(img_segment.shape[0]):
        #         cv2.imwrite(f'./{out_path}/{uttr_id}/image_{i}.jpg', img_segment[i,:,:,:])
        imgL = img_segment_L[i,:,:,:]
        imgR = img_segment_R[i,:,:,:]
        if grayscale:
            imgL = rgb2gray(imgL)
            imgR = rgb2gray(imgR)
        cv2.imwrite(f'./{out_path}/{uttr_id}/image_L_{i}.jpg', imgL)
        cv2.imwrite(f'./{out_path}/{uttr_id}/image_R_{i}.jpg', imgR)

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def crop(imgs, target_size=224):
    # imgs.shape = (180, 480, 360, 3)
    _, h, w, _ = imgs.shape
    offset_h = (h - target_size) // 2
    offset_w = (w - target_size) // 2
    imgs = imgs[:, offset_h:-offset_h, offset_w:-offset_w, :]
    return imgs

#preprocessing of the origional dataset
all_metas = {}

base_path='../System/video'

avi_fname =f'{base_path}/Ses01F_impro02.avi'
wav_fname=f'{base_path}/Ses01F_impro02.wav'
script_fname=f'{base_path}/Ses01F_impro02_text.txt'
eval_fname=f'{base_path}/Ses01F_impro02_label.txt'

eval_lines = open(eval_fname).readlines()

transcript_lines = open(script_fname).readlines()

signal,sr=torchaudio.load(wav_fname)

images, fps = read_video(avi_fname)

metas = parse_evaluation_transcript(eval_lines, transcript_lines)

for uttr_id, metadata in metas.items():
    audio_segment, sr = retrieve_audio(signal, sr, metadata['start_time'], metadata['end_time'])
    metadata['sr'] = sr
    img_segment, fps = retrieve_video(images, fps, metadata['start_time'], metadata['end_time'])
    metadata['fps'] = fps
    all_metas.update(metas)

audio_feature_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762]

def getEmotionDict() -> Dict[str, int]:
    return {'ang': 0, 'exc': 1, 'fru': 2, 'hap': 3, 'neu': 4, 'sad': 5}

def get_dataset_iemocap(data_folder: str, phase: str, img_interval: int, hand_crafted_features: Optional[bool] = False):
    meta = all_metas

    emoDict = getEmotionDict()
    uttr_ids = [keyname for keyname in meta]
    texts = [meta[uttr_id]['text'] for uttr_id in uttr_ids]
    labels = [emoDict[meta[uttr_id]['label']] for uttr_id in uttr_ids]

    this_dataset = IEMOCAP(
        main_folder=data_folder,
        utterance_ids=uttr_ids,
        texts=texts,
        labels=labels,
        label_annotations=list(emoDict.keys()),
        img_interval=img_interval
        )

    return this_dataset


class IEMOCAP(Dataset):
    def __init__(self, main_folder: str, utterance_ids: List[str], texts: List[List[int]], labels: List[int],
                 label_annotations: List[str], img_interval: int):
        super(IEMOCAP, self).__init__()
        self.utterance_ids = utterance_ids
        self.texts = texts
        self.labels = F.one_hot((torch.tensor(labels)),num_classes=6).numpy() #add number_classes=6
        self.label_annotations = label_annotations

        self.utteranceFolders = {
            folder.split('/')[-1]: folder
            for folder in glob.glob(os.path.join(main_folder, '**/*'))
        }
        self.img_interval = img_interval

    def get_annotations(self) -> List[str]:
        return self.label_annotations

    def use_left(self, utteranceFolder: str) -> bool:
        entries = utteranceFolder.split('_')
        a=entries[0][-1]
        b=entries[-1][0]
        return a == b


    def cutWavToPieces(self, waveform, sampleRate):
        # Split the audio waveform by second
        total = int(np.ceil(waveform.size(-1) / sampleRate))
        waveformPieces = []
        for i in range(total):
            waveformPieces.append(waveform[:, i * sampleRate:(i + 1) * sampleRate])

        # Pad the last piece
        lastPieceLength = waveformPieces[-1].size(-1)
        if lastPieceLength < sampleRate:
            padLeft = (sampleRate - lastPieceLength) // 2
            padRight = sampleRate - lastPieceLength - padLeft
            waveformPieces[-1] = F.pad(waveformPieces[-1], (padLeft, padRight))
        return waveformPieces

    def cutSpecToPieces(self, spec, stride=32):
        # Split the audio waveform by second
        total = -(-spec.size(-1) // stride)
        specs = []
        for i in range(total):
            specs.append(spec[:, :, :, i * stride:(i + 1) * stride])

        # Pad the last piece
        lastPieceLength = specs[-1].size(-1)
        if lastPieceLength < stride:
            padRight = stride - lastPieceLength
            specs[-1] = F.pad(specs[-1], (0, padRight))
        return specs

    def getPosWeight(self):
        pos_nums = self.labels.sum(axis=0)
        neg_nums = self.__len__() - pos_nums
        pos_weight = neg_nums / pos_nums
        return pos_weight

    def __len__(self):
        return len(self.utterance_ids)

    def __getitem__(self, ind: int) -> Tuple[str, np.array, List[torch.tensor], List[int], np.array]:
        uttrId = self.utterance_ids[ind]
        metadata=all_metas[uttrId]
        use_left = self.use_left(uttrId)
        suffix = 'L' if use_left else 'R'

        audio_segment, sr = retrieve_audio(signal, 16000, metadata['start_time'], metadata['end_time'])
        img_segment, fps = retrieve_video(images, 29.97, metadata['start_time'], metadata['end_time'])
        img_segment_L, img_segment_R = img_segment[:,:,:img_segment.shape[2] // 2,:], img_segment[:,:,img_segment.shape[2] // 2:,:]
        img_segment_L = crop(img_segment_L)
        img_segment_R = crop(img_segment_R)
        nums=len(img_segment)
        step = int(self.img_interval / 1000 * fps)+1
        if suffix == 'L':
            sampledImgs=np.array([np.float32(img_segment_L[i]) for i in list(range(0,nums,step)) ])
            waveform=audio_segment[0] #left audio channel
        else:
            sampledImgs=np.array([np.float32(img_segment_R[i]) for i in list(range(0,nums,step)) ])
            waveform=audio_segment[1]
        waveform=waveform.unsqueeze(0)#add one dim
        specgram = torchaudio.transforms.MelSpectrogram(sample_rate=sr, win_length=int(float(sr) / 16000 * 400))(waveform).unsqueeze(0)
        specgrams = self.cutSpecToPieces(specgram)

        return uttrId, sampledImgs, specgrams, self.texts[ind], self.labels[ind]

def collate_fn(batch):
    utterance_ids = []
    texts = []
    labels = []

    newSampledImgs = None
    imgSeqLens = []

    specgrams = []
    specgramSeqLens = []

    for dp in batch:
        utteranceId, sampledImgs, specgram, text, label = dp
        if sampledImgs.shape[0] == 0:
            continue
        utterance_ids.append(utteranceId)
        texts.append(text)
        labels.append(label)

        imgSeqLens.append(sampledImgs.shape[0])
        newSampledImgs = sampledImgs if newSampledImgs is None else np.concatenate((newSampledImgs, sampledImgs), axis=0)

        specgramSeqLens.append(len(specgram))
        specgrams.append(torch.cat(specgram, dim=0))

    imgs = newSampledImgs
    # print(utterance_ids)  ## by ling

    return (
        utterance_ids,
        imgs,
        imgSeqLens,
        torch.cat(specgrams, dim=0),
        specgramSeqLens,
        texts,
        torch.tensor(labels, dtype=torch.float32)
    )




