import torch
from tqdm import tqdm
from src.trainers.basetrainer import TrainerBase
from transformers import AlbertTokenizer
import numpy as np

class IemocapTrainer(TrainerBase):
    def __init__(self, args, model, criterion, optimizer, scheduler, device, dataloaders):
        super(IemocapTrainer, self).__init__(args, model, criterion, optimizer, scheduler, device, dataloaders)

        self.args = args
        self.text_max_len = args['text_max_len']
        # self.tokenizer = AlbertTokenizer.from_pretrained(f'albert-{args["text_model_size"]}-v2')
        self.tokenizer = AlbertTokenizer.from_pretrained('./src/models/albert-base-v2')
        self.all_test_stats = []
        annotations = dataloaders['test'].dataset.get_annotations()
        self.best_epoch = -1

    def test(self):
        test_stats = self.eval_one_epoch('test')

    def eval_one_epoch(self, phase='valid', thresholds=None):

        for m in self.model.modules():
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()  # turn to deploy every modules
        self.model.eval()
        dataloader = self.dataloaders[phase]

        data_size = 0
        total_logits = []
        total_Y = []
        pbar = tqdm(dataloader, desc=phase)

        for uttranceId, imgs, imgLens, specgrams, specgramLens, text, Y in pbar:
            text = self.tokenizer(text, return_tensors='pt', max_length=self.text_max_len, padding='max_length', truncation=True)

            # imgs = imgs.to(device=self.device)
            specgrams = specgrams.to(device=self.device)
            text = text.to(device=self.device)
            Y = Y.to(device=self.device)

            with torch.set_grad_enabled(False):
                logits = self.model(imgs, imgLens, specgrams, specgramLens, text) # (batch_size, num_classes)
                data_size += Y.size(0)

            total_logits.append(logits.cpu())
            total_Y.append(Y.cpu())


        total_logits = torch.cat(total_logits, dim=0)
        total_Y = torch.cat(total_Y, dim=0)
        preds=torch.sigmoid(total_logits)
        mean_preds=torch.mean(preds,dim=0)
        print('six emotional values for one video:'+'\n')
        print(mean_preds)

        with open("result.txt", 'a') as f:
            mean = np.array(mean_preds)
            for i in range(len(mean)):
                f.write(str(mean[i]))
                f.write('\n')
        print("write txt finish!")## save result.txt

        return total_logits, total_Y
