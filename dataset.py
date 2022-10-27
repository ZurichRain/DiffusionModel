from symbol import pass_stmt
from torch.utils.data import Dataset
from transformers import BertTokenizer,BertTokenizerFast
import torch


class CurDataset(Dataset):
    bert_pathname = '/opt/data/private/sxu/fwang/transformers_model/bert-base-uncased/'
    def __init__(self, seq_data_lis):
        self.seq_data_lis = seq_data_lis
        self.tokenizer = BertTokenizerFast.from_pretrained(self.bert_pathname)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.dataset = self.preprocess()

    def preprocess(self):
        self.get_seq_tok()
        input_data=[]
        for data in self.seq_data_lis:
            cur_dict=dict()
            cur_dict['seq'] = data['onehot_tok']
            input_data.append(cur_dict)
        return input_data

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        """get dataset size"""
        return len(self.dataset)

    def get_seq_tok(self):
        for seq_data in self.seq_data_lis:
            dt = seq_data['text']
            dtok=self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(dt))
            if(len(dtok)>128):
                seq_data['dtok'] = dtok[:128]
            else:
                seq_data['dtok'] = dtok
                for _ in range(128-len(dtok)):
                    seq_data['dtok'].append(0)
        for seq_data in self.seq_data_lis:
            seq_data['onehot_tok'] = []
            for tokid in seq_data['dtok']:
                c_onehot=[0]*self.tokenizer.vocab_size
                c_onehot[tokid]=1
                seq_data['onehot_tok'].append(c_onehot)


    def collate_fn(self, batch):
        seq_tok = [data['seq'] for data in batch]

        seq_tok = torch.tensor(seq_tok, dtype=torch.float).to(self.device)

        return {
            'seq_tok':seq_tok,
        }
