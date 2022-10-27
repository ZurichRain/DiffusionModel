import torch
from transformers import BertTokenizer,BertTokenizerFast

if __name__ == '__main__':
    # a=torch.linspace(1e-4, 0.2, 10)
    # b = 1.-a
    # b_hat = torch.cumprod(b,dim=0)
    # print(b_hat)
    # print(torch.sqrt(b_hat[0]))
    # t = torch.randint(low=1, high=10, size=(3,))
    # print(b_hat[t])
    # print(torch.sqrt(b_hat[t]))
    # c = torch.sqrt(b_hat[t])[:,None, None] # n,m
    # print(c)
    # print(c.size())
    # x = torch.randn((3,3,5))
    # print((c * x).size())
    # print(a)
    # Ɛ = torch.randn_like(a)
    # print(Ɛ)
    # print(torch.randint(low=1, high=10, size=(3,)))
    bert_pathname = '/opt/data/private/sxu/fwang/transformers_model/bert-base-uncased/'
    tokenizer = BertTokenizerFast.from_pretrained(bert_pathname)
    tok_lis = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('I love you'))
    print(tok_lis)
    print(tokenizer.vocab_size)
