# code by Tae Hwan Jung(Jeff Jung) @graykode, Derek Miller @dmmiller612
# Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
#           https://github.com/JayParks/transformer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import DataLoader, Sampler, random_split
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from GCN_model import GCN1
from batch_embed2 import get_embed
from adj_matrix1 import get_A
from ADG_order import get_ADG_order
from AST_order import get_AST_order
from make_data_newMG import make_nl_vocab
from make_data_newMG import make_code_vocab
from make_data_newMG import generate_sentences
from make_data_newMG import generate_code_datasets
from make_data_newMG import generate_nl_datasets
from Generate_AST_num import generate_AST_num

Device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps

def make_batch():
    # input_batch, output_batch, target_batch = [],[],[]
    input_batch1, output_batch1, target_batch1 = [], [], []
    for i in range(len(sentences)):
        input_batch = [[src_vocab[n] for n in sentences[i][0].split()]]
        output_batch = [[tgt_vocab[n] for n in sentences[i][1].split()]]
        target_batch = [[tgt_vocab[n] for n in sentences[i][2].split()]]
        # print('00000')

        input_batch1.extend(input_batch)
        output_batch1.extend(output_batch)
        target_batch1.extend(target_batch)
    # print(output_batch)
    # exit()
    # print(input_batch1)
    # print(output_batch1[0])
    # exit()
    print(len(output_batch1[0]))
    print(len(target_batch1[0]))

    return torch.LongTensor(input_batch1), torch.LongTensor(output_batch1), torch.LongTensor(target_batch1)

def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)

def get_attn_pad_mask(seq_q, seq_k):
    # print(seq_q)
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_vocab_size, d_model),freeze=True)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs): # enc_inputs : [batch_size x source_len]
        # enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(torch.LongTensor([[1,2,3,4,5,6,2,7]]))
        # print(enc_inputs)
        # print(enc_inputs.max())
        # print(enc_inputs.min())
        # print(self.src_emb(enc_inputs))
        print(self.src_emb(enc_inputs).shape)
        # print(self.pos_emb(enc_inputs))
        print(self.pos_emb(enc_inputs).shape)
        # exit()
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(enc_inputs)

        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        # self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_len + 1, d_model), freeze=True)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_vocab_size, d_model), freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):  # dec_inputs : [batch_size x target_len]

        # print("这是问题所在的地方吗？", dec_inputs)
        # print(self.tgt_emb(dec_inputs))
        print((self.pos_emb(dec_inputs)).shape)
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(dec_inputs)
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        print(dec_self_attn_pad_mask.shape)
        print(dec_self_attn_subsequent_mask.shape)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask.to(Device)), 0)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)
    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs) # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

def greedy_decoder(model, enc_input, start_symbol):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, tgt_len).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0, tgt_len):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input


def showgraph(attn):
    attn = attn[-1].squeeze(0)[0]
    attn = attn.squeeze(0).data.numpy()
    fig = plt.figure(figsize=(n_heads, n_heads)) # [n_heads, n_heads]
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attn, cmap='viridis')
    ax.set_xticklabels(['']+sentences[0].split(), fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels(['']+sentences[2].split(), fontdict={'fontsize': 14})
    plt.show()




if __name__ == '__main__':
    # sentences = [
    #     ['flush the internal buffer and close the writer  ',
    #              'S CompilationUnit PackageDeclaration testshell NodeList ClassOrInterfaceDeclaration Hello NodeList MethodDeclaration script pydmlfromurl NodeList '
    #              'Parameter scripturlpath string BlockStmt NodeList ReturnStmt MethodCallExpr scriptfromurl NodeList scripturlpath FieldAccessExp scripttype pydml',
    #              'CompilationUnit PackageDeclaration testshell NodeList ClassOrInterfaceDeclaration Hello NodeList MethodDeclaration script pydmlfromurl NodeList '
    #              'Parameter scripturlpath string BlockStmt NodeList ReturnStmt MethodCallExpr scriptfromurl NodeList scripturlpath FieldAccessExp scripttype pydml E'],
    #     ['flush the internal buffer and close and writer  ',
    #      'S CompilationUnit PackageDeclaration testshell NodeList ClassOrInterfaceDeclaration Hello NodeList MethodDeclaration NodeList pydmlfromurl NodeList '
    #      'Parameter scripturlpath string BlockStmt NodeList ReturnStmt MethodCallExpr scriptfromurl NodeList scripturlpath FieldAccessExp scripttype pydml',
    #      'CompilationUnit PackageDeclaration testshell NodeList ClassOrInterfaceDeclaration Hello NodeList MethodDeclaration script pydmlfromurl NodeList '
    #      'Parameter scripturlpath string BlockStmt NodeList ReturnStmt MethodCallExpr scriptfromurl NodeList scripturlpath FieldAccessExp scripttype pydml E']
    # ]

    n1,n2=generate_nl_datasets()
    c1,c2,c3=generate_code_datasets()
    sentences=generate_sentences()
    # print(sentences[0])
    # print(sentences[1])
    # exit()
    # Transformer Parameters
    # Padding Should be Zero index
    # src_vocab = {'p': 0, 'flush': 1, 'the': 2, 'internal': 3, 'buffer': 4, 'and': 5, 'close': 6, 'writer': 7}
    src_vocab=make_nl_vocab()
    # print("11111",src_vocab)
    src_vocab_size = len(src_vocab)

    tgt_vocab =make_code_vocab()
    # print("22222",tgt_vocab)
    print(len(sentences))
    print("自然语言的最大长度为：",n2)
    print(len(c1)+len(c2))
    print("代码的最大长度为：",c3)
    # exit()
    # print(src_vocab['SOS'])
    # print(src_vocab['pad'])
    # print(src_vocab['EOS'])
    #
    # print(tgt_vocab['SOS'])
    # print(tgt_vocab['pad'])
    # print(tgt_vocab['EOS'])
    # exit()
    # print(src_vocab_size)
    # print(n2)
    # exit()
    # print(tgt_vocab)
    number_dict = {i: w for i, w in enumerate(tgt_vocab)}
    tgt_vocab_size = len(tgt_vocab)
    # print(tgt_vocab_size)
    # exit()

    src_len = n2 # length of source
    tgt_len = c3 # length of target

    d_model = 512  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention
    model = Transformer().to(Device)
    # print(11111)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.99)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # print(len(sentences))
    enc_inputs, dec_inputs, target_batch = make_batch()
    # print(dec_inputs)
    # exit()
    # print(len(sentences))
    # print(enc_inputs)
    # print("输出8份")
    # print(enc_inputs[:8])
    # print(len(enc_inputs[:8]))
    t=int(0.8*len(sentences))
    enc_inputs1 = enc_inputs[:t]
    # enc_inputs2 = enc_inputs[8:10]
    dec_inputs1 = dec_inputs[:t]
    # dec_inputs2 = dec_inputs[8:10]
    target_batch1 = target_batch[:t]
    # target_batch2 = target_batch[8:10]
    # print(len(enc_inputs))
    # enc_inputs1,enc_inputs2 = random_split(enc_inputs, [8, 2])
    # dec_inputs1,dec_inputs2 = random_split(dec_inputs, [8, 2])
    # target_batch1,target_batch2 = random_split(target_batch, [8, 2])
    # print(enc_inputs1)
    # print(len(enc_inputs1))
    # print(dec_inputs1)
    # print(len(dec_inputs1))
    # print(target_batch1)
    # print(len(target_batch1))


    loader = Data.DataLoader(MyDataSet(enc_inputs1, dec_inputs1, target_batch1), 8, True)

    # for epoch in range(5):
    #
    #     optimizer.zero_grad()
    #     outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
    #     loss = criterion(outputs, target_batch.contiguous().view(-1))
    #     print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    #     loss.backward(retain_graph=True)
    #     optimizer.step()
    #
    # best_test_loss = float('inf')
    # start_time1 = time.time()
    for epoch in range(300):
        # start_time = time.time()
        # print("该轮数的开始时间",start_time)
        # sum_loss = 0
        for enc_inputs, dec_inputs, target_batch in loader:
            '''
            enc_inputs: [batch_size, src_len]
            dec_inputs: [batch_size, tgt_len]
            target_batch: [batch_size, tgt_len]
            '''
            enc_inputs, dec_inputs, target_batch = enc_inputs.to(Device), dec_inputs.to(Device), target_batch.to(Device)
            optimizer.zero_grad()
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, target_batch.contiguous().view(-1))

            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
            # loss.backward(retain_graph=True)
            # sum_loss=sum_loss+loss

            loss.backward()
            optimizer.step()
        # end_time = time.time()
        # print("该轮训练的总损失为：", sum_loss)
        # if best_test_loss > sum_loss:
        #     best_test_loss = sum_loss
        #
        # print("该轮训练之后的最好损失为：", best_test_loss)

        # print("该轮数的结束时间：",end_time)
    # end_time1 = time.time()
    # spend_time=end_time1-start_time1
    #
    # print("训练花费的总时间：", spend_time)
    torch.save(model.state_dict(), 'save_model1/trans_model1_8_1000_300.pt')

    # best_test_loss = float('inf')
    # for epoch in range(2):
    #     # start_time = time.time()
    #     for enc_inputs, dec_inputs, target_batch in loader:
    #         '''
    #         enc_inputs: [batch_size, src_len]
    #         dec_inputs: [batch_size, tgt_len]
    #         target_batch: [batch_size, tgt_len]
    #         '''
    #         enc_inputs, dec_inputs, target_batch = enc_inputs, dec_inputs, target_batch
    #         optimizer.zero_grad()
    #         outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
    #         eval_loss = criterion(outputs, target_batch.contiguous().view(-1))
    #         print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    #         if eval_loss < best_test_loss:
    #             best_test_loss = eval_loss
    #             torch.save(model.state_dict(),'save_model/trans_model1.pt')
    #             # torch.save(GCN_model.state_dict(), 'GCN_model.pt')
    #
    #     # end_time = time.time()

    print(enc_inputs.shape)
    print(type(enc_inputs[0]))
    # Test
    enc_inputs, dec_inputs, target_batch = make_batch()
    print(enc_inputs)
    print(enc_inputs.shape)
    print(enc_inputs[0].shape)
    print(torch.unsqueeze(enc_inputs[0],0).shape)

    with open('predict_source_code/output1_8_1000_300.txt', 'w', encoding='utf-8') as f1:
        f1.write('')
    with open('predict_source_code/output1_target_8_1000_300.txt', 'w', encoding='utf-8') as f2:
        f2.write('')
    print("sentence的长度：", len(sentences))

    j2 = int(len(sentences) * 0.2)
    j1 = len(sentences) - j2
    for j in range(j1, len(sentences)):
        model.load_state_dict(torch.load('save_model1/trans_model1_8_1000_300.pt'))
        greedy_dec_input = greedy_decoder(model, torch.unsqueeze(enc_inputs[j], 0).to(Device),
                                          start_symbol=tgt_vocab["SOS"])
        predict, _, _, _ = model(torch.unsqueeze(enc_inputs[j], 0).to(Device), greedy_dec_input)
        predict = predict.data.max(1, keepdim=True)[1]

        print(sentences[j][0], '->', [number_dict[n.item()] for n in predict.squeeze()])
        predict_batch = [number_dict[n.item()] for n in predict.squeeze()]
        # print(predict_batch)
        # print(sentences[j][2])
        str1 = " ".join(predict_batch)
        str2 = sentences[j][2]
        with open('predict_source_code/output1_8_1000_300.txt', 'a', encoding='utf-8') as f1:
            f1.write(str1)

            f1.write('\n')
        with open('predict_source_code/output1_target_8_1000_300.txt', 'a', encoding='utf-8') as f2:
            f2.write(str2)
            f2.write('\n')
    # model.load_state_dict(torch.load('save_model/trans_model1.pt'))
    # greedy_dec_input = greedy_decoder(model, torch.unsqueeze(enc_inputs[0], 0).to(Device), start_symbol=tgt_vocab["SOS"])
    # predict, _, _, _ = model(torch.unsqueeze(enc_inputs[0], 0).to(Device), greedy_dec_input)
    # predict = predict.data.max(1, keepdim=True)[1]
    #
    # print(sentences[0][0], '->', [number_dict[n.item()] for n in predict.squeeze()])
    # predict_batch = [number_dict[n.item()] for n in predict.squeeze()]
    # print(predict_batch)
    # print(sentences[0][2])
    # str1 = " ".join(predict_batch)
    # str2 = sentences[0][2]
    # with open('predict_code/output.txt', 'w', encoding='utf-8') as f1:
    #     f1.write(str1)
    # with open('predict_code/output_target.txt', 'w', encoding='utf-8') as f2:
    #     f2.write(str2)




    # print('first head of last state enc_self_attns')
    # showgraph(enc_self_attns)
    #
    # print('first head of last state dec_self_attns')
    # showgraph(dec_self_attns)
    #
    # print('first head of last state dec_enc_attns')
    # showgraph(dec_enc_attns)