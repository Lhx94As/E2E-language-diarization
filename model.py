import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_util
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from transformer import *


class BLSTM_E2E_LID(nn.Module):
    def __init__(self,
                 n_lang=2,
                 dropout=0.25,
                 input_dim=111,
                 hidden_size=256,
                 num_emb_layer=2,
                 num_lstm_layer=3,
                 emb_dim=256):
        super(BLSTM_E2E_LID, self).__init__()
        self.num_classes = n_lang
        self.dropout = dropout
        self.input_dim = input_dim
        self.embed_dim = emb_dim
        self.hidden_size = hidden_size
        self.num_emb_layer = num_emb_layer
        self.num_lstm_layer = num_lstm_layer
        self.embedding_layer = nn.LSTM(input_size=self.input_dim,
                                       hidden_size=self.hidden_size,
                                       num_layers=self.num_emb_layer,
                                       dropout=self.dropout,
                                       batch_first=True,
                                       bidirectional=True)
        self.embedding_fc = nn.Linear(self.hidden_size*2, self.embed_dim)
        self.embedding_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.1, affine=False)
        self.blstm_layer = nn.LSTM(input_size=self.hidden_size*2,
                                   hidden_size=self.hidden_size,
                                   num_layers=self.num_lstm_layer,
                                   dropout=self.dropout,
                                   batch_first=True,
                                   bidirectional=True)
        self.output_fc = nn.Linear(self.hidden_size*2, self.num_classes)

    def forward(self, x):
        # embedding block: stacked BiLSTM+Linear+Tanh+normalize)
        output, _ = self.embedding_layer(x)
        output_ = output.data
        embedding = F.normalize(torch.tanh(self.embedding_fc(output_)))
        # output block: stacked BiLSTM+Linear+Sigmoid
        output, _ = self.blstm_layer(output)
        output_ = output.data
        output = torch.sigmoid(self.output_fc(output_))
        return output.view(-1,self.num_classes), embedding


class Transformer_E2E_LID(nn.Module):
    def __init__(self, input_dim, feat_dim,
                 d_k, d_v, d_ff, n_heads=4,
                 dropout=0.1,n_lang=3, max_seq_len=140,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(Transformer_E2E_LID, self).__init__()
        self.transform = nn.Linear(input_dim, feat_dim)
        self.layernorm1 = LayerNorm(feat_dim)
        self.pos_encoding = PositionalEncoding(max_seq_len=max_seq_len, features_dim=256, device=device)
        self.layernorm2 = LayerNorm(feat_dim)
        self.attention_block1 = EncoderBlock(feat_dim, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.attention_block2 = EncoderBlock(feat_dim, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.attention_block3 = EncoderBlock(feat_dim, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.attention_block4 = EncoderBlock(feat_dim, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.output_fc = nn.Linear(feat_dim, n_lang)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, seq_len, atten_mask):
        output = self.transform(x) #x [B, T, input_dim] => [B, T feat_dim]
        output = self.layernorm1(output)
        output = self.pos_encoding(output,seq_len)
        output = self.layernorm2(output)
        output, _ = self.attention_block1(output, atten_mask)
        output, _ = self.attention_block2(output, atten_mask)
        output, _ = self.attention_block3(output, atten_mask)
        output, _ = self.attention_block4(output, atten_mask)
        output = self.sigmoid(self.output_fc(output))
        return output


class X_Transformer_E2E_LID(nn.Module):
    def __init__(self, input_dim, feat_dim,
                 d_k, d_v, d_ff, n_heads=4,
                 dropout=0.1,n_lang=3, max_seq_len=140,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(X_Transformer_E2E_LID, self).__init__()
        self.input_dim = input_dim
        self.feat_dim = feat_dim
        self.device = device
        # x-vector module
        self.dropout = nn.Dropout(p=dropout)
        self.tdnn1 = nn.Conv1d(in_channels=input_dim, out_channels=512, kernel_size=5, dilation=1)
        self.bn1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.tdnn2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, dilation=2)
        self.bn2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.tdnn3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        self.bn3 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.tdnn4 = nn.Conv1d(in_channels=512, out_channels=1500, kernel_size=1, dilation=1)
        self.bn4 = nn.BatchNorm1d(1500, momentum=0.1, affine=False)
        self.fc5 = nn.Linear(3000, feat_dim)
        self.bn5 = nn.BatchNorm1d(feat_dim, momentum=0.1, affine=False)
        self.fc6 = nn.Linear(feat_dim, feat_dim)
        self.bn6 = nn.BatchNorm1d(feat_dim, momentum=0.1, affine=False)  # momentum=0.5 in asv-subtools
        self.fc7 = nn.Linear(feat_dim, n_lang)
        # attention module
        self.layernorm1 = LayerNorm(feat_dim)
        self.pos_encoding = PositionalEncoding(max_seq_len=max_seq_len, features_dim=256, device=device)
        self.layernorm2 = LayerNorm(feat_dim)
        self.attention_block1 = EncoderBlock(feat_dim, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.attention_block2 = EncoderBlock(feat_dim, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.attention_block3 = EncoderBlock(feat_dim, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.attention_block4 = EncoderBlock(feat_dim, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.output_fc = nn.Linear(feat_dim, n_lang)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, seq_len, atten_mask, eps=1e-5):
        batch_size = x.size(0)
        T_len = x.size(1)
        x = x.view(batch_size*T_len, self.input_dim, -1) # [B,T,input_dim,K]=>[B*T,input_dim,K]
        x = self.bn1(F.relu(self.tdnn1(x)))
        x = self.dropout(x)
        x = self.bn2(F.relu(self.tdnn2(x)))
        x = self.dropout(x)
        x = self.bn3(F.relu(self.tdnn3(x)))
        x = self.dropout(x)
        x = self.bn4(F.relu(self.tdnn4(x)))

        if self.training:
            shape = x.size()
            noise = torch.FloatTensor(shape).to(self.device)
            torch.randn(shape, out=noise)
            x += noise * eps

        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        embedding = self.fc5(stats)
        x = self.bn5(F.relu(embedding))
        x = self.dropout(x)
        x = self.bn6(F.relu(self.fc6(x)))
        x = self.dropout(x)
        cnn_output = self.fc7(x)

        embedding = embedding.view(batch_size, T_len, self.feat_dim) # embedding:[B*T,feat_dim]=>[B, T, feat_dim]
        output = self.layernorm1(embedding)
        output = self.pos_encoding(output,seq_len)
        output = self.layernorm2(output)
        output, _ = self.attention_block1(output, atten_mask)
        output, _ = self.attention_block2(output, atten_mask)
        output, _ = self.attention_block3(output, atten_mask)
        output, _ = self.attention_block4(output, atten_mask)
        output = self.sigmoid(self.output_fc(output))
        return output, cnn_output
