import torch
import torch.nn as nn
import math
from pytorch_transformers import *
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from functools import reduce


class PositionalEncoding(nn.Module):
    def __init__(
        self, d_model: int, dropout: float = 0.1, max_len: int = 5000, weight=1
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        self.pos_weight = weight

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)] * self.pos_weight
        return self.dropout(x)


def similarity(text, video_seq, similarity_type="cos"):
    text = text.expand_as(video_seq)  # (1,b,728)
    if similarity_type == "cos":
        sim = torch.nn.functional.cosine_similarity(text, video_seq, dim=2).unsqueeze(2)
        sim = (sim + 1) / 2
    return sim


class model(nn.Module):
    def __init__(self, args):
        super(model, self).__init__()
        self.with_info_guide = getattr(args, "with_info_guide", False)
        self.nhid_f = 1024

        self.pos_encoder = PositionalEncoding(
            self.nhid_f, args.dropout_vid, weight=args.pos_enc_weight
        )

        encoder_layers = TransformerEncoderLayer(
            self.nhid_f, args.nhead_vid, args.nhid_vid, args.dropout_vid
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, args.nlayers_vid)

        self.score_predict = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(0.1),
                    nn.Linear(1024, 1),
                    nn.Sigmoid(),
                )
                for _ in range(args.iter)
            ]
        )

        if self.with_info_guide:
            bert = BertModel.from_pretrained("bert-base-uncased")
            self.bert_encoder = bert.encoder
            self.bert_embeddings = bert.embeddings
            self.bert_pooler = bert.pooler

            self.nhid_w = 768

            for param in self.bert_encoder.parameters():
                param.requires_grad = False
            for param in self.bert_embeddings.parameters():
                param.requires_grad = False

            if not args.text_trainable:
                for param in self.bert_pooler.parameters():
                    param.requires_grad = False

            self.num_hidden_layers = bert.config.num_hidden_layers
            self.project_t_v = nn.Sequential(
                nn.Linear(self.nhid_w, self.nhid_w),
                nn.ReLU(),
                nn.Linear(self.nhid_w, self.nhid_f),
            )

        self.v_cls = nn.Embedding(1, self.nhid_f)

        self.iter = args.iter

    def text_modeling(self, info_ids):

        token_type_ids = torch.zeros_like(info_ids)
        bert_input = self.bert_embeddings(
            info_ids, position_ids=None, token_type_ids=token_type_ids
        )
        attention_mask = torch.ones(bert_input.size(0), bert_input.size(1)).to(
            bert_input.device
        )
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.bert_encoder.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.num_hidden_layers

        encoder_outputs = self.bert_encoder(
            bert_input, extended_attention_mask, head_mask=head_mask
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.bert_pooler(sequence_output)
        return pooled_output

    def video_modeling(self, video, iter, text_rep=None):
        video = self.prepare_input(video)
        src = self.pos_encoder(video)
        output = self.transformer_encoder(src)[1:] + video[1:]
        if self.with_info_guide and iter == 0:
            output = output + text_rep

        score = self.score_predict[iter](output)
        return score

    def prepare_input(self, vid_feature):
        vid_feature = torch.cat(
            [
                self.v_cls.weight.unsqueeze(1).expand(-1, vid_feature.size(1), -1),
                vid_feature,
            ],
            dim=0,
        )  # l,b,d
        return vid_feature

    def forward(self, inputs):

        vid_feature = inputs["video_feature"].transpose(0, 1)

        if self.with_info_guide:
            info_ids = inputs["info_ids"]
            text_rep = self.text_modeling(info_ids)  # b,d'
            projected_t_v = self.project_t_v(text_rep)  # b,d
            projected_t_v = projected_t_v.unsqueeze(0).expand_as(vid_feature)  # l,b,d
        else:  # no info guide
            projected_t_v = None

        score_list = []
        for i in range(self.iter):
            score = self.video_modeling(vid_feature, i, projected_t_v)
            score_list.append(score)
            vid_feature = vid_feature * score + vid_feature
        final_score = reduce(lambda x, y: x * y, score_list)
        output = {"frame_score": final_score.transpose(0, 1)}  # b,l,1
        return output
