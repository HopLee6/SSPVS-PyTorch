import torch
import torch.nn as nn
import math
from pytorch_transformers import *
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pdb


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
    text = text.unsqueeze(0).expand_as(video_seq)  # (1,b,728)
    if similarity_type == "cos":
        sim = torch.nn.functional.cosine_similarity(text, video_seq, dim=2).relu()
    return sim


class model(nn.Module):
    def __init__(self, args):
        super(model, self).__init__()
        bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert_encoder = bert.encoder
        self.bert_embeddings = bert.embeddings
        self.bert_pooler = bert.pooler

        if not args.text_trainable:
            for param in self.bert_encoder.parameters():
                param.requires_grad = False
            for param in self.bert_embeddings.parameters():
                param.requires_grad = False
            for param in self.bert_pooler.parameters():
                param.requires_grad = False

        self.num_hidden_layers = bert.config.num_hidden_layers

        self.nhid_f = 1024
        self.nhid_w = 768
        self.v_cls = nn.Embedding(1, self.nhid_f)
        self.v_mask = nn.Embedding(1, self.nhid_f)

        self.window_radius = int((args.window_size - 1) / 2)

        self.pad = (0, 0, 0, 0, self.window_radius, self.window_radius)

        self.pos_encoder = PositionalEncoding(
            self.nhid_f, args.dropout_vid, weight=args.pos_enc_weight
        )

        encoder_layers = TransformerEncoderLayer(
            self.nhid_f, args.nhead_vid, args.nhid_vid, args.dropout_vid
        )
        self.transformer_encoder_naive = TransformerEncoder(encoder_layers, 1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, args.nlayers_vid)

        self.prob_dist = nn.Sequential(
            nn.Linear(self.nhid_f, self.nhid_f // 2),
            nn.ReLU(),
            nn.Linear(self.nhid_f // 2, 2),
        )
        self.video_decoder = nn.Sequential(
            nn.Linear(self.nhid_f, self.nhid_f), nn.ReLU()
        )
        self.video_decoder_naive = nn.Sequential(
            nn.Linear(self.nhid_f, self.nhid_f), nn.ReLU()
        )

        self.vid_cls_decoder = nn.Linear(self.nhid_f, self.nhid_f)

        self.project_t_v = nn.Sequential(
            nn.Linear(self.nhid_w, self.nhid_w),
            nn.ReLU(),
            nn.Linear(self.nhid_w, self.nhid_f),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.nhid_w + self.nhid_f, 1024),
            nn.ReLU(),
            nn.Dropout(args.dropout_classifier),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout_classifier),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

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
        return sequence_output, pooled_output

    def video_modeling(self, video):
        src = self.pos_encoder(video)
        output = self.transformer_encoder(src)
        decoded_vid_cls = self.vid_cls_decoder(output[0]).tanh()
        return output, decoded_vid_cls

    def batch_mask(self, vid_feature, masked_pos):
        for i in range(vid_feature.size(1)):
            vid_feature[masked_pos[i], i : i + 1] = self.v_mask.weight
        return vid_feature

    def batch_index_neighbor(self, vid_feature, masked_pos, window_size):
        temp = []
        for i in range(vid_feature.size(1)):
            temp.append(
                vid_feature[
                    masked_pos[i] - window_size : masked_pos[i] + window_size + 1, i
                ]
            )
        return torch.stack(temp, dim=1)

    def interpolate(self, neighbor):
        center = neighbor.size(0) // 2
        src = self.pos_encoder(neighbor)
        output = self.transformer_encoder_naive(src)
        output = self.video_decoder_naive(output[center])
        return output

    def mask_pred(self, vid_feature_masked, encoded_feature, masked_pos):

        vid_feature_padded = torch.nn.functional.pad(
            vid_feature_masked[1:], self.pad, "constant"
        )
        mask_neighbor = self.batch_index_neighbor(
            vid_feature_padded, masked_pos + self.window_radius, self.window_radius
        )  # k,b,d
        interpolated_mask_feature = self.interpolate(mask_neighbor)  # b,d

        encoded_mask_feature = self.batch_index_neighbor(
            encoded_feature, masked_pos + 1, 0
        ).squeeze(
            0
        )  # b,d

        prob_dist = self.prob_dist(encoded_mask_feature).softmax(1)

        decoded_mask_feature = self.video_decoder(encoded_mask_feature).squeeze(
            0
        )  # b,d

        predict_mask_feature = (
            interpolated_mask_feature * prob_dist[:, 0:1]
            + decoded_mask_feature * prob_dist[:, 1:2]
        )

        return predict_mask_feature

    def prepare_input(self, vid_feature, masked_pos):
        vid_feature_masked = self.batch_mask(vid_feature, masked_pos)  # l,b,d
        vid_feature_masked = torch.cat(
            [
                self.v_cls.weight.unsqueeze(1).expand(-1, vid_feature.size(1), -1),
                vid_feature_masked,
            ],
            dim=0,
        )  # l,b,d
        return vid_feature_masked

    def forward(self, inputs):

        vid_feature, info_ids, masked_pos = (
            inputs["video_feature"].transpose(1, 0),  # b,l,d
            inputs["info_embed"],  # b,ll,dd
            inputs["masked_pos"],  # b
        )

        vid_feature_masked = self.prepare_input(vid_feature, masked_pos)
        encoded_feature, decoded_vid_cls = self.video_modeling(
            vid_feature_masked
        )  # l+1,b,d

        predict_mask_feature = self.mask_pred(
            vid_feature_masked, encoded_feature, masked_pos
        )

        encoded_word, text_rep = self.text_modeling(info_ids)

        projected_t_v = self.project_t_v(encoded_word.transpose(1, 0))
        prob_v = self.classifier(torch.cat([text_rep, decoded_vid_cls], 1))
        return {
            "predict_mask_feature": predict_mask_feature,
            "prob_v": prob_v,
            "encoded_frame": encoded_feature,
            "encoded_word": projected_t_v,
        }
