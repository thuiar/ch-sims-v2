import torch
from torch import nn

from transformers import Wav2Vec2Model

_HIDDEN_STATES_START_POSITION = 2

class Wav2vec2Baseline(nn.Module):
    """ If you want to use pretrained model, or simply the standard structure implemented
        by Pytorch official, please use this template. It enable you to easily control whether
        use or not the pretrained weights, and whether to freeze the internal layers or not,
        This is made for wav2vec, but you can also adapt it to other structures by changing
        the `torch.hub.load` content.
    """
    def __init__(
        self, 
        use_weighted_layer_sum=True,
        model_type='dense',
        hidden_size=128,
        num_labels=1,
        freeze=True,
    ):
        super().__init__()
        self.__dict__.update(locals())
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        num_layers = self.wav2vec2.config.num_hidden_layers + 1  # transformer layers + input embeddings
        if use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        if model_type == 'dense':
            self.projector = nn.Sequential (
                nn.Linear(self.wav2vec2.config.hidden_size, hidden_size), 
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(p=0.2),
            )
            self.classifier = nn.Linear(hidden_size, num_labels)
        elif model_type == 'lstm':
            self.projector = nn.Sequential (
                nn.Linear(self.wav2vec2.config.hidden_size, hidden_size), 
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, bidirectional=True)
            )
            self.classifier = nn.Linear(hidden_size*2, num_labels)
        if freeze:
            self.freeze_base_model()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature extractor so that its parameters
        will not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def prepare_mask(self, length, shape, dtype, device):
        #Modified from huggingface
        mask = torch.zeros(
            shape, dtype=dtype, device=device
        )
        # these two operations makes sure that all values
        # before the output lengths indices are attended to
        mask[
            (torch.arange(mask.shape[0], device=device), length - 1)
        ] = 1
        mask = mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return mask

    #From huggingface
    def _get_feat_extract_output_lengths(self, input_length):
        """
        Computes the output length of the convolutional layers
        """
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1
        for kernel_size, stride in zip(self.wav2vec2.config.conv_kernel, self.wav2vec2.config.conv_stride):
            input_length = _conv_out_length(input_length, kernel_size, stride)
        return input_length

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths)
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

    def forward(
        self, 
        input_values,
        input_length=None,
        output_hidden_states=None
        ):
        output_hidden_states = True if self.use_weighted_layer_sum else output_hidden_states

        if input_length is not None:
            attention_mask = self.prepare_mask(input_length, input_values.shape[:2], input_values.dtype, input_values.device)

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
        
        hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
        hidden_states = torch.stack(hidden_states, dim=1)
        norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
        hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        self.hidden_states = hidden_states
        if self.model_type == 'dense':
            hidden_states = self.projector(hidden_states)
        elif self.model_type == 'lstm':
            hidden_states, _ = self.projector(hidden_states)
        elif self.model_type == 'fusion':
            pass

        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states = hidden_states * padding_mask.unsqueeze(-1)
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        # return self.classifier(pooled_output)
        return pooled_output