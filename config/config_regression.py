import os
import argparse

from utils.functions import Storage

class ConfigRegression():
    def __init__(self, args):
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            # multi-task
            'v1': self.__V1,
            'v1_semi': self.__V1_Semi,
        }
        # hyper parameters for datasets
        HYPER_DATASET_MAP = self.__datasetCommonParams()

        # normalize
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]
        dataArgs = dataArgs['aligned'] if (commonArgs['need_data_aligned'] and 'aligned' in dataArgs) else dataArgs['unaligned']
        # integrate all parameters
        self.args = Storage(dict(vars(args),
                            **dataArgs,
                            **commonArgs,
                            **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                            ))
    
    def __datasetCommonParams(self):
        root_dataset_dir = '/home/sharing/disk1/liuyihe/SIMS2'
        tmp = {
            'sims3l':{
                'aligned': {
                    # 'dataPath': os.path.join('simsv3_unsup.pkl'),
                    'dataPath': os.path.join(root_dataset_dir,'CHSims_aligned2.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (50, 925, 232), # (text, audio, video)
                    'feature_dims': (768, 25, 177), # (text, audio, video)
                    'train_samples': 2722,
                    'train_mix_samples': 12000,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'Loss',
                },
                'unaligned': {
                    # 'dataPath': os.path.join('simsv3_unsup.pkl'),
                    'dataPath': os.path.join(root_dataset_dir,'SimsLargeV6.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (50, 925, 232), # (text, audio, video)
                    'feature_dims': (768, 25, 177), # (text, audio, video)
                    'train_samples': 2722,
                    'train_mix_samples': 12000,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'Loss',
                }
            },
        }
        return tmp

    def __V1(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': True,
                'use_bert':True,
                'use_bert_finetune': False,
                'early_stop': 8
            },
            # dataset
            'datasetParas':{
                'sims3l':{
                    'hidden_dims': (64, 16, 16),
                    'post_text_dim': 32,
                    'post_audio_dim': 32,
                    'post_video_dim': 64,
                    'post_fusion_out': 16,
                    'dropouts': (0.1,0.1,0.1),
                    'post_dropouts': (0.3,0.3,0.3,0.3),
                    'batch_size': 32,
                    'M': 1.0,
                    'T': 0.2,
                    'A': 0.8,
                    'V': 0.4,
                    'learning_rate_bert': 5e-4,
                    'learning_rate_audio': 5e-4,
                    'learning_rate_video': 1e-3,
                    'learning_rate_other': 5e-4,
                    'weight_decay_bert': 1e-4,
                    'weight_decay_audio': 0,
                    'weight_decay_video': 0,
                    'weight_decay_other': 5e-4,
                }
            },
        }
        return tmp
    
    def __V1_Semi(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_sampling': False,
                'need_sampling_fix': False,
                'need_normalized': False,
                'use_bert':True,
                'use_bert_finetune': False,
                'early_stop': 8
            },
            # dataset
            'datasetParas':{
                'sims3l':{
                    # 'batch_size': 32,
                    # 'learning_rate_bert': 5e-5,
                    # 'learning_rate_audio': 5e-4,
                    # 'learning_rate_video': 5e-4,
                    # 'learning_rate_other': 5e-4,
                    # 'hidden_dims': (128, 32, 128),
                    # 'post_fusion_dim': 16,
                    # 'post_text_dim': 16,
                    # 'post_audio_dim': 8,
                    # 'post_video_dim': 32,
                    # 'dropouts': (0.2,0.2,0.2),
                    # 'post_dropouts': (0.3,0.3,0.3,0.3),
                    # 'M': 1.0,
                    # 'T': 0.2,
                    # 'A': 0.2,
                    # 'V': 0.4,
                    # 'weight_decay_bert': 1e-5,
                    # 'weight_decay_audio': 1e-4,
                    # 'weight_decay_video': 0,
                    # 'weight_decay_other': 5e-4,
                    # 'rec_dropout': 0,
                    # 't_rec': 0.2,
                    # 'a_rec': 0.2,
                    # 'v_rec': 1.0,

                    'batch_size': 64,
                    'learning_rate_bert': 5e-4,
                    'learning_rate_audio': 1e-3,
                    'learning_rate_video': 1e-3,
                    'learning_rate_other': 5e-4,
                    'hidden_dims': (64, 32, 32),
                    'post_fusion_dim': 32,
                    'post_text_dim': 16,
                    'post_audio_dim': 8,
                    'post_video_dim': 64,
                    'dropouts': (0.1,0.1,0.1),
                    'post_dropouts': (0.4,0.4,0.4,0.4),
                    'M': 0.8,
                    'T': 0.6,
                    'A': 0.2,
                    'V': 0.8,
                    'weight_decay_bert': 1e-5,
                    'weight_decay_audio': 0,
                    'weight_decay_video': 1e-5,
                    'weight_decay_other': 1e-5,
                    # 'rec_dropout': 0,
                    # 't_rec': 0.2,
                    # 'a_rec': 0.2,
                    # 'v_rec': 1.0,
                }
            },
        }
        return tmp

    def get_config(self):
        return self.args