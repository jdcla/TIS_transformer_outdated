import argparse
import sys
import os

from src.transformer_model import TranscriptMLM, TranscriptTIS
from src.transcript_loader import TranscriptLoader, collate_fn

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

def boolean(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.MetavarTypeHelpFormatter):
    pass


class ParseArgs(object):
        def __init__(self):
            parser = argparse.ArgumentParser(
                        description='TIS transformer launch pad',
                        usage='''TIS_transformer.py <command> [<args>]
             Commands:
               pretrain  Pretrain a model using MLM objective
               train     Train a model to detect TIS locations on transcripts
               impute    Impute TIS locations from transcript sequence
            ''')
            parser.add_argument('command', help='Subcommand to run')
            args = parser.parse_args(sys.argv[1:2])
            if args.command not in ['pretrain', 'train', 'impute']:
                print('Unrecognized command')
                parser.print_help()
                exit(1)
            # use dispatch pattern to invoke method with same name
            if args.command == 'pretrain':
                self.pretrain_train(mlm=True)
            elif args.command == 'train':
                self.pretrain_train(mlm=False)
            else:
                self.impute()

        def pretrain_train(self, mlm):
            parser = argparse.ArgumentParser(
                       description=f'{"Pretrain TIS transformer using MLM objective" if mlm else "train TIS transformer"}',
                       formatter_class=CustomFormatter)
            # TWO argvs, ie the command (git) and the subcommand (commit)
            parser.add_argument('data_path', type=str, metavar='data_path',
                                help="path to folder containing the data files")
            parser.add_argument('val_set', type=str, metavar='val_set',
                                help="file in data_path folder used for validation")
            parser.add_argument('test_set', type=str, metavar='test_set',
                                help="file in data_path folder used for testing")
            parser.add_argument('--name', type=str, default='',
                               help="name of the experiment, defines save dir name")

            dl_parse = parser.add_argument_group('DataLoader', 'data loader arguments')
            dl_parse.add_argument('--max_seq_len', type=int, default=25000,
                                help="maximum sequence length of transcripts")
            dl_parse.add_argument('--num_workers', type=int, default=12, 
                                help="number of data loader workers")
            dl_parse.add_argument('--max_transcripts_per_batch', type=int, default=400, 
                                help="maximum of transcripts per batch")
            
            tf_parse = parser.add_argument_group('Model', f'Transformer arguments {"for MLM objective" if mlm else ""}')
            
            if mlm:
                tf_parse.add_argument('--mask_frac', type=float, default=0.85,
                                    help="fraction of inputs that are masked")
                tf_parse.add_argument('--rand_frac', type=float, default=0.10, 
                                    help="fraction of masked inputs that are randomized")
            else:
                tf_parse.add_argument('--transfer_checkpoint', type=str,
                                     help="Path to checkpoint pretrained model")
            
            tf_parse.add_argument('--lr', type=float, default=1e-3,
                                help="learning rate")
            tf_parse.add_argument('--decay_rate', type=float, default=0.95,
                                help="linearly decays learning rate for every epoch")
            tf_parse.add_argument('--num_tokens', type=int, default=7, 
                                help="number of unique input tokens")      
            tf_parse.add_argument('--dim', type=int, default=30,
                                help="dimension of the hidden states")
            tf_parse.add_argument('--depth', type=int, default=6, 
                                help="number of layers")
            tf_parse.add_argument('--heads', type=int, default=6, 
                                help="number of attention heads in every layer")
            tf_parse.add_argument('--dim_head', type=int, default=16,
                                help="dimension of the attention head matrices")
            tf_parse.add_argument('--nb_features', type=int, default=80, 
                                help="number of random features, if not set, will default to (d * log(d)),"\
                                "where d is the dimension of each head") 
            tf_parse.add_argument('--feature_redraw_interval', type=int, default=100, 
                                help="how frequently to redraw the projection matrix")      
            tf_parse.add_argument('--generalized_attention', type=boolean, default=True,
                                help="applies generalized attention functions")
            tf_parse.add_argument('--kernel_fn', type=boolean, default=torch.nn.ReLU(),
                                help="generalized attention function to apply (if generalized attention)")
            tf_parse.add_argument('--reversible', type=boolean, default=True, 
                                help="reversible layers, from Reformer paper")
            tf_parse.add_argument('--ff_chunks', type=int, default=10,
                                help="chunk feedforward layer, from Reformer paper")
            tf_parse.add_argument('--use_scalenorm', type=boolean, default=False,
                                help="use scale norm, from 'Transformers without Tears' paper")
            tf_parse.add_argument('--use_rezero', type=boolean, default=False, 
                                help="use rezero, from 'Rezero is all you need' paper") 
            tf_parse.add_argument('--ff_glu', type=boolean, default=True, 
                                help="use GLU variant for feedforward")      
            tf_parse.add_argument('--emb_dropout', type=float, default=0.1,
                                help="embedding dropout")
            tf_parse.add_argument('--ff_dropout', type=float, default=0.1, 
                                help="feedforward dropout")
            tf_parse.add_argument('--attn_dropout', type=float, default=0.1, 
                                help="post-attn dropout")            
            tf_parse.add_argument('--local_attn_heads', type=int, default=4,
                                help="the amount of heads used for local attention")
            tf_parse.add_argument('--local_window_size', type=int, default=256,
                                help="window size of local attention")
            
            #tr_parse = parser.add_argument_group('Trainer', 'pytorch lightning Trainer arguments')
            #tr_parse = pl.Trainer.add_argparse_args(tr_parse, inplace=True)
            
            parser = pl.Trainer.add_argparse_args(parser)
            args = parser.parse_args(sys.argv[2:])
            
            if mlm:
                print('Training a masked language model training with: {}'.format(args))
                mlm_train(args)
            else:
                print('Training a TIS transformer with: {}'.format(args))
                train(args)

        def impute(self):
            parser = argparse.ArgumentParser(description='Impute TIS locations',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser.add_argument('input', type=str, metavar='test_set',)
            parser.add_argument('transfer_checkpoint', type=str, metavar='checkpoint',
                                help="path to checkpoint of trained model")
            parser.add_argument('--save_path', type=str, metavar='save_path', default='results.npy',
                    help="file in data_path folder used for testing")
            
            dl_parse = parser.add_argument_group('DataLoader', 'data loader arguments')
            dl_parse.add_argument('--max_seq_len', type=int, default=25000,
                                help="maximum sequence length of transcripts")
            dl_parse.add_argument('--num_workers', type=int, default=12, 
                                help="number of data loader workers")
            dl_parse.add_argument('--max_transcripts_per_batch', type=int, default=400, 
                                help="maximum of transcripts per batch")
            
            parser = pl.Trainer.add_argparse_args(parser)
            args = parser.parse_args(sys.argv[2:])
            
            print('Imputing labels from trained model: {}'.format(args))
            impute(args)


def DNA2vec(dna_seq):
    seq_dict = {'A': 0, 'T': 1, 'U':1, 'C': 2, 'G': 3, 'N': 4}
    dna_vec = np.zeros(len(dna_seq), dtype=int)
    for idx in np.arange(len(dna_seq)):
        dna_vec[idx] = seq_dict[dna_seq[idx]]

    return dna_vec


def prep_input(x, device):
    x = torch.LongTensor(np.hstack(([5], x, [6]))).view(1,-1)
    nt_mask = torch.LongTensor(np.hstack(([False], np.full(x.shape[1]-2, True), [False]))).view(1,-1)
    mask = torch.ones_like(x)

    return x.to(device), mask.bool().to(device), nt_mask.bool().to(device)


def mlm_train(args):
    
    mlm = TranscriptMLM(args.mask_frac, args.rand_frac, args.lr, args.decay_rate, args.num_tokens, 
                        args.max_seq_len, args.dim, args.depth, args.heads, args.dim_head, False, 
                        args.nb_features, args.feature_redraw_interval, args.generalized_attention,
                        args.kernel_fn, args.reversible, args.ff_chunks, args.use_scalenorm, 
                        args.use_rezero, False, args.ff_glu, args.emb_dropout, args.ff_dropout,
                        args.attn_dropout, args.local_attn_heads, args.local_window_size)
    
    tr_loader = TranscriptLoader(args.data_path, args.max_seq_len, args.num_workers, 
                                 args.max_transcripts_per_batch, collate_fn)
    tr_loader.setup(val_set=args.val_set, test_set=args.test_set)
    
    tb_logger = pl.loggers.TensorBoardLogger('.', os.path.join('lightning_logs', args.name))
    trainer = pl.Trainer.from_argparse_args(args, reload_dataloaders_every_epoch=True, logger=tb_logger)

    trainer.fit(mlm, datamodule=tr_loader)
    
    trainer.test(mlm, test_dataloaders=tr_loader.test_dataloader())


def train(args):
    
    tis_tr = TranscriptTIS(args.lr, args.decay_rate, args.num_tokens, args.max_seq_len,
                        args.dim, args.depth, args.heads, args.dim_head, False, args.nb_features, 
                        args.feature_redraw_interval, args.generalized_attention, args.kernel_fn, 
                        args.reversible, args.ff_chunks, args.use_scalenorm, args.use_rezero, False,
                        args.ff_glu, args.emb_dropout, args.ff_dropout, args.attn_dropout,
                        args.local_attn_heads, args.local_window_size)

    if args.transfer_checkpoint:
        tis_tr = tis_tr.load_from_checkpoint(args.transfer_checkpoint, lr=args.lr, strict=False)

    tr_loader = TranscriptLoader(args.data_path, args.max_seq_len, args.num_workers, 
                                 args.max_transcripts_per_batch, collate_fn)
    tr_loader.setup(val_set=args.val_set, test_set=args.test_set)

    tb_logger = pl.loggers.TensorBoardLogger('.', os.path.join('lightning_logs', args.name))
    trainer = pl.Trainer.from_argparse_args(args, reload_dataloaders_every_epoch=True, logger=tb_logger)

    trainer.fit(tis_tr, datamodule=tr_loader)

    trainer.test(tis_tr, test_dataloaders=tr_loader.test_dataloader())

def impute(args):
    device = torch.device('cuda') if args.gpus else torch.device('cpu')
    tis_tr = TranscriptTIS.load_from_checkpoint(args.transfer_checkpoint)
    tis_tr.to(device)
    tis_tr.eval()

    if args.input[-4:] == '.npy':
        data = np.load(args.input, allow_pickle=True)
        x_data = np.array([d[:,0] for d in data[:,0] if len(d[:,0]) <  args.max_seq_len], dtype=object)
        tr_ids = data[:,1]
    elif args.input[-3:] == '.fa':
        file = open(args.input)
        data = file.readlines()
        file.close()
        tr_ids = data[0::2]
        tr_seqs = data[1::2]
        tr_ids = [seq.replace('\n','') for seq in tr_ids]
        tr_seqs = [seq.replace('\n','').upper() for seq in tr_seqs]
        x_data = [DNA2vec(seq) for seq in tr_seqs if len(seq) < args.max_seq_len]
    else:
        assert len(args.input) < args.max_seq_len, f'input is longer than maximum input length: {args.max_seq_len}'
        x_data = [DNA2vec(args.input.upper())]
        tr_ids = ['NaN']
    
    print('\nProcessing data')
    out_data = []
    for i,x in enumerate(x_data):
        print('\r{:.2f}%'.format(i/len(x_data)*100), end='')
        out = F.softmax(tis_tr.forward(*prep_input(x, device)), dim=1)[:,1]
        out_data.append(out.detach().cpu().numpy())

    if len(out_data) > 1:
        results = np.vstack((tr_ids, np.array(x_data, dtype=object), 
                             np.array(out_data, dtype=object))).T
    else:
        results = np.array([tr_ids[0], x_data[0], out_data[0]], dtype=object)
    np.save(args.save_path, results)


if __name__ == "__main__":
    args = ParseArgs()