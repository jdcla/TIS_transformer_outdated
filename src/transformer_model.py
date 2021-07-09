import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from performer_pytorch import PerformerLM
from pdb import set_trace

class TranscriptMLM(pl.LightningModule):
    def __init__(self, mask_frac, rand_frac, lr,  decay_rate, num_tokens, max_seq_len, dim, 
                 depth, heads, dim_head, causal, nb_features, feature_redraw_interval,
                 generalized_attention, kernel_fn, reversible, ff_chunks, use_scalenorm,
                 use_rezero, tie_embed, ff_glu, emb_dropout, ff_dropout, attn_dropout,
                 local_attn_heads, local_window_size):
        super().__init__()
        self.model = PerformerLM(num_tokens=num_tokens, max_seq_len=max_seq_len, 
                                 dim=dim, depth=depth, heads=heads, dim_head=dim_head, 
                                 causal=causal, nb_features=nb_features, 
                                 feature_redraw_interval=feature_redraw_interval,
                                 generalized_attention=generalized_attention, kernel_fn=kernel_fn,
                                 reversible=reversible, ff_chunks=ff_chunks, use_scalenorm=use_scalenorm,
                                 use_rezero=use_rezero, tie_embed=tie_embed,
                                 ff_glu=ff_glu, emb_dropout=emb_dropout,
                                 ff_dropout=ff_dropout, attn_dropout=attn_dropout, 
                                 local_attn_heads=local_attn_heads, local_window_size=local_window_size,
                                auto_check_redraw=False)
        
        self.mask_c = mask_frac
        self.mask_m = self.mask_c + (1 - self.mask_c)*(1-rand_frac)
        
        self.mask_token = 4
        
        self.dim = dim
        self.lr = lr
        self.decay_rate = decay_rate

        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
    
        self.save_hyperparameters()

    def randomisation(self, x, dist, val=False):
        # self supervised learning protocol
        # apply unknown (N) tokens to input
        mask = torch.logical_and(dist > self.mask_c, dist < self.mask_m)
        x[mask] = self.mask_token

        # apply random tokens to input
        if not val:
            mask = dist > self.mask_m
            x[mask] = torch.randint(0,self.mask_token,(mask.sum(),), device=self.device)

        return x            

    def forward(self, x, val=False):
        
        x = self.model(x, mask=torch.ones_like(x, device=self.device).bool(), 
                       return_encodings=False)

        return x

    def training_step(self, batch, batch_idx):
        x = batch[0]

        dist = torch.empty(x.shape, device=self.device).uniform_(0,1,)
        mask = dist > self.mask_c
        y_true = x.masked_select(mask)
        y_hat = self(self.randomisation(x, dist, val=True))[mask]

        loss = F.cross_entropy(y_hat, y_true)
        self.log('train_loss', loss, sync_dist=True)
        self.train_acc(F.softmax(y_hat, dim=1), y_true)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, sync_dist=True)

        return loss
        
    def validation_step(self, batch, batch_idx):
        x = batch[0]
        
        dist = torch.empty(x.shape, device=self.device).uniform_(0,1,)
        mask = dist > self.mask_c
        y_true = x.masked_select(mask)
        y_hat = self(self.randomisation(x, dist, val=True))[mask]

        self.log('val_loss', F.cross_entropy(y_hat, y_true), on_step=False, 
                 on_epoch=True, sync_dist=True)
        self.val_acc(F.softmax(y_hat, dim=1), y_true)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, sync_dist=True)
                
    def test_step(self, batch, batch_idx):
        x = batch[0]
        
        dist = torch.empty(x.shape, device=self.device).uniform_(0,1,)
        mask = dist > self.mask_c
        y_true = x.masked_select(mask)
        y_hat = self(self.randomisation(x, dist, val=True))[mask]
        
        self.log('test_loss', F.cross_entropy(y_hat, y_true), on_step=False, 
                 on_epoch=True, sync_dist=True)
        self.test_acc(F.softmax(y_hat, dim=1), y_true)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, sync_dist=True)
            
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lambda1 = lambda epoch: self.decay_rate**epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda1)
        return optimizer

    
class TranscriptTIS(pl.LightningModule):
    def __init__(self, lr,  decay_rate, num_tokens, max_seq_len, dim, 
                 depth, heads, dim_head, causal, nb_features, feature_redraw_interval,
                 generalized_attention, kernel_fn, reversible, ff_chunks, use_scalenorm,
                 use_rezero, tie_embed, ff_glu, emb_dropout, ff_dropout, attn_dropout,
                 local_attn_heads, local_window_size):
        super().__init__()
        self.model = PerformerLM(num_tokens=num_tokens, max_seq_len=max_seq_len, 
                                 dim=dim, depth=depth, heads=heads, dim_head=dim_head, 
                                 causal=causal, nb_features=nb_features, 
                                 feature_redraw_interval=feature_redraw_interval,
                                 generalized_attention=generalized_attention, kernel_fn=kernel_fn,
                                 reversible=reversible, ff_chunks=ff_chunks, use_scalenorm=use_scalenorm,
                                 use_rezero=use_rezero, tie_embed=tie_embed,
                                 ff_glu=ff_glu, emb_dropout=emb_dropout,
                                 ff_dropout=ff_dropout, attn_dropout=attn_dropout, 
                                 local_attn_heads=local_attn_heads, local_window_size=local_window_size)
        
        self.mask_token = 4
        
        self.dim = dim
        self.lr = lr
        self.decay_rate = decay_rate

        self.val_rocauc = pl.metrics.AUROC(pos_label=1, compute_on_step=False)
        self.val_prauc = pl.metrics.AveragePrecision(pos_label=1, compute_on_step=False)
        
        self.test_rocauc = pl.metrics.AUROC(pos_label=1, compute_on_step=False)
        self.test_prauc = pl.metrics.AveragePrecision(pos_label=1, compute_on_step=False)
        
        self.ff_1 = torch.nn.Linear(dim,dim*2)
        self.ff_2 = torch.nn.Linear(dim*2,2)
    
        self.save_hyperparameters()

    def forward(self, x, mask=None, nt_mask=None):
        x = self.model(x, return_encodings=True, mask=mask)
        x = x[torch.logical_and(mask, nt_mask)]
        x = x.view(-1,self.dim)
        
        x = F.relu(self.ff_1(x))
        x = self.ff_2(x)
        
        return x

    def training_step(self, batch, batch_idx):
        x = batch[0]
        y_true =  batch[1][torch.logical_and(batch[3],batch[4])].view(-1)

        y_hat = self(x, batch[3], batch[4])
        
        loss = F.cross_entropy(y_hat, y_true)
        self.log('train_loss', loss)

        return loss
        
    def validation_step(self, batch, batch_idx):
        x = batch[0]
        y_true = batch[1][torch.logical_and(batch[3],batch[4])].view(-1)
        
        y_hat = self(x, batch[3], batch[4]) 
        
        self.val_prauc(F.softmax(y_hat, dim=1)[:,1], y_true)
        self.val_rocauc(F.softmax(y_hat, dim=1)[:,1], y_true)
        
        self.log('val_loss', F.cross_entropy(y_hat, y_true))
        self.log('val_prauc', self.val_prauc, on_step=False, on_epoch=True)
        self.log('val_rocauc', self.val_rocauc, on_step=False, on_epoch=True)
                
    def test_step(self, batch, batch_idx, ):
        mask_outer = torch.logical_and(batch[3],batch[4])
        
        x = batch[0]
        y_true = batch[1][mask_outer].view(-1)
                
        y_hat = self(x, batch[3], batch[4])
        
        self.test_prauc(F.softmax(y_hat, dim=1)[:,1], y_true)
        self.test_rocauc(F.softmax(y_hat, dim=1)[:,1], y_true)

        self.log('test_loss', F.cross_entropy(y_hat, y_true))
        self.log('test_prauc', self.test_prauc, on_step=False, on_epoch=True)
        self.log('test_rocauc', self.test_rocauc, on_step=False, on_epoch=True)

        y_hat_grouped = torch.split(F.softmax(y_hat, dim=1)[:,1], list(batch[5]))
        y_true_grouped = torch.split(batch[1][mask_outer], list(batch[5]))
        x_grouped = torch.split(batch[0][mask_outer], list(batch[5]))
        
        return y_hat_grouped, y_true_grouped, x_grouped, batch[2]
    
    def on_test_epoch_start(self):
        self.test_outputs = []
        self.test_targets = []
        self.x_in = []
        self.labels = []
        
    def test_step_end(self, results):
        # this out is now the full size of the batch
        self.test_outputs = self.test_outputs + list(results[0])
        self.test_targets = self.test_targets + list(results[1])
        self.x_in = self.x_in + list(results[2])
        self.labels = self.labels + list(results[3])
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lambda1 = lambda epoch: self.decay_rate**epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda1)

        return optimizer