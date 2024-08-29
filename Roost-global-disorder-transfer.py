import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition

import torch
import torch.nn as nn
import os
import re
import json
import pytorch_lightning as L
import wandb

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from torch.optim import AdamW
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR, StepLR
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss, ReLU, NLLLoss
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, roc_auc_score, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

from roost.Data import data_from_composition_general
from roost.Model import Roost
from roost.utils import count_parameters, Scaler, DummyScaler, BCEWithLogitsLoss, Lamb, Lookahead, get_compute_device

from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt

data_type_np = np.float32
data_type_torch = torch.float32
device=get_compute_device()


class RoostDataModule(L.LightningDataModule):
    def __init__(self, train_file: str , 
                 val_file: str, 
                 test_file: str, 
                 batch_size = 256,
                 features='onehot'):
        super().__init__()
        self.train_path = train_file
        self.val_path = val_file
        self.test_path = test_file
        self.batch_size = batch_size
        self.features=features

    def prepare_data(self):
        path='data/el-embeddings/'
        if(self.features == 'onehot'):
            with open(path+'onehot-embedding.json',"r") as f:
                elem_features=json.load(f)
        elif(self.features == 'matscholar'):
            with open(path+'matscholar-embedding.json',"r") as f:
                elem_features=json.load(f)
        elif(self.features == 'mat2vec'):
            with open(path+'mat2vec.json',"r") as f:
                elem_features=json.load(f)
        elif(self.features == 'cgcnn'):
            with open(path+'cgcnn-embedding.json',"r") as f:
                elem_features=json.load(f)
        
        ### loading and encoding trianing data
        if(re.search('.json', self.train_path )):
            self.data_train=pd.read_json(self.train_path)
        elif(re.search('.csv', self.train_path)):
            self.data_train=pd.read_csv(self.train_path)

        self.train_dataset = data_from_composition_general(self.data_train,elem_features)
        self.train_len = len(self.train_dataset)
        
        ### loading and encoding validation data
        if(re.search('.json', self.val_path )):
            self.data_val=pd.read_json(self.val_path)
        elif(re.search('.csv', self.val_path)):
            self.data_val=pd.read_csv(self.val_path)
        
        self.val_dataset = data_from_composition_general(self.data_val,elem_features)
        self.val_len = len(self.val_dataset)

        ### loading and encoding testing data
        if(re.search('.json', self.test_path )):
            self.data_test=pd.read_json(self.test_path)
        elif(re.search('.csv', self.test_path)):
            self.data_test=pd.read_csv(self.test_path)
        
        self.test_dataset = data_from_composition_general(self.data_test,elem_features)
        self.test_len = len(self.test_dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_len, shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_len, shuffle=False)
    

class RoostLightningClass(L.LightningModule):
    def __init__(self, **config):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        self.batch_size=config['data_params']['batch_size']
        self.out_dims=config['model_params']['output_dim']
        self.n_graphs=config['model_params']['n_graphs']
        self.comp_heads=config['model_params']['comp_heads']
        self.internal_elem_dim=config['model_params']['internal_elem_dim']
        self.setup=config['setup_params']
        self.model = Roost(**config['model_params'])
        self.classification = config['classification']
        # maybe need to do it, to unify Roost and CrabNet
        print('\n Model architecture: out_dims, n_graphs, heads, internal_elem_dim')
        print(f'{self.out_dims}, {self.n_graphs}, '
                  f'{self.comp_heads}, {self.internal_elem_dim}')
        print(f'Model size: {count_parameters(self.model)} parameters\n')
        if(config['classification']==True):
            if(config['setup_params']['loss'] == 'BCEWithLogitsLoss'):
                self.criterion = BCEWithLogitsLoss

            if(re.search('.json', config['data_params']['train_path'] )):
                train_data=pd.read_json(config['data_params']['train_path'])
            elif(re.search('.csv', config['data_params']['train_path'])):
                train_data=pd.read_csv(config['data_params']['train_path'])
            y=train_data['disorder'].values
            self.step_size = len(y)
            if(np.sum(y)>0):
                self.weight=torch.tensor(((len(y)-np.sum(y))/np.sum(y)),dtype=data_type_torch).to(device)
        elif(config['classification']==False):
            self.criterion = L1Loss()
            if(re.search('.json', config['data_params']['train_path'] )):
                train_data=pd.read_json(config['data_params']['train_path'])
            elif(re.search('.csv', config['data_params']['train_path'])):
                train_data=pd.read_csv(config['data_params']['train_path'])
            y=train_data['disorder'].values
            self.step_size = len(y)
            self.scaler=Scaler(y)

    def define_loss(self):
        if(config['classification']==True):
            self.criterion = BCEWithLogitsLoss
            if(re.search('.json', config['data_params']['train_path'] )):
                train_data=pd.read_json(config['data_params']['train_path'])
            elif(re.search('.csv', config['data_params']['train_path'])):
                train_data=pd.read_csv(config['data_params']['train_path'])
            y=train_data['disorder'].values
            self.step_size = len(y)
            if(np.sum(y)>0):
                self.weight=torch.tensor(((len(y)-np.sum(y))/np.sum(y)),dtype=data_type_torch).to(device)
        elif(config['classification']==False):
            self.criterion = L1Loss()
        self.save_hyperparameters()
        
    def forward(self, batch):
        out = self.model(batch.x, batch.edge_index, batch.pos, batch.batch)
        return out

    def configure_optimizers(self):
        if(self.setup['optim'] == 'AdamW'):
        # We use AdamW optimizer with MultistepLR scheduler as in the original Roost model
            optimizer = torch.optim.AdamW(self.parameters(),lr=self.setup['learning_rate'], 
                                        weight_decay=self.setup['weight_decay']) 
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=self.setup['gamma'])

        elif(self.setup['optim'] == 'Lamb'):
            base_optim = Lamb(params=self.model.parameters(),lr=0.001)
            optimizer = Lookahead(base_optimizer=base_optim)
            scheduler = CyclicLR(optimizer,
                                base_lr=self.setup['base_lr'],
                                max_lr=self.setup['max_lr'],
                                cycle_momentum=False,
                                step_size_up=self.step_size)

        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        logits=self(batch)
        if(self.classification == True):
            loss=self.criterion(logits, batch.y,self.weight)
            prediction = torch.sigmoid(logits)
            y_pred = prediction.detach().cpu().numpy() > 0.5
            acc=balanced_accuracy_score(batch.y.detach().cpu().numpy().astype(bool),y_pred)
            f1=f1_score(batch.y.detach().cpu().numpy().astype(bool),y_pred)
            mc=matthews_corrcoef(batch.y.detach().cpu().numpy().astype(bool),y_pred)

            self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("train_f1", f1, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            self.log("train_mc", mc, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        elif(self.classification == False):
            y=self.scaler.scale(batch.y)
            loss=self.criterion(logits, y)
            y=self.scaler.unscale(y)
            prediction=self.scaler.unscale(logits)

            mse = mean_squared_error(y.detach().cpu().numpy(), prediction.detach().cpu().numpy())
            mae = mean_absolute_error(y.detach().cpu().numpy(), prediction.detach().cpu().numpy())
            self.log("train_mse", mse, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            self.log("train_mae", mae, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits=self(batch)
        if(self.classification == True):
            loss=self.criterion(logits, batch.y,self.weight)
            prediction = torch.sigmoid(logits)
            y_pred = prediction.view(-1).detach().cpu().numpy() > 0.5
            acc=balanced_accuracy_score(batch.y.detach().cpu().numpy().astype(bool),y_pred)
            f1=f1_score(batch.y.detach().cpu().numpy().astype(bool),y_pred)
            mc=matthews_corrcoef(batch.y.detach().cpu().numpy().astype(bool),y_pred)

            self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("val_f1", f1, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            self.log("val_mc", mc, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        elif(self.classification == False):
            y=self.scaler.scale(batch.y)
            loss=self.criterion(logits, y)
            y=self.scaler.unscale(y)
            prediction=self.scaler.unscale(logits)

            mse = mean_squared_error(y.detach().cpu().numpy(), prediction.detach().cpu().numpy())
            mae = mean_absolute_error(y.detach().cpu().numpy(), prediction.detach().cpu().numpy())
            self.log("val_mse", mse, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            self.log("val_mae", mae, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        return loss
    
    def test_step(self, batch, batch_idx):
        logits=self(batch)
        if(self.classification == True):
            loss=self.criterion(logits, batch.y,self.weight)
            prediction = torch.sigmoid(logits)
            y_pred = prediction.view(-1).detach().cpu().numpy() > 0.5
            acc=balanced_accuracy_score(batch.y.detach().cpu().numpy().astype(bool),y_pred)
            f1=f1_score(batch.y.detach().cpu().numpy().astype(bool),y_pred)
            mc=matthews_corrcoef(batch.y.detach().cpu().numpy().astype(bool),y_pred)

            self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("test_f1", f1, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            self.log("test_mc", mc, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)

        elif(self.classification == False):
            y=self.scaler.scale(batch.y)
            loss=self.criterion(logits, y)
            y=self.scaler.unscale(y)
            prediction=self.scaler.unscale(logits)

            mse = mean_squared_error(y.detach().cpu().numpy(), prediction.detach().cpu().numpy())
            mae = mean_absolute_error(y.detach().cpu().numpy(), prediction.detach().cpu().numpy())
            self.log("test_mse", mse, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            self.log("test_mae", mae, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits=self(batch)
        if(self.classification == True):
            prediction = torch.sigmoid(logits)
            y_pred = prediction.view(-1).detach().cpu().numpy() > 0.5
            return batch.y.view(-1).detach().cpu().numpy(), prediction, y_pred
        elif(self.classification == False):
            prediction=self.scaler.unscale(logits)
            return batch.y.view(-1).detach().cpu().numpy(), prediction

    
def main(**config):
    L.seed_everything(config['seed'])

    # data_file = 'data/general_disorder.csv'
    # df=pd.read_csv(data_file,usecols=['formula', 'disorder'])
    # index=np.linspace(0,len(df)-1,len(df),dtype=int)
    # train_idx,test_idx= train_test_split(index, test_size=0.2, random_state=config['seed'])
    # train_idx,val_idx= train_test_split(train_idx, test_size=0.1, random_state=config['seed'])
    # val_set = df.iloc[val_idx]
    # val_set.to_csv('data/roost_data/val.csv',index=False)
    # test_set = df.iloc[test_idx]
    # test_set.to_csv('data/roost_data/test.csv',index=False)
    # train_set = df.iloc[train_idx]
    # train_set.to_csv('data/roost_data/train.csv',index=False)
    
    # wandb_logger = WandbLogger(project="Roost-formation-energy", config=config, log_model="all")
    # model = RoostLightningClass(**config)
    # trainer = Trainer(devices=1, accelerator='gpu',max_epochs=config['epochs'], logger=wandb_logger, 
    #                   callbacks=[StochasticWeightAveraging(swa_epoch_start=config['setup_params']['swa_epoch_start'],swa_lrs=config['setup_params']['swa_lrs']),
    #                          ModelCheckpoint(monitor='val_mae', mode='min',dirpath='roost_energy_models/trained_models/', filename='energy-{epoch:02d}-{val_acc:.2f}'),
    #                          EarlyStopping(monitor='val_loss', mode='min', patience=config['patience']),
    #                          LearningRateMonitor(logging_interval='step')])
    # energy_data = RoostDataModule(config['data_params']['train_path'],
    #                                config['data_params']['val_path'],
    #                                config['data_params']['test_path'], features=config['data_params']['embed'])
    # trainer.fit(model, datamodule=energy_data)
    # y_true,  y_pred=trainer.predict(ckpt_path='best', datamodule=energy_data)[0]
    # metrics={}
    # metrics['mae']=mean_absolute_error(y_true,y_pred)
    # metrics['mse']=mean_squared_error(y_true,y_pred)
    # metrics['r2']=r2_score(y_true,y_pred)
    # pred_matrix={}
    # pred_matrix['y_true']=y_true
    # pred_matrix['y_true']=y_pred
                                                     
    if(re.search('.json', config['data_params']['train_path'] )):
        train_data=pd.read_json(config['data_params']['train_path'])
    elif(re.search('.csv', config['data_params']['train_path'])):
        train_data=pd.read_csv(config['data_params']['train_path'])
    y=train_data['disorder'].values
    step_size = len(y)
    if(np.sum(y)>0):
        weight=torch.tensor(((len(y)-np.sum(y))/np.sum(y)),dtype=data_type_torch).to(device)
    
    wandb_logger = WandbLogger(project="Roost-global-disorder", config=config, log_model="all")
    model = RoostLightningClass.load_from_checkpoint('roost_energy_models/trained_models/energy-epoch=93-val_acc=0.00.ckpt',
                                                     classification=True, criterion=BCEWithLogitsLoss, weight=weight,
                                                     step_size=step_size, batch_size=config['data_params']['batch_size'],
                                                     out_dims=config['model_params']['output_dim'],
                                                     n_graphs=config['model_params']['n_graphs'],
                                                     comp_heads=config['model_params']['comp_heads'],
                                                     internal_elem_dim=config['model_params']['internal_elem_dim'],
                                                     setup=config['setup_params'])
    model.define_loss()
    model.setup=config['setup_params']
    
    print('hyperparameters:', model.hparams)
    
    trainer = Trainer(devices=1, accelerator='gpu',max_epochs=config['epochs'], logger=wandb_logger, 
                  callbacks=[StochasticWeightAveraging(swa_epoch_start=config['setup_params']['swa_epoch_start'],swa_lrs=config['setup_params']['swa_lrs']),
                             ModelCheckpoint(monitor='val_acc', mode='max',dirpath='roost_models/trained_models/', filename='disorder-{epoch:02d}-{val_acc:.2f}'),
                             EarlyStopping(monitor='val_acc', mode='min', patience=config['patience']),
                             LearningRateMonitor(logging_interval='step')])
    disorder_data = RoostDataModule(config['data_params']['train_path'],
                                   config['data_params']['val_path'],
                                   config['data_params']['test_path'], features=config['data_params']['embed'])
    trainer.fit(model, datamodule=disorder_data)
    y_true, prediction, y_pred=trainer.predict(ckpt_path='best', datamodule=disorder_data)[0]
    metrics={}
    metrics['acc']=balanced_accuracy_score(y_true,y_pred)
    metrics['f1']=f1_score(y_true,y_pred)
    metrics['precision']=precision_score(y_true,y_pred)
    metrics['recall']=recall_score(y_true,y_pred)
    metrics['mc']=matthews_corrcoef(y_true,y_pred)
    metrics['roc_auc']=roc_auc_score(y_true,prediction)
    pred_matrix={}
    pred_matrix['y_true']=y_true
    pred_matrix['y_score']=prediction.detach().numpy()
    pred_matrix['y_true']=y_pred
   
    wandb.log(metrics)
    wandb.log(pred_matrix)
    return
  

if __name__=='__main__':
    wandb.init(project="Roost-global-disorder")
    wandb.login(key='b11d318e434d456c201ef1d3c86a3c1ce31b98d7')

    with open('roost/roost_config.json','r') as f:
        config=json.load(f)

    path='data/el-embeddings/'
    if(config['data_params']['embed']=='onehot'):
        with open(path+'onehot-embedding.json',"r") as f:
            elem_features=json.load(f)
    elif(config['data_params']['embed']=='matscholar'):
        with open(path+'matscholar-embedding.json',"r") as f:
            elem_features=json.load(f)
    elif(config['data_params']['embed']=='mat2vec'):
        with open(path+'mat2vec.json',"r") as f:
            elem_features=json.load(f)
    elif(config['data_params']['embed']=='cgcnn'):
        with open(path+'cgcnn-embedding.json',"r") as f:
            elem_features=json.load(f)

    elem_emb_len=len(elem_features['H'])
    config['model_params']['input_dim']=elem_emb_len

    
    sweep_config = {
    'method': 'random',
    'parameters': {config['setup_params']['optim']: {'values': ['AdamW','Lamb']},
                   config['setup_params']['learning_rate']: {'values': [0.001,0.0005,0.0001,0.000005]},
                   config['setup_params']['weight_decay']: {'values': [1e-6,5e-6,5e-5]},
                   config['setup_params']['base_lr']: {'values': [1e-4,1e-3,1e-5]},
                   config['setup_params']['base_lr']: {'values': [5e-3,1e-3,1e-4]},
                   config['model_params']['comp_heads']: {'values':[3,4,5]},
                   config['model_params']['internal_elem_dim']: {'values': [64,128,256]},
                   config['data_params']['batch_size']: {'values': [256,512,1024]}
    }
    }
    print('Start sweeping with different parameters for Roost...')

    # sweep_id = wandb.sweep(sweep=sweep_config, project="Roost-global-disorder")

    # wandb.agent(sweep_id, function=main, count=3)

    main(**config)

    wandb.finish()
