import torch
import torch.optim as optim
import torch.nn as nn
from .ist_lstm_model import IST_LSTM_Predictor
from .ist_lstm_util import masked_mae, masked_mape, masked_rmse


class IST_LSTM_Trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, hidden_dim, dropout,
                 lrate, wdecay, device, adj_conn=None, adj_cor=None, adj_sml=None,
                 num_heads=4, window_size=2, num_layers_main=3, num_layers_sub=2,
                 pred_steps=1, use_dcg=True):
        
        self.model = IST_LSTM_Predictor(
            num_nodes=num_nodes,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            num_layers_main=num_layers_main,
            num_layers_sub=num_layers_sub,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout,
            adj_conn=adj_conn,
            adj_cor=adj_cor,
            adj_sml=adj_sml,
            use_dcg=use_dcg,
            pred_steps=pred_steps
        )
        self.model.to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = masked_mae
        self.scaler = scaler
        self.clip = 5
        self.device = device
        self.pred_steps = pred_steps

    def train(self, input, real_val):
        """
        input: (B, T, N, C_in) or (B, C_in, N, T)
        real_val: (B, T_out, N) or (B, N, T_out)
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        output = self.model(input)  # (B, pred_steps, N, 1)
        output = output.squeeze(-1)  # (B, pred_steps, N)
        
        # 反归一化
        predict = self.scaler.inverse_transform(output)
        
        # 确保real_val维度正确
        if real_val.dim() == 3 and real_val.size(1) == self.num_nodes:
            real_val = real_val.permute(0, 2, 1)  # (B, T_out, N)
        
        loss = self.loss(predict, real_val, 0.0)
        loss.backward()
        
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        
        self.optimizer.step()
        
        mape = masked_mape(predict, real_val, 0.0).item()
        rmse = masked_rmse(predict, real_val, 0.0).item()
        
        return loss.item(), mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(input)
            output = output.squeeze(-1)
            predict = self.scaler.inverse_transform(output)
            
            if real_val.dim() == 3 and real_val.size(1) == self.num_nodes:
                real_val = real_val.permute(0, 2, 1)
            
            loss = self.loss(predict, real_val, 0.0)
            mape = masked_mape(predict, real_val, 0.0).item()
            rmse = masked_rmse(predict, real_val, 0.0).item()
        
        return loss.item(), mape, rmse
    
    @property
    def num_nodes(self):
        return self.model.encoder.num_nodes
