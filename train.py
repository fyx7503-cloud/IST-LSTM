import torch
import numpy as np
import argparse
import time
import os
from models.ist_lstm_util import (
    load_pickle_data, create_dataloaders, load_adj_pickle,
    metric, StandardScaler
)
from models.ist_lstm_engine import IST_LSTM_Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str, default='hangzhou', help='data path: hangzhou or shanghai')
parser.add_argument('--adj_conn', type=str, default=None, help='connectivity adjacency matrix path')
parser.add_argument('--adj_cor', type=str, default=None, help='correlation adjacency matrix path')
parser.add_argument('--adj_sml', type=str, default=None, help='similarity adjacency matrix path')
parser.add_argument('--use_dcg', action='store_true', help='whether to use DCG graph structure')

# 模型参数
parser.add_argument('--seq_length', type=int, default=4, help='input time steps')
parser.add_argument('--pred_steps', type=int, default=4, help='prediction time steps')
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')
parser.add_argument('--num_layers_main', type=int, default=3, help='number of main branch LSTM layers')
parser.add_argument('--num_layers_sub', type=int, default=2, help='number of sub branch IST-LSTM layers')
parser.add_argument('--num_heads', type=int, default=4, help='number of attention heads')
parser.add_argument('--window_size', type=int, default=2, help='Swin Transformer window size')

# 训练参数
parser.add_argument('--in_dim', type=int, default=2, help='input feature dimension')
parser.add_argument('--num_nodes', type=int, default=80, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=200, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--save', type=str, default='./checkpoints/ist_lstm', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')

args = parser.parse_args()


def main():
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 自动推断数据集参数
    if args.data == 'hangzhou':
        args.num_nodes = 80
        if args.adj_conn is None:
            args.adj_conn = 'hangzhou/graph_hz_conn.pkl'
        if args.adj_cor is None:
            args.adj_cor = 'hangzhou/graph_hz_cor.pkl'
        if args.adj_sml is None:
            args.adj_sml = 'hangzhou/graph_hz_sml.pkl'
    elif args.data == 'shanghai':
        args.num_nodes = 288
        if args.adj_conn is None:
            args.adj_conn = 'shanghai/graph_sh_conn.pkl'
        if args.adj_cor is None:
            args.adj_cor = 'shanghai/graph_sh_cor.pkl'
        if args.adj_sml is None:
            args.adj_sml = 'shanghai/graph_sh_sml.pkl'
    
    # 加载邻接矩阵
    adj_conn = load_adj_pickle(args.adj_conn) if args.adj_conn and os.path.exists(args.adj_conn) else None
    adj_cor = load_adj_pickle(args.adj_cor) if args.adj_cor and os.path.exists(args.adj_cor) else None
    adj_sml = load_adj_pickle(args.adj_sml) if args.adj_sml and os.path.exists(args.adj_sml) else None
    
    # 加载数据
    data, scaler = load_pickle_data(args.data)
    data = create_dataloaders(data, args.batch_size, args.batch_size, args.batch_size)
    
    print(args)
    print(f"Data shapes: x_train={data['x_train'].shape}, y_train={data['y_train'].shape}")
    print(f"Adjacency matrices: conn={adj_conn.shape if adj_conn is not None else None}, "
          f"cor={adj_cor.shape if adj_cor is not None else None}, "
          f"sml={adj_sml.shape if adj_sml is not None else None}")
    
    # 创建训练器
    engine = IST_LSTM_Trainer(
        scaler=scaler,
        in_dim=args.in_dim,
        seq_length=args.seq_length,
        num_nodes=args.num_nodes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lrate=args.learning_rate,
        wdecay=args.weight_decay,
        device=device,
        adj_conn=adj_conn,
        adj_cor=adj_cor,
        adj_sml=adj_sml,
        num_heads=args.num_heads,
        window_size=args.window_size,
        num_layers_main=args.num_layers_main,
        num_layers_sub=args.num_layers_sub,
        pred_steps=args.pred_steps,
        use_dcg=args.use_dcg
    )
    
    # 创建保存目录
    os.makedirs(os.path.dirname(args.save) if os.path.dirname(args.save) else '.', exist_ok=True)
    
    print("Start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    
    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        
        t1 = time.time()
        data['train_loader'].shuffle()
        
        for iter, (x, y) in enumerate(data['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)      # (B, T, N, C)
            trainy = torch.Tensor(y).to(device)      # (B, T_out, N, C)
            
            # 只取第一个特征进行预测
            trainy_target = trainy[..., 0]  # (B, T_out, N)
            
            metrics = engine.train(trainx, trainy_target)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
        
        t2 = time.time()
        train_time.append(t2 - t1)
        
        # Validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        
        s1 = time.time()
        for iter, (x, y) in enumerate(data['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testy = torch.Tensor(y).to(device)
            testy_target = testy[..., 0]
            
            metrics = engine.eval(testx, testy_target)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        
        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)
        
        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, ' \
              'Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse,
                         mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)), flush=True)
        
        torch.save(engine.model.state_dict(), args.save + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")
    
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    
    # Testing
    bestid = np.argmin(his_loss)
    best_path = args.save + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"
    engine.model.load_state_dict(torch.load(best_path, map_location=device))
    
    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))
    
    # 测试集评估
    outputs = []
    realy = torch.Tensor(data['y_test']).to(device)
    realy_target = realy[..., 0]  # (B, T_out, N)
    
    for iter, (x, y) in enumerate(data['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        with torch.no_grad():
            preds = engine.model(testx).squeeze(-1)  # (B, pred_steps, N)
        outputs.append(preds)
    
    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy_target.size(0), ...]
    
    # 反归一化
    yhat = scaler.inverse_transform(yhat)
    
    # 多步预测评估
    pred_steps = min(yhat.size(1), realy_target.size(1))
    
    amae = []
    amape = []
    armse = []
    
    for i in range(pred_steps):
        pred = yhat[:, i, :]
        real = realy_target[:, i, :]
        metrics = metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
    
    log = 'On average over {:d} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(pred_steps, np.mean(amae), np.mean(amape), np.mean(armse)))
    
    torch.save(engine.model.state_dict(), args.save + "_exp" + str(args.expid) + "_best_" + str(round(his_loss[bestid], 2)) + ".pth")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
