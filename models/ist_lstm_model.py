import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque


class WindowAttention(nn.Module):
    """基于窗口的多头自注意力 (W-MSA / SW-MSA)"""
    def __init__(self, dim, num_heads=4, window_size=2, shift_size=0, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, N, T, C) -> 沿时间维度T做窗口注意力
        """
        B, N, T, C = x.shape
        
        # 循环移位 (仅用于SW-MSA)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=-self.shift_size, dims=2)
        
        # 填充使T能被window_size整除
        pad_t = (self.window_size - T % self.window_size) % self.window_size
        if pad_t > 0:
            x = F.pad(x, (0, 0, 0, pad_t))
        
        _, _, T_pad, _ = x.shape
        n_win = T_pad // self.window_size
        
        # reshape为 (B, N, n_win, window_size, C)
        x_win = x.view(B, N, n_win, self.window_size, C)
        
        # 对每个窗口内做多头自注意力
        # 合并B和N维度: (B*N*n_win, window_size, C)
        x_win = x_win.reshape(B * N * n_win, self.window_size, C)
        
        qkv = self.qkv(x_win).reshape(B * N * n_win, self.window_size, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*N*n_win, num_heads, window_size, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B * N * n_win, self.window_size, C)
        out = self.proj(out)
        out = self.dropout(out)
        
        # reshape回 (B, N, T_pad, C)
        out = out.reshape(B, N, n_win, self.window_size, C).reshape(B, N, T_pad, C)
        
        # 移除填充
        if pad_t > 0:
            out = out[:, :, :T, :]
        
        # 逆循环移位
        if self.shift_size > 0:
            out = torch.roll(out, shifts=self.shift_size, dims=2)
        
        return out


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block，用于替代LSTM输入门"""
    def __init__(self, dim, num_heads=4, window_size=2, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = WindowAttention(dim, num_heads, window_size, shift_size=0, dropout=dropout)
        self.attn2 = WindowAttention(dim, num_heads, window_size, shift_size=window_size // 2, dropout=dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        x: (B, N, T, C)
        """
        # W-MSA + 残差
        shortcut = x
        x = self.norm1(x)
        x = self.attn1(x) + shortcut
        
        # SW-MSA + 残差
        shortcut = x
        x = self.norm1(x)
        x = self.attn2(x) + shortcut
        
        # MLP + 残差
        shortcut = x
        x = self.norm2(x)
        B, N, T, C = x.shape
        x = x.reshape(B * N * T, C)
        x = self.mlp(x)
        x = x.reshape(B, N, T, C)
        x = x + shortcut
        
        return x


class GraphConv(nn.Module):
    """简单的图卷积层"""
    def __init__(self, c_in, c_out, dropout=0.3):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))
        self.dropout = dropout
    
    def forward(self, x, adj):
        """
        x: (B, C, N, T)
        adj: (N, N)
        """
        x = torch.einsum('bcnt,nm->bcmt', x, adj)
        x = self.mlp(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class DCGGraph(nn.Module):
    """
    扩散与汇聚图 (Diffusion and Convergence Graph)
    A = alpha * D + beta * T + gamma * F
    """
    def __init__(self, num_nodes, adj_conn=None, adj_cor=None, adj_sml=None):
        super().__init__()
        self.num_nodes = num_nodes
        
        # 可学习的融合权重 (经softmax归一化)
        self.weights = nn.Parameter(torch.ones(3), requires_grad=True)
        
        # 深度关系矩阵 D: 基于DFS构建
        if adj_conn is not None:
            self.register_buffer('D', self._build_depth_matrix(adj_conn))
        else:
            self.register_buffer('D', torch.eye(num_nodes))
        
        # 旅行距离/相关性权重矩阵 T
        if adj_cor is not None:
            T = torch.from_numpy(adj_cor).float() if isinstance(adj_cor, np.ndarray) else adj_cor.float()
            self.register_buffer('T', T)
        else:
            self.register_buffer('T', torch.eye(num_nodes))
        
        # 乘客流量权重矩阵 F
        if adj_sml is not None:
            F_mat = torch.from_numpy(adj_sml).float() if isinstance(adj_sml, np.ndarray) else adj_sml.float()
            self.register_buffer('F', F_mat)
        else:
            self.register_buffer('F', torch.eye(num_nodes))
    
    def _build_depth_matrix(self, adj):
        """基于BFS构建深度关系矩阵"""
        if isinstance(adj, np.ndarray):
            adj = torch.from_numpy(adj).float()
        
        N = adj.shape[0]
        adj_np = adj.cpu().numpy() if isinstance(adj, torch.Tensor) else adj
        adj_binary = (adj_np > 0).astype(int)
        
        # 使用BFS计算深度连接矩阵
        depth_conn = np.zeros((N, N))
        for start in range(N):
            visited = [False] * N
            queue = deque([(start, 0)])
            visited[start] = True
            while queue:
                node, depth = queue.popleft()
                if depth > 0:
                    depth_conn[start, node] = 1.0 / (depth + 1)
                for neighbor in range(N):
                    if adj_binary[node, neighbor] > 0 and not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append((neighbor, depth + 1))
        
        # 深度关系矩阵: (D_connect + I)^{-1}
        depth_conn = torch.from_numpy(depth_conn).float()
        D_connect = depth_conn + torch.eye(N)
        try:
            D_inv = torch.inverse(D_connect)
        except:
            D_inv = torch.pinverse(D_connect)
        
        return D_inv
    
    def get_adj(self):
        """获取复合邻接矩阵"""
        w = F.softmax(self.weights, dim=0)
        adj = w[0] * self.D + w[1] * self.T + w[2] * self.F
        
        # 归一化
        row_sum = adj.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1
        adj = adj / row_sum
        
        return adj


class IST_LSTM_Cell(nn.Module):
    """
    IST-LSTM预测单元
    用Swin Transformer Block替代传统输入门
    遗忘门、候选细胞状态、输出门保留标准LSTM
    """
    def __init__(self, input_dim, hidden_dim, num_nodes, num_heads=4, 
                 window_size=2, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        
        # Swin Transformer Block 替代输入门
        # STB处理时间序列，输入维度为 input_dim + hidden_dim (拼接x_t和h_{t-1})
        self.stb = SwinTransformerBlock(
            dim=input_dim + hidden_dim, 
            num_heads=num_heads, 
            window_size=max(2, window_size),
            dropout=dropout
        )
        
        # 输入门投影
        self.input_gate_proj = nn.Linear(input_dim + hidden_dim, hidden_dim)
        
        # 遗忘门、候选状态、输出门 (标准LSTM)
        self.W_f = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_c = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_o = nn.Linear(input_dim + hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_t, h_prev, c_prev):
        """
        x_t: (B, N, C_in)  当前时间步输入特征
        h_prev: (B, N, C_hidden)  上一时刻隐藏状态
        c_prev: (B, N, C_hidden)  上一时刻细胞状态
        """
        B, N, C_in = x_t.shape
        
        # 拼接当前输入和前一隐藏状态
        concat = torch.cat([x_t, h_prev], dim=-1)  # (B, N, C_in + C_hidden)
        
        # ---- 输入门: Swin Transformer Block ----
        # 为了STB需要至少window_size个时间步，复制当前输入
        concat_stb = concat.unsqueeze(2)  # (B, N, 1, C_in + C_hidden)
        # 拼接当前和前一时刻信息，构造短期序列给STB
        stb_input = torch.cat([concat_stb, concat_stb], dim=2)  # (B, N, 2, ...)
        stb_out = self.stb(stb_input)  # (B, N, 2, C_in + C_hidden)
        stb_out = stb_out[:, :, -1, :]  # 取最后一个时间步 (B, N, C_in + C_hidden)
        
        i_t = torch.sigmoid(self.input_gate_proj(stb_out))  # (B, N, C_hidden)
        
        # ---- 遗忘门 ----
        f_t = torch.sigmoid(self.W_f(concat))  # (B, N, C_hidden)
        
        # ---- 候选细胞状态 ----
        c_tilde = torch.tanh(self.W_c(concat))  # (B, N, C_hidden)
        
        # ---- 细胞状态更新 ----
        c_t = f_t * c_prev + i_t * c_tilde  # (B, N, C_hidden)
        
        # ---- 输出门 ----
        o_t = torch.sigmoid(self.W_o(concat))  # (B, N, C_hidden)
        h_t = o_t * torch.tanh(c_t)  # (B, N, C_hidden)
        
        return h_t, c_t


class LSTM_Cell(nn.Module):
    """标准LSTM Cell (用于主分支)"""
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.W_i = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_f = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_c = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_o = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, h_prev, c_prev):
        """
        x: (B, N, C_in)
        h_prev: (B, N, C_hidden)
        c_prev: (B, N, C_hidden)
        """
        concat = torch.cat([x, h_prev], dim=-1)
        i_t = torch.sigmoid(self.W_i(concat))
        f_t = torch.sigmoid(self.W_f(concat))
        c_tilde = torch.tanh(self.W_c(concat))
        o_t = torch.sigmoid(self.W_o(concat))
        
        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, c_t


class IST_LSTM(nn.Module):
    """
    IST-LSTM 双分支网络
    主分支: 3层标准LSTM，聚焦断面客流模式
    次分支: 2层IST-LSTM，编码全局上下文信息(OD驱动的扩散模态)
    """
    def __init__(self, num_nodes, in_dim=2, hidden_dim=64, out_dim=1,
                 num_layers_main=3, num_layers_sub=2,
                 num_heads=4, window_size=2, dropout=0.3,
                 adj_conn=None, adj_cor=None, adj_sml=None,
                 use_dcg=True):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers_main = num_layers_main
        self.num_layers_sub = num_layers_sub
        self.use_dcg = use_dcg
        
        # DCG图结构
        if use_dcg:
            self.dcg = DCGGraph(num_nodes, adj_conn, adj_cor, adj_sml)
            self.graph_conv = GraphConv(in_dim, in_dim, dropout)
        
        # 输入投影
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # ---- 主分支: 标准LSTM ----
        self.main_lstms = nn.ModuleList([
            LSTM_Cell(hidden_dim, hidden_dim, dropout) 
            for _ in range(num_layers_main)
        ])
        
        # ---- 次分支: IST-LSTM ----
        self.sub_lstms = nn.ModuleList([
            IST_LSTM_Cell(hidden_dim, hidden_dim, num_nodes, num_heads, window_size, dropout)
            for _ in range(num_layers_sub)
        ])
        
        # 时间差分学习投影 (从主分支第三层LSTM导出直接经验信号)
        self.diff_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 主分支状态注入次分支的投影
        self.main_to_sub_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 融合与输出层
        fusion_dim = hidden_dim * 2  # 主分支 + 次分支
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.output_proj = nn.Linear(hidden_dim // 2, out_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: (B, T, N, C_in) 或 (B, C_in, N, T)
        return: (B, N, out_dim)
        """
        # 统一输入格式为 (B, T, N, C)
        if x.dim() == 4:
            if x.size(1) == self.in_dim and x.size(2) == self.num_nodes:
                # (B, C, N, T) -> (B, T, N, C)
                x = x.permute(0, 3, 2, 1)
        
        B, T, N, C = x.shape
        
        # ---- DCG图卷积 (空间特征提取) ----
        if self.use_dcg:
            adj = self.dcg.get_adj()  # (N, N)
            x_gcn = x.permute(0, 3, 2, 1)  # (B, C, N, T)
            x_gcn = self.graph_conv(x_gcn, adj)
            x = x_gcn.permute(0, 3, 2, 1)  # (B, T, N, C)
        
        # 输入投影
        x = self.input_proj(x)  # (B, T, N, hidden_dim)
        
        # ==================== 主分支: 3层LSTM ====================
        h_main = [torch.zeros(B, N, self.hidden_dim, device=x.device) for _ in range(self.num_layers_main)]
        c_main = [torch.zeros(B, N, self.hidden_dim, device=x.device) for _ in range(self.num_layers_main)]
        
        main_outputs = []
        for t in range(T):
            x_t = x[:, t, :, :]  # (B, N, hidden_dim)
            for l in range(self.num_layers_main):
                h_main[l], c_main[l] = self.main_lstms[l](x_t, h_main[l], c_main[l])
                x_t = h_main[l]
            main_outputs.append(x_t)
        
        h_main_final = h_main[-1]  # (B, N, hidden_dim) 最后一层最后一个时间步
        
        # 时间差分学习: 从主成分特征中导出直接经验信号
        if T > 1:
            diff = main_outputs[-1] - main_outputs[-2]  # (B, N, hidden_dim)
        else:
            diff = torch.zeros_like(main_outputs[-1])
        diff_signal = torch.tanh(self.diff_proj(diff))  # (B, N, hidden_dim)
        
        # 将主分支学习到的状态注入到次分支
        main_injected = torch.tanh(self.main_to_sub_proj(h_main_final))
        
        # ==================== 次分支: 2层IST-LSTM ====================
        # 次分支输入: 原始流量 + 主分支注入信号 + 差分信号
        x_sub = x  # (B, T, N, hidden_dim)
        
        h_sub = [torch.zeros(B, N, self.hidden_dim, device=x.device) for _ in range(self.num_layers_sub)]
        c_sub = [torch.zeros(B, N, self.hidden_dim, device=x.device) for _ in range(self.num_layers_sub)]
        
        for t in range(T):
            x_t = x_sub[:, t, :, :]  # (B, N, hidden_dim)
            # 在第一个时间步加入注入信号
            if t == T - 1:
                x_t = x_t + main_injected + diff_signal
            
            for l in range(self.num_layers_sub):
                h_sub[l], c_sub[l] = self.sub_lstms[l](x_t, h_sub[l], c_sub[l])
                x_t = h_sub[l]
        
        h_sub_final = h_sub[-1]  # (B, N, hidden_dim)
        
        # ==================== 融合与输出 ====================
        fused = torch.cat([h_main_final, h_sub_final], dim=-1)  # (B, N, hidden_dim*2)
        fused = self.fusion(fused)  # (B, N, hidden_dim//2)
        fused = self.dropout(fused)
        
        out = self.output_proj(fused)  # (B, N, out_dim)
        
        return out


class IST_LSTM_Predictor(nn.Module):
    """
    IST-LSTM 预测器封装
    适配多步预测 (输出未来多个时间步)
    """
    def __init__(self, num_nodes, in_dim=2, hidden_dim=64, out_dim=1,
                 num_layers_main=3, num_layers_sub=2,
                 num_heads=4, window_size=2, dropout=0.3,
                 adj_conn=None, adj_cor=None, adj_sml=None,
                 use_dcg=True, pred_steps=1):
        super().__init__()
        self.pred_steps = pred_steps
        self.out_dim = out_dim
        
        self.encoder = IST_LSTM(
            num_nodes=num_nodes,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers_main=num_layers_main,
            num_layers_sub=num_layers_sub,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout,
            adj_conn=adj_conn,
            adj_cor=adj_cor,
            adj_sml=adj_sml,
            use_dcg=use_dcg
        )
        
        # 多步预测头
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_steps * out_dim)
        )
    
    def forward(self, x):
        """
        x: (B, T, N, C_in) 或 (B, C_in, N, T)
        return: (B, pred_steps, N, out_dim)
        """
        h = self.encoder(x)  # (B, N, hidden_dim)
        out = self.pred_head(h)  # (B, N, pred_steps * out_dim)
        B, N, _ = out.shape
        out = out.view(B, N, self.pred_steps, self.out_dim)
        out = out.permute(0, 2, 1, 3)  # (B, pred_steps, N, out_dim)
        return out
