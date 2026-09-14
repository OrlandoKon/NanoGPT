import torch
import torch.nn as nn
import torch.nn.functional as F

# 超参数
batch_size = 64
context_length = 256
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4
n_embd = 384
n_head = 6
n_layer = 6
drop_rate = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# --------------

torch.manual_seed(1337)
# 下载输入文件（莎士比亚作品全文）
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# 打开输入文件
with open('./data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 计算词典
vocabulary = sorted(list(set(text)))
vocab_size = len(vocabulary)

# Tokenizer，将输入映射成为一张词汇表，由编码器和解码器组成
## 1. 构建String To Integer 的映射字典
stoi = { ch : i for i, ch in enumerate(vocabulary) }
## 2. 构建Integer To String 的映射字段
itos = { i : ch for i, ch in enumerate(vocabulary) }
## 3. Encoder 将输入的字符串转化为Token List
def encoder(x):
    return [stoi[ch] for ch in x]
## 4. Decoder 将Token List 转化为人能读的字符串
def decoder(digits_list):
    return ''.join([itos[i] for i in digits_list])

# 将输入文件加载为张量
data = torch.tensor(encoder(text), dtype=torch.long)
# 划分训练集和测试集
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data

    ix = torch.randint(len(data) - context_length, (batch_size,))
    xb = torch.stack([data[i:i + context_length] for i in ix])
    yb = torch.stack([data[i + 1: i + context_length + 1] for i in ix])

    x = xb.to(device)
    y = yb.to(device)

    return x, y

# 跨批次的损失
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) 
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 构建单头注意力机制
class Head(nn.Module):
    def __init__(self, head_size, drop_rate):
        super().__init__()

        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.drop = nn.Dropout(drop_rate)
        self.register_buffer('triu', torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1))

    def forward(self, x):
        _, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.triu[:T, :T], float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.drop(wei)

        out = wei @ v

        return out

# 多头自注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size, drop_rate):
        super().__init__()
        ## 多头自注意力机制
        self.heads = nn.ModuleList([Head(head_size, drop_rate) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1) ## 以Token的C维度进行拼接
        out = self.proj(out)
        out = self.drop(out)

        return out

# 前馈神经网络：用于整理和提取当前Token从其他Token中获取到的信息
class FeedForwardNet(nn.Module):
    def __init__(self, n_embd, drop_rate):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        return self.net(x)

# Transformer Core Block: Communication and Computation
class Block(nn.Module):
    def __init__(self, n_embd, n_head, drop_rate):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head, drop_rate)
        self.ffn = FeedForwardNet(n_embd, drop_rate)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x
         

# 构建BigramLanguageModel
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, context_length, n_head, n_layer, drop_rate):
        super().__init__()

        self.token_embedding_tale = nn.Embedding(vocab_size, n_embd)             # (B, T, C)
        self.position_embedding_table = nn.Embedding(context_length, n_embd)     # (B, T, C)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, drop_rate) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emb = self.token_embedding_tale(idx)                  
        position_emb = self.position_embedding_table(torch.arange(T, device=device)) # (B, T, C)
        x = token_emb + position_emb
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_length:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]
            prob = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(prob, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

model = BigramLanguageModel(vocab_size, n_embd, context_length, n_head, n_layer, drop_rate).to(device)

# 设置优化器为AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 训练model
for iter in range(max_iters):
    # 跨批次评估损失
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    # 获取样本
    xb, yb = get_batch('train')
    # 计算logits和loss
    logits, loss = model(xb, yb)
    # 清空上一步的梯度
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 更新参数
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decoder(model.generate(context, max_new_tokens=500)[0].tolist()))