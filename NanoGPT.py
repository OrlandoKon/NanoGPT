import torch
import torch.nn as nn
import torch.nn.functional as F

# 超参数
context_length = 8
batch_size = 32
learning_rate = 1e-3
embed_dim = 32
eval_interval = 300
eval_iters = 200
max_iters = 3000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters) 
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item
        out[split] = losses.mean()
    model.train()
    return out


# 构建BigramLanguageModel
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_tale = nn.Embedding(vocab_size, embed_dim)             # (B, T, C)
        self.position_embedding_table = nn.Embedding(context_length, embed_dim)     # (B, T, C)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emb = self.token_embedding_tale(idx)                  
        position_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (B, T, C)
        x = token_emb + position_emb
        logits = self.lm_head(x)

        if targets == None:
            return logits
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

            return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(idx)

            logits = logits[:, -1, :]
            prob = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(prob, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

model = BigramLanguageModel().to(device)

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
print(loss.item())

context = torch.zeros((1, 1), dtype=torch.long)
print(decoder(model.generate(context, max_new_tokens=500)[0].tolist()))