import torch

# Create a simple example showing positional encoding slicing
d_model = 6  # Embedding dimension, must be even
seq_len = 4  # Sequence length

# Create position angles
def get_position_angle_vec(position, d_model):
    # Create angle rates
    angle_rates = 1 / torch.pow(10000, (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model))
    # Initialize with zeros
    result = torch.zeros(d_model)
    # Fill even indices with position * angle_rates
    result[0::2] = position * angle_rates
    # The odd indices will be filled with cos later
    return result

# Initialize position encoding matrix
position_encodings = torch.zeros(seq_len, d_model)

for pos in range(seq_len):
    position_encodings[pos] = get_position_angle_vec(pos, d_model)

# Print the raw angles before applying sin/cos
print("Raw position angles:")
print(position_encodings)
# print(position_encodings[:, 0::2])
# print(position_encodings[:, 1::2])
# Save a copy for comparison
original_angles = position_encodings.clone()

# Apply sin to even indices and cos to odd indices
position_encodings[:, 0::2] = torch.sin(position_encodings[:, 0::2])  # 0, 2, 4, ...
position_encodings[:, 1::2] = torch.cos(position_encodings[:, 1::2])  # 1, 3, 5, ...

# Print the final position embeddings
print("Final sinusoidal position embeddings:")
print(position_encodings)
print()


def compute_theta(dim: int, base: float = 10000.0, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    计算旋转位置编码中的 Theta 角度值。

    参数：
    - d (int): 嵌入向量的维度（必须为偶数）。
    - base (float): 基础频率参数, 默认为10000.0。
    - device (torch.device): 计算设备, 默认为CPU。

    返回：
    - torch.Tensor: 包含Theta值的1D张量, 形状为 [d/2]。
    """
    if dim % 2 != 0:
        print("嵌入维度 dim 必须为偶数")
    i = torch.arange(1, (dim//2) + 1, dtype=torch.float32, device=device)
    theta_i = base ** (-2*(i - 1) / dim)

    return theta_i

def precompute_freqs_cis(dim: int, seq_len: int, base: float = 10000.0, device: torch.device = torch.device('cpu')):
    theta = compute_theta(dim, base, device) # theta 角度值序列，向量, 大小为 dim // 2
    m = torch.arange(seq_len, device=device) # # token 位置值序列，向量，大小为 seq_len
    m_theta = torch.outer(m, theta) # 所有 token 位置的所有 Theta 值范围, 矩阵，尺寸为 [seq_len, dim // 2]
    freqs_cis = torch.polar(torch.ones_like(m_theta), m_theta) # e^{i*m*\theta}，本质上是旋转矩阵
    return freqs_cis

def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert ndim >= 2
    assert freqs_cis.shape == (x.shape[1],x.shape[-1]), "the last two dimension of freqs_cis, x must match"
    shape = [d if i==1 or i==ndim-1 else 1 for i,d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, device: torch.device = torch.device('cpu')):
    """
    参数:
        - x_q(torch.Tensor): 实际上是权重 W_q * 词嵌入向量值, 来自上一个线性层的输出, 形状为 [batch_size, seq_len, n_heads, head_dim]
        - x_k(torch.Tensor): 实际上是权重 W_k * 词嵌入向量值, 来自上一个线性层的输出, 形状为 [batch_size, seq_len, n_heads, head_dim]
        - freqs_cis (torch.Tensor): 频率复数张量, 形状为 [max_seq_len, head_dim]
    返回:
        - Tuple[torch.Tensor, torch.Tensor]: 旋转编码后的查询和键张量
    """
    # 实数域张量转为复数域张量
    xq_reshape = xq.reshape(*xq.shape[:-1], -1, 2) # [batch_size, seq_len, dim] -> [batch_size, seq_len, dim//2, 2] 
    xk_reshape = xk.reshape(*xk.shape[:-1], -1, 2) # [batch_size, seq_len, dim] -> [batch_size, seq_len, dim//2, 2] 
    xq_complex = torch.view_as_complex(xq_reshape) # 复数形式张量
    xk_complex = torch.view_as_complex(xk_reshape) # 复数形式张量

    # 旋转矩阵（freqs_cis）的维度在序列长度（seq_len，维度 1）和头部维度（head_dim，维度 3）上需要与嵌入的维度一致。
    # 此外，freqs_cis 的形状必须与 xq 和 xk 相匹配，因此我们需要将 freqs_cis 的形状从 [max_seq_len, head_dim] 调整为 [1, max_seq_len, 1, head_dim]。
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_complex) # [max_seq_len, 1, 1, dim // 2]

    # 应用旋转操作，并将结果转回实数域。# flatten(2) 将后面两个维度压成一个维度
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(3) 
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# Demo parameters
batch_size = 1
seq_len = 10
embed_dim = 8  # must be even
head_dim = embed_dim // 2

# Create sample query and key vectors
torch.manual_seed(42)  # 固定随机种子以便结果可复现
sample_query = torch.randn(batch_size, seq_len, embed_dim)
sample_key = torch.randn(batch_size, seq_len, embed_dim)

# Compute RoPE frequencies
freqs_cis = precompute_freqs_cis(embed_dim, seq_len)
print(f"freqs_cis.shape: {freqs_cis.shape}")

# Apply RoPE to query and key
rotated_query, rotated_key = apply_rotary_emb(sample_query, sample_key, freqs_cis)


# Compute attention scores before and after RoPE
attn_before = torch.matmul(sample_query, sample_key.transpose(-2, -1)) / (embed_dim ** 0.5)
attn_after = torch.matmul(rotated_query, rotated_key.transpose(-2, -1)) / (embed_dim ** 0.5)

print(attn_before)
print(attn_after)
