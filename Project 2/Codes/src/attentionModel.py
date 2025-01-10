import torch.nn as nn
import torch 
import math
import torch.nn.functional as F



class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, attention_dim):
        super(SelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.W_q = nn.Linear(embedding_dim, attention_dim)
        self.W_k = nn.Linear(embedding_dim, attention_dim)
        self.W_v = nn.Linear(embedding_dim, attention_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, embedding_dim)
        Q = self.W_q(x)  # Queries
        K = self.W_k(x)  # Keys
        V = self.W_v(x)  # Values
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.attention_dim)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Compute the context vector
        context = torch.matmul(attention_weights, V)
        return context, attention_weights




class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim, attention_dim, context_length):
        super(CausalSelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.W_q = nn.Linear(embedding_dim, attention_dim)
        self.W_k = nn.Linear(embedding_dim, attention_dim)
        self.W_v = nn.Linear(embedding_dim, attention_dim)
        
        
        # Register a buffer for the causal mask
        self.register_buffer(  'mask',torch.tril(torch.ones(context_length, context_length)).unsqueeze(0))

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.attention_dim)
        
        # Apply the causal mask
        mask = self.mask[:, :seq_length, :seq_length]
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        return context, attention_weights



class MultiHeadCausalAttention(nn.Module):
    def __init__(self, embedding_dim, attention_dim, num_heads, context_length, dropout=0.1):
        super(MultiHeadCausalAttention, self).__init__()
        assert attention_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads

        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.W_q = nn.Linear(embedding_dim, attention_dim)
        self.W_k = nn.Linear(embedding_dim, attention_dim)
        self.W_v = nn.Linear(embedding_dim, attention_dim)
        self.fc_out = nn.Linear(attention_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer( 'mask',torch.tril(torch.ones(context_length, context_length)).unsqueeze(0))
   
    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Split the embedding into self.num_heads different pieces
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        #print("Q,K,V: ", Q.shape, K.shape, V.shape)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        #print("scores: ", scores.shape)
        # Apply the causal mask
        mask = self.mask[:, :seq_length, :seq_length]
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        #print("attention weights, values: ", attention_weights.shape, V.shape)
        context = torch.matmul(attention_weights, V)
        #print("context: ", context.shape)
        context = context.transpose(1, 2).contiguous()
        #print("context after transpose: ", context.shape)
        context = context.view(batch_size, seq_length, self.attention_dim)


        out = self.fc_out(context)
        return out, attention_weights



class LanguageModelWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, attention_dim, context_length, hidden_dim=256, num_heads=8, dropout=0.2):
        super(LanguageModelWithAttention, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.attention = MultiHeadCausalAttention(embedding_dim, attention_dim, num_heads, context_length, dropout)
        
        # Adding more layers and non-linear activation functions
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        token_embeds = self.token_embedding(x)
        position_embeds = self.position_embedding(positions)
        
        embeddings = token_embeds + position_embeds
        embeddings = self.dropout(embeddings)
        
        # Apply the attention layer
        context, attention_weights = self.attention(embeddings)
        
        # Apply the feedforward layers
        out = self.fc1(context)
        out = self.relu(out)
        logits = self.fc2(out)
        return logits, attention_weights


def generate_text_attention(model, tokenizer, start_text,device, context_length=32, temperature=1.0):
    model.eval()
    generated = tokenizer.encode(start_text)
    context = torch.tensor(generated, dtype=torch.long,
                          device=device).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(context_length):
            if context.size(1) >= context_length:
                break
            logits, _ = model(context)
            next_token_logits = logits[0, -1, :] / temperature
            probabilities = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probabilities, num_samples=1)
            context = torch.cat(
                [context, next_token_id.unsqueeze(0)], dim=1
            )
    
    generated_text = tokenizer.decode(context[0].tolist())
    return generated_text



class MultiHeadCausalAttentionCorrect(nn.Module):
    def __init__(self, embedding_dim, attention_dim, num_heads, context_length, dropout=0.1):
        super(MultiHeadCausalAttentionCorrect, self).__init__()
        assert attention_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads

        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.W_q = nn.Linear(embedding_dim, attention_dim)
        self.W_k = nn.Linear(embedding_dim, attention_dim)
        self.W_v = nn.Linear(embedding_dim, attention_dim)
        self.fc_out = nn.Linear(attention_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer( 'mask',torch.tril(torch.ones(context_length, context_length)).unsqueeze(0))

    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split the embedding into self.num_heads different pieces
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        #print("Q,K,V: ", Q.shape, K.shape, V.shape)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        #print("scores: ", scores.shape)
        # Apply the causal mask
        mask = self.mask[:, :seq_length, :seq_length]
        scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        #print("attention weights, values: ", attention_weights.shape, V.shape)
        context = torch.matmul(attention_weights, V)
        #print("context: ", context.shape)

        context = context.sum(dim=1)
        context = context.repeat_interleave(self.num_heads, dim=-1)

        out = self.fc_out(context)
        return out, attention_weights