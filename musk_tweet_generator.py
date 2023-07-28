import torch, csv
import torch.nn as nn
from torch.nn import functional as PyFun


# Defining the model hyperparameters
batch_size = 16
lr = 0.001
seq_len = 32
max_iterations = 5000
eval_interval = 50
eval_iters = 200
num_embed = 64
num_head = 4
layers_num = 6
dropout = 0.2

# File to write the model logs
logs_filename = "log.txt"


# Function to write to the log file
def write_to_log(text):
    with open(logs_filename, "a") as f:
        f.write(text + "\n")


# Data source: https://www.kaggle.com/datasets/gpreda/elon-musk-tweets
# Extracting the data
file_path = "elon_musk_tweets.csv"
tweets = []
with open(file_path, newline="") as csvfile:
    csv_reader = csv.reader(csvfile)

    # Skip the header row if it exists
    next(csv_reader, None)

    # Retrieve tweet texts
    for tweet in csv_reader:
        tweet_text = tweet[-6]
        # Remove non alphanumeric characters but keep punctuation marks
        tweet_text = "".join(
            [
                i
                for i in tweet_text
                if i.isalnum()
                or i
                in [
                    " ",
                    ".",
                    ",",
                    "!",
                    "?",
                    ":",
                    ";",
                    "'",
                    '"',
                    "-",
                    "(",
                    ")",
                    "[",
                    "]",
                    "{",
                    "}",
                ]
            ]
        )
        # Remove non english characters
        tweet_text = tweet_text.encode("ascii", errors="ignore").decode()
        tweets.append(tweet_text)

# Retrieve characters
characters = sorted(list(set([char for tweet in tweets for char in tweet])))
num_chars = len(characters)

# Now, we must map characters to integers
char_to_int = dict((c, i) for i, c in enumerate(characters))
int_to_char = dict((i, c) for i, c in enumerate(characters))

# Create an encoder and decoder
encoder = lambda text: [char_to_int[char] for char in text]
decoder = lambda encoded_text: "".join([int_to_char[i] for i in encoded_text])

# Now that the encoder + decoder functions have been validated, encode the entire
# text using a torch tensor
# Tranform the tweets into a single string
raw_tweets = " ".join(tweets)
torch_encoded = torch.tensor(encoder(raw_tweets), dtype=torch.int64)

# Train / val split
train_size = int(torch_encoded.shape[0] * 0.8)
val_size = torch_encoded.shape[0] - train_size

# Generating training and validation sets
train_set = torch_encoded[:train_size]
val_set = torch_encoded[train_size : train_size + val_size]

# Fit the data to the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_set = train_set.to(device)
val_set = val_set.to(device)


def generate_batch(source):
    # determine data source
    if source == "train":
        dataset = train_set
    elif source == "val":
        dataset = val_set

    random_characters = torch.randint(len(dataset) - seq_len, (batch_size,))
    contexts = torch.stack([dataset[i : i + seq_len] for i in random_characters])
    targets = torch.stack([dataset[i + 1 : i + seq_len + 1] for i in random_characters])
    return contexts, targets


# Estimating the loss in training and validation and saving it to a dictionary
@torch.no_grad()
def calculate_loss():
    loss_estimates = {}
    model.eval()

    # Calculate the loss for the training and validation sets
    for dataset_type in ["train", "val"]:
        accumulated_losses = torch.zeros(eval_iters)
        for iteration_index in range(eval_iters):
            features, targets = generate_batch(dataset_type)
            _, loss = model(features, targets)
            accumulated_losses[iteration_index] = loss.item()

        loss_estimates[dataset_type] = accumulated_losses.mean()
    model.train()
    return loss_estimates


# Layer normalization class called by the bigramLM and the transformer block
class LayerNorm:
    # Intiializing the layer norm parameters
    def __init__(self, dimension, epsilon=1e-5):
        self.epsilon = epsilon
        self.scale = torch.ones(dimension)
        self.shift = torch.zeros(dimension)

    # Calculating mean and variance across the batch
    def calculate_mean_variance(self, input):
        mean_input = input.mean(1, keepdim=True)
        var_input = input.var(1, keepdim=True)
        return mean_input, var_input

    # Normalizing the input to have unit variance and zero mean
    def normalize_input(self, input, mean_input, var_input):
        input_normalized = (input - mean_input) / torch.sqrt(var_input + self.epsilon)
        return input_normalized

    # Applying the layer norm transformation
    def apply(self, input):
        mean_input, var_input = self.calculate_mean_variance(input)
        input_normalized = self.normalize_input(input, mean_input, var_input)
        self.output = self.scale * input_normalized + self.shift
        return self.output

    # Retrieving the layer norm parameters for backprop
    def get_parameters(self):
        return [self.scale, self.shift]


# Feed forward network that will work in conjunction with the self-attention layer
class FeedFoward(nn.Module):
    # Initializing parameters
    def __init__(self, num_embed):
        super().__init__()
        # Like Karpathy, we multiply the embedding size by 4 to
        # go from 512 to 2048 dimensionality as specified by the self-attention paper
        self.net = nn.Sequential(
            nn.Linear(num_embed, 4 * num_embed),
            nn.ReLU(),
            nn.Linear(4 * num_embed, num_embed),
            nn.Dropout(dropout),
        )

    # Forward pass
    def forward(self, x):
        return self.net(x)


# One head of self-attention
class HeadAttention(nn.Module):
    # Initializing parameters
    def __init__(self, size_head):
        super().__init__()
        # key, query, and value like in the self attention paper
        self.key = nn.Linear(num_embed, size_head, bias=False)
        self.query = nn.Linear(num_embed, size_head, bias=False)
        self.value = nn.Linear(num_embed, size_head, bias=False)
        # dropout to add normalization
        self.dropout = nn.Dropout(dropout)
        # Here, we don't want the tril buffer to be updated by the optimizer, so
        # we register it as a buffer
        self.register_buffer("tril", torch.tril(torch.ones(seq_len, seq_len)))

    # Function to process the weights
    def weight_processing(self, weight):
        weight = weight.masked_fill(self.tril[:, :] == 0, float("-inf"))
        weight = PyFun.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        return weight

    # Forward pass through the self attention head
    def forward(self, x):
        num_channels = x.shape[-1]
        k = self.key(x)
        q = self.query(x)
        # calculate the weights as specified in the self-attention paper
        w = q @ k.transpose(-2, -1) * num_channels**-0.5
        w = self.weight_processing(w)
        v = self.value(x)
        # multiply the weigths by the value
        out = w @ v
        return out


# Parallel multiple heads of self-attention
class MultiHeadAttention(nn.Module):
    def __init__(self, heads_num, size_head):
        super().__init__()
        # The ModuleList allows us to parallelize the heads
        self.heads = nn.ModuleList([HeadAttention(size_head) for _ in range(heads_num)])
        # We add a linear layer to combine the heads and form the output
        self.linear = nn.Linear(num_embed, num_embed)
        # We add a dropout layer to add regularization
        self.dropout = nn.Dropout(dropout)

    # Forward pass through the multiple heads
    def forward(self, x):
        # We concatenate the outcome of the parallel heads
        # we must also concatenate the heads along the last dimension to
        # have (batch_size, num_heads*channels) as the output dimension
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.linear(out))
        return out


# Transformer block that combines the self-attention and feed forward layers
class TransformerBlock(nn.Module):
    # Initializing parameters
    def __init__(self, num_embed, num_head):
        super().__init__()
        # All heads together must have the same dimensionality as the embedding
        size_head = num_embed // num_head
        # We make calls to the multihead attention and feed forward layers
        self.self_attention = MultiHeadAttention(num_head, size_head)
        self.ffwd = FeedFoward(num_embed)
        # We normalize the output of the self-attention layer
        self.layer_norm = nn.LayerNorm(num_embed)

    # Forward pass
    def forward(self, x):
        normalized = self.layer_norm(x)
        # Notice the residual connections that help solve the
        # vanishing gradient problem
        x = x + self.self_attention(normalized)
        normalized = self.layer_norm(x)
        x = x + self.ffwd(normalized)
        return x


# We define a bigram language model
class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(num_chars, num_embed)
        self.position_embedding_table = nn.Embedding(num_chars, num_embed)
        block_sequence = [
            TransformerBlock(num_embed, num_head=num_head) for _ in range(layers_num)
        ]
        self.blocks = nn.Sequential(*block_sequence)
        # We normalize the output of the aggregated transformer blocks
        self.layer_norm = nn.LayerNorm(num_embed)
        # Prediction linear layer
        self.linear = nn.Linear(num_embed, num_chars)

    def forward(self, indexes, targets=None):
        """
        Perform a forward pass through the model.

        Args:
            indexes (torch.Tensor): A tensor containing the indices of the input tokens.
            targets (torch.Tensor, optional): A tensor containing the target tokens. Defaults to None.

        Returns:
            logits (torch.Tensor): The model's predictions for each token in the input.
            loss (torch.Tensor, optional): The cross entropy loss between the logits and the targets, if targets were provided. Defaults to None.
        """

        # Shape of the input tensor
        batch_size, seq_len_curr = indexes.shape
        # Transform the indexes of the input tokens into embeddings
        token_embeddings = self.token_embedding_table(
            indexes
        )  # Shape: (batch_size, seq_length, embedding_dim)

        # Generate a sequence of position indexes and transform them into position embeddings
        position_indexes = torch.arange(seq_len_curr, device=device)
        position_embeddings = self.position_embedding_table(
            position_indexes
        )  # Shape: (seq_length, embedding_dim)

        # Add the token and position embeddings together
        embeddings = (
            token_embeddings + position_embeddings
        )  # Shape: (batch_size, seq_length, embedding_dim)

        # Pass the embeddings through the transformer blocks
        transformer_output = self.blocks(
            embeddings
        )  # Shape: (batch_size, seq_length, embedding_dim)

        # Apply layer normalization to the output of the blocks
        normalized_output = self.layer_norm(
            transformer_output
        )  # Shape: (batch_size, seq_length, embedding_dim)

        # Pass the normalized output through a linear layer to get the logits
        logits = self.linear(
            normalized_output
        )  # Shape: (batch_size, seq_length, vocab_size)

        # If targets are provided, compute the loss; otherwise, set it to None
        loss = None
        # Get the dimensions of the logits tensor
        batch_size, seq_len_curr, _ = logits.shape
        if targets is not None:
            # Flatten the logits and targets tensors for the loss computation
            flattened_logits = logits.view(batch_size * seq_len_curr, -1)
            flattened_targets = targets.view(batch_size * seq_len_curr)

            # Compute the cross entropy loss between the logits and the targets
            loss = PyFun.cross_entropy(flattened_logits, flattened_targets)

        # Return the logits and the loss
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last seq_len tokens
            idx_cond = idx[:, -seq_len:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = PyFun.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# Generate the model
model = BigramLM().to(device)

# create a PyTorch optimizer - we use AdamW and not Adam since we use weight decay
# to regularize the model parameters (see https://arxiv.org/abs/1711.05101)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


def train_model(model, optimizer, max_iterations, eval_interval):
    """
    Train the model for a specified number of iterations.

    Args:
        model: The model to be trained.
        optimizer: The optimizer to be used for training.
        max_iterations (int): The number of iterations to train for.
        eval_interval (int): The number of iterations between evaluations of the loss on the training and validation sets.
    """
    # Loop over the specified number of iterations
    for iteration in range(max_iterations):
        # At each eval_interval, or at the final iteration, compute and display the training and validation losses
        if iteration % eval_interval == 0 or iteration == max_iterations - 1:
            losses = calculate_loss()
            write_to_log(
                str(f"Iteration {iteration}: Training loss {losses['train']:.5f}")
            )

        # Generate a batch of training data
        input_batch, target_batch = generate_batch("train")

        # Compute the model's predictions and loss for the current batch
        _, loss = model(input_batch, target_batch)

        # Before calculating the gradients, we need to ensure that they are zero.
        # set_to_none=True is more efficient than using zero_grad(), according to the PyTorch docs.
        optimizer.zero_grad(set_to_none=True)

        # Compute the gradients for the model parameters
        loss.backward()

        # Update the model parameters
        optimizer.step()


# Model validation
def validate_model(model):
    """
    Compute and prints the loss on the validation set.

    Args:
        model: The model to be evaluated.
    """
    losses = calculate_loss()
    write_to_log(f"Validation loss {losses['val']:.5f}")


# Train the model
train_model(model, optimizer, max_iterations, eval_interval)

# Validate the model
validate_model(model)

# example model output
context = torch.zeros((1, 1), dtype=torch.long, device=device)
write_to_log(decoder(model.generate(context, max_new_tokens=200)[0].tolist()))
