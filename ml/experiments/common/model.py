import torch
from torch import nn
from torch.nn.parameter import Parameter


def create_batch_of_identity(batch_size, n):
    x = torch.eye(n)
    x = x.reshape((1, n, n))
    y = x.repeat(batch_size, 1, 1)
    return y


def batch_to_batch_onehot_tensor(batch, vocab_size):
    # shape: B x seq x sent x sent x V x V
    sequence_len = batch.size()[1]
    sentence_len = batch.size()[2]
    batch_size = batch.size()[0]
    batch_onehot_tensor = torch.zeros((batch_size, sequence_len, sentence_len, 
                                   sentence_len, vocab_size, vocab_size))
    for b in range(batch_size):
        for seq in range(sequence_len):
            for l1 in range(sentence_len):
                for l2 in range(sentence_len):
                    key = batch[b][seq][l1]
                    val = batch[b][seq][l2]
                    batch_onehot_tensor[b][seq][l1][l2][key][val] = 1
    return batch_onehot_tensor


def batch_to_batch_onehot_tensor2(batch, vocab_size):
    # shape: B x seq x sent x V x V
    sequence_len = batch.size()[1]
    sentence_len = batch.size()[2]
    batch_size = batch.size()[0]
    batch_onehot_tensor2 = torch.zeros((batch_size, sequence_len, sentence_len, 
                                       vocab_size, vocab_size))
    for b in range(batch_size):
        for seq in range(sequence_len):
            for l in range(sentence_len):
                key = batch[b][seq][l]
                batch_onehot_tensor2[b][seq][l][key][key] = 1
    return batch_onehot_tensor2


class MemoryNetwork(nn.Module):
    def __init__(self, sentence_len, vocab_size, n_line_types):
        super().__init__()
        self.n_line_types = n_line_types
        self.vocab_size = vocab_size
        self.a = Parameter(torch.zeros((n_line_types, sentence_len * sentence_len,), requires_grad=True))
        self.b = Parameter(torch.zeros((n_line_types, sentence_len,), requires_grad=True))
    
    def forward(self, batch, line_types):
        batch_onehot_tensor = batch_to_batch_onehot_tensor(batch, self.vocab_size)    # B x seq x sent x sent x V x V
        batch_onehot_tensor2 = batch_to_batch_onehot_tensor2(batch, self.vocab_size)  # B x seq x sent x V x V

        seq_len = batch.size()[1]
        sentence_len = batch.size()[2]
        batch_size = batch.size()[0]
        memory = torch.zeros((batch_size, vocab_size, vocab_size))
        identity = create_batch_of_identity(batch_size, vocab_size)
        ones = torch.ones((batch_size, vocab_size, vocab_size))

        for i in range(seq_len):
            a = self.a.view(1, self.n_line_types, sentence_len, sentence_len, 1, 1).repeat(batch_size, 1, 1, 1, 1, 1)
            a = a[torch.arange(batch_size), line_types[:, i]]
            b = self.b.view(1, self.n_line_types, sentence_len, 1, 1).repeat(batch_size, 1, 1, 1, 1)
            b = b[torch.arange(batch_size), line_types[:, i]]
            write_to_memory = torch.sum(torch.tanh(a) * batch_onehot_tensor[:, i], axis=[1, 2])
            modified_identity = identity - torch.minimum(torch.sum(torch.sigmoid(b) * batch_onehot_tensor2[:, i], axis=1), ones)
            remember = torch.bmm(modified_identity, memory)
            # bmm: batch matrix multiplication
            memory = write_to_memory + remember
        return torch.mean(memory)


if __name__ == '__main__':
    batch_size = 32
    batch = torch.tensor([[[1, 2, 3, 4, 5, 6, 7],
                           [3, 4, 5, 6, 7, 8, 9]] * 7] * batch_size, dtype=torch.long)
    line_types = torch.tensor([[0, 1] * 7] *batch_size, dtype=torch.long)
    sequence_len = batch.size()[1]
    sentence_len = batch.size()[2]
    batch_size = batch.size()[0]
    vocab_size = 23
    n_line_types = 2

    net = MemoryNetwork(sentence_len, vocab_size, n_line_types)
    opt = torch.optim.Adam(net.parameters())
    loss = net(batch, line_types)
    opt.zero_grad()
    loss.backward()
    opt.step()