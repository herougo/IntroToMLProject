import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from ml.metrics.pytorch_metrics import ClassAccuracyMetric
from ml.utils.utils import flatten_dict_of_dict


def create_batch_of_identity(batch_size, n):
    x = torch.eye(n)
    x = x.reshape((1, n, n))
    y = x.repeat(batch_size, 1, 1)
    return y


def batch_to_batch_onehot_tensor(batch, vocab_size):
    # output shape: B x seq x sent x sent x V x V
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
    # output shape: B x seq x sent x V x V
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
    
    def forward(self, sentence_sequences, questions, line_types):
        raise NotImplementedError()

        batch_onehot_tensor = batch_to_batch_onehot_tensor(sentence_sequences, self.vocab_size)    # B x seq x sent x sent x V x V
        batch_onehot_tensor2 = batch_to_batch_onehot_tensor2(sentence_sequences, self.vocab_size)  # B x seq x sent x V x V

        seq_len = sentence_sequences.size()[1]
        sentence_len = sentence_sequences.size()[2]
        batch_size = sentence_sequences.size()[0]
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

# maps dict to dict, includes metrics and loss
class MemoryNetworkModel(nn.Module):
    def __init__(self, sentence_len, vocab_size, n_line_types):
        super(MemoryNetworkModel, self).__init__()
        self.net = MemoryNetwork(sentence_len, vocab_size, n_line_types)
        self.metrics = {
            'class_acc': ClassAccuracyMetric()
        }
        self.loss = nn.CrossEntropyLoss()


    def forward(self, input_dict, skip_metrics=False):
        # input_dict: {'x': ..., 'label': ...}
        result = {}

        sentence_sequences = input_dict.get('sentence_sequence')
        questions = input_dict.get('question')
        line_types = input_dict.get('line_type')
        labels = input_dict.get('label', None)
        logits = self.net(sentence_sequences, questions, line_types, labels)
        result['logits'] = logits
        result['pred'] = F.softmax(logits, dim=1)

        if labels is not None:
            loss = self.loss(logits, labels)
            if not skip_metrics:
                for metric in self.metrics.values():
                    metric(logits, labels)
            result['loss'] = loss

        return result

    def get_metrics(self, reset=False):
        return flatten_dict_of_dict({
            metric_name: metric.get_metric(reset)
            for metric_name, metric in self.metrics.items()
        }, delimiter='/')


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