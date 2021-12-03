import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from ml.metrics.pytorch_metrics import ClassAccuracyMetric
from ml.metrics.meter_metrics import AverageMeter
from ml.utils.utils import flatten_dict_of_dict
from ml.interpret.utils import interpret_to_code_batch, simulate_code_lines
from easydict import EasyDict as edict
from ml.parsing.tensoriser import tensorised_to_full_sequence


def create_batch_of_identity(batch_size, n):
    x = torch.eye(n)
    x = x.reshape((1, n, n))
    y = x.repeat(batch_size, 1, 1)
    return y


def batch_to_batch_onehot_tensor(batch, vocab_size):
    '''
    output a tensor with shape: B x seq x sent x sent x V x V
    batch_onehot_tensor[b, seq_i, sent_j, sent_k, v1, v2] == 1 iff the following is true.
    - v1 corresponds to the jth word in the ith sentence in batch example b
    - v2 corresponds to the kth word in the ith sentence in batch example b
    '''
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
                    batch_onehot_tensor[b][seq][l1][l2][key][val] = 1.0
    return batch_onehot_tensor


def batch_to_batch_onehot_tensor2(batch, vocab_size):
    '''
    output a tensor with shape: B x seq x sent x V x V
    batch_onehot_tensor2[b, seq_i, sent_j, v, v] == 1 iff the following is true.
    - v corresponds to the jth word in the ith sentence in batch example b
    '''
    sequence_len = batch.size()[1]
    sentence_len = batch.size()[2]
    batch_size = batch.size()[0]
    batch_onehot_tensor2 = torch.zeros((batch_size, sequence_len, sentence_len, 
                                       vocab_size, vocab_size))
    for b in range(batch_size):
        for seq in range(sequence_len):
            for l in range(sentence_len):
                key = batch[b][seq][l]
                batch_onehot_tensor2[b][seq][l][key][key] = 1.0
    return batch_onehot_tensor2


def questions_to_questions_onehot_tensor(questions, vocab_size):
    '''
    Output a tensor of shape: B x sent x V x 1
    result[b, sent_i, v, 0] == 1 iff v corresponds to the ith word in the sentence of batch example b
    '''
    batch_size = questions.size()[0]
    sentence_len = questions.size()[1]
    result = torch.zeros((batch_size, sentence_len, vocab_size, 1))
    for b in range(batch_size):
        for l in range(sentence_len):
            key = questions[b][l]
            result[b][l][key][0] = 1.0
    return result


class MemoryNetwork(nn.Module):
    def __init__(self, sentence_len, vocab_size, n_sentence_line_types, n_question_line_types, use_cuda,
                 learn_abc_from_input=False, embedding_dim=100):
        super().__init__()
        self.n_sentence_line_types = n_sentence_line_types
        self.n_question_line_types = n_question_line_types
        self.vocab_size = vocab_size
        self.learn_abc_from_input = learn_abc_from_input
        if learn_abc_from_input:
            self.embedding_dim = embedding_dim
            self.sent_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
            self.ab_hidden = nn.Linear(sentence_len * embedding_dim, sentence_len * embedding_dim)
            self.a_head = nn.Linear(sentence_len * embedding_dim, sentence_len * sentence_len)
            self.b_head = nn.Linear(sentence_len * embedding_dim, sentence_len)
            self.question_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
            self.c_hidden = nn.Linear(sentence_len * embedding_dim, sentence_len * embedding_dim)
            self.c_head = nn.Linear(sentence_len * embedding_dim, sentence_len)
        else:
            self.a = Parameter(torch.zeros((n_sentence_line_types, sentence_len, sentence_len), requires_grad=True))
            self.b = Parameter(torch.zeros((n_sentence_line_types, sentence_len), requires_grad=True))
            self.c = Parameter(torch.zeros((n_question_line_types, sentence_len), requires_grad=True))
        self.use_cuda = use_cuda
    
    def forward(self, sentence_sequences, questions, sentence_line_types, question_line_types, story_lengths):
        batch_onehot_tensor = batch_to_batch_onehot_tensor(sentence_sequences, self.vocab_size)
        # B x seq x sent x sent x V x V
        batch_onehot_tensor2 = batch_to_batch_onehot_tensor2(sentence_sequences, self.vocab_size)
        # B x seq x sent x V x V

        seq_len = sentence_sequences.size()[1]
        sentence_len = sentence_sequences.size()[2]
        batch_size = sentence_sequences.size()[0]
        memory = torch.zeros((batch_size, self.vocab_size, self.vocab_size))
        identity = create_batch_of_identity(batch_size, self.vocab_size)
        ones = torch.ones((batch_size, self.vocab_size, self.vocab_size))
        arange = torch.arange(batch_size)
        questions_onehot_tensor = questions_to_questions_onehot_tensor(questions, self.vocab_size)

        if self.use_cuda:
            batch_onehot_tensor = batch_onehot_tensor.cuda()
            batch_onehot_tensor2 = batch_onehot_tensor2.cuda()
            memory = memory.cuda()
            identity = identity.cuda()
            ones = ones.cuda()
            arange = arange.cuda()
            questions_onehot_tensor = questions_onehot_tensor.cuda()

        a_s = []
        b_s = []

        for i in range(seq_len):
            if self.learn_abc_from_input:
                emb = self.sent_emb(sentence_sequences[:, i]).view(batch_size, self.embedding_dim * sentence_len)
                ab_hidden = self.ab_hidden(emb)
                ab_hidden = torch.relu(ab_hidden)
                a = self.a_head(ab_hidden).view(batch_size, sentence_len, sentence_len, 1, 1)
                b = self.b_head(ab_hidden).view(batch_size, sentence_len, 1, 1)
            else:
                a = self.a.view(1, self.n_sentence_line_types, sentence_len, sentence_len, 1, 1).repeat(batch_size, 1, 1, 1, 1, 1)
                a = a[arange, sentence_line_types[:, i]]
                b = self.b.view(1, self.n_sentence_line_types, sentence_len, 1, 1).repeat(batch_size, 1, 1, 1, 1)
                b = b[arange, sentence_line_types[:, i]]
            a_s.append(a)
            b_s.append(b)
            write_to_memory = torch.sum(torch.tanh(a) * batch_onehot_tensor[:, i], axis=[1, 2])
            modified_identity = identity - torch.minimum(torch.sum(torch.sigmoid(b) * batch_onehot_tensor2[:, i], axis=1), ones)
            remember = torch.bmm(modified_identity, memory)
            # bmm: batch matrix multiplication
            memory = write_to_memory + remember

        memory_view = memory.view(batch_size, 1, self.vocab_size, self.vocab_size).repeat(1, sentence_len, 1, 1)

        if self.learn_abc_from_input:
            emb = self.question_emb(questions).view(batch_size, self.embedding_dim * sentence_len)
            c_hidden = self.c_hidden(emb)
            c_hidden = torch.relu(c_hidden)
            c = self.c_head(c_hidden).view(batch_size, sentence_len, 1, 1)
        else:
            c = self.c.view(1, self.n_question_line_types, sentence_len).repeat(batch_size, 1, 1)
            c = c[arange, question_line_types]  # batch_size x sentence_len
            c = c.view(batch_size, sentence_len, 1, 1)

        mem_onehot_matmul = torch.bmm(
            memory_view.transpose(2, 3).view(-1, self.vocab_size, self.vocab_size),
            questions_onehot_tensor.view(-1, self.vocab_size, 1)
        ).view(batch_size, sentence_len, self.vocab_size, 1)

        result = torch.sum(torch.sigmoid(c) * mem_onehot_matmul, axis=[1, 3])

        in_between_values = edict({
            'a_s': a_s,
            'b_s': b_s,
            'c': c
        })

        return result, in_between_values


# maps dict to dict, includes metrics and loss
class MemoryNetworkModel(nn.Module):
    def __init__(self, sentence_len, vocab_size, n_sentence_line_types, n_question_line_types, use_cuda,
                 word_map, use_interpretability=False, linear_layer_output=False,
                 learn_abc_from_input=False, embedding_dim=100):
        super(MemoryNetworkModel, self).__init__()
        self.net = MemoryNetwork(sentence_len, vocab_size, n_sentence_line_types, n_question_line_types, use_cuda,
                                 learn_abc_from_input=learn_abc_from_input, embedding_dim=embedding_dim)
        self.linear_layer_output = linear_layer_output
        if linear_layer_output:
            self.linear = nn.Linear(vocab_size, vocab_size)
        else:
            self.linear = None
        self.metrics = {
            'class_acc': ClassAccuracyMetric(),
            'interpret_class_acc': AverageMeter()
        }
        self.loss = nn.CrossEntropyLoss()
        self.word_map = word_map
        self.use_interpretability = use_interpretability

    def forward(self, input_dict, skip_metrics=False):
        result = {}

        sentence_sequences = input_dict.get('sentence_sequence')
        questions = input_dict.get('question')
        sentence_line_types = input_dict.get('sentence_line_type')
        question_line_types = input_dict.get('question_line_type')
        story_lengths = input_dict.get('story_lengths')
        labels = input_dict.get('label', None)
        raw_memory_result, in_between_values = self.net(sentence_sequences, questions, sentence_line_types,
                                                        question_line_types, story_lengths)

        if self.linear_layer_output:
            logits = self.linear(raw_memory_result)
        else:
            # TODO: change
            logits = 10 * raw_memory_result - 5

        result['logits'] = logits
        result['pred'] = F.softmax(logits, dim=1)

        if labels is not None:
            loss = self.loss(logits, labels)
            if not skip_metrics:
                self.metrics['class_acc'](logits, labels)

                # interpretability
                if self.use_interpretability:
                    code_batch = interpret_to_code_batch(sentence_sequences, in_between_values, story_lengths)
                    batch_size = sentence_sequences.size()[0]
                    batch_interpret_right = 0

                    for i in range(batch_size):
                        code = code_batch[i]
                        tensorised_story = sentence_sequences[i][:story_lengths[i]]
                        tensorised_question = questions[i]
                        full_sequence = tensorised_to_full_sequence(tensorised_story, tensorised_question,
                                                                    self.word_map)
                        answer = simulate_code_lines(full_sequence, code)
                        answer_id = self.word_map.get_id(answer)
                        label = int(labels[i])
                        batch_interpret_right += answer_id == label
                    self.metrics['interpret_class_acc'].update(batch_interpret_right / batch_size, n=batch_size)
            result['loss'] = loss

        return result

    def get_metrics(self, reset=False):
        return flatten_dict_of_dict({
            metric_name: metric.get_metric(reset)
            for metric_name, metric in self.metrics.items()
        }, delimiter='/')


def main():
    batch_size = 32
    batch = torch.tensor([[[1, 2, 3, 4, 5, 6, 7],
                           [3, 4, 5, 6, 7, 8, 9]] * 7] * batch_size, dtype=torch.long)
    question = torch.tensor([[1, 2, 3, 4, 5, 6, 7]] * batch_size, dtype=torch.long)
    sentence_line_types = torch.tensor([[0, 1] * 7] * batch_size, dtype=torch.long)
    question_line_types = torch.tensor([0] * 7 * batch_size, dtype=torch.long)
    sequence_len = batch.size()[1]
    sentence_len = batch.size()[2]
    vocab_size = 23
    n_sentence_line_types = 2
    n_question_line_types = 1

    net = MemoryNetwork(sentence_len, vocab_size, n_sentence_line_types, n_question_line_types)
    opt = torch.optim.Adam(net.parameters())
    loss = net(batch, question, sentence_line_types, question_line_types)
    opt.zero_grad()
    loss.backward()
    opt.step()


if __name__ == '__main__':
    main()