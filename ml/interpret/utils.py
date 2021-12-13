import torch
import numpy as np


def interpret_to_code_batch(sentence_sequences, in_between_values, story_lengths):
    batch_size = sentence_sequences.size()[0]
    sent_len = sentence_sequences.size()[2]
    a_s = [a.detach().cpu() for a in in_between_values.a_s]
    b_s = [b.detach().cpu() for b in in_between_values.b_s]
    c = in_between_values.c.detach().cpu()
    code_batch = []
    for i in range(batch_size):
        code = []
        for j in range(story_lengths[i]):
            code_block = []
            b_slice = torch.sigmoid(b_s[j][i]).view(sent_len).numpy()
            for row in np.argwhere(b_slice >= 0.8):
                code_block.append(f'memory[sent[{row[0]}]] = None')
            a_slice = torch.tanh(a_s[j][i]).view(sent_len, sent_len).numpy()
            for row in np.argwhere(np.abs(a_slice) >= 0.8):
                code_block.append(f'memory[sent[{row[0]}]] = sent[{row[1]}]')
            code.append(code_block)
        c_slice = torch.sigmoid(c[i]).view(sent_len).numpy()
        code_block = []
        for row in np.argwhere(c_slice >= 0.8):
            code_block.append(f'return memory[sent[{row[0]}]]')
        code.append(code_block)

        code_batch.append(code)

    return code_batch


def simulate_code_lines(full_sequence, code):
    '''
    full_sequence: e.g. [['John', 'went', 'to', 'the', 'bathroom'], ... , ['where', 'is', 'john']]
    code: e.g. [['memory[sent[0]] = sent[4]'], ... ['return memory[sent[2]]']]
    '''
    memory = {}
    result = []
    try:
        for sent, code_block in zip(full_sequence, code):
            for line in code_block:
                if line.startswith('return'):
                    result.append(str(eval(line[7:])))
                else:
                    exec(line)
        return ','.join(result)
    except Exception as ex:
        return ''
