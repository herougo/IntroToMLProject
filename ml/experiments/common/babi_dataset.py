from torch.utils.data import Dataset


def recursive_yield(maybe_list):
    if isinstance(maybe_list, (list, tuple)):
        for element in maybe_list:
            for result in recursive_yield(element):
                yield result
        else:
            yield maybe_list


def iter_all_text(dataset, tokenized_text_fields):
    for i in range(len(dataset)):
        example = dataset[i]
        for key in tokenized_text_fields:
            for result in recursive_yield(example[key]):
                yield result


class BABITask(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self._parse_babi_text()

    def _parse_babi_txt(self):
        self.sentence_sequences = None
        self.questions = None
        self.line_types = None
        self.labels = None
        self.max_seq_len = None
        self.max_sentence_len = None

        raise NotImplementedError()

    def __getitem__(self, ix):
        return {
            'sentence_sequence': self.sentence_sequences[ix],
            'question': self.questions[ix],
            'line_types': self.line_types[ix],
            'label': self.label[ix]
        }

    def __len__(self):
        return len(self.sentence_sequences)
