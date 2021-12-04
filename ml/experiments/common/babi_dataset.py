import torch
from torch.utils.data import Dataset, DataLoader
from easydict import EasyDict as edict

from ml.parsing.parser import parse_file
from ml.parsing.data import StoryCollection
from ml.parsing.tensoriser import Tensoriser


SPLITS = {'train', 'val', 'test'}


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


def get_dummy_babi_dataloaders():
    batch_size = 32
    sentences = torch.tensor([[[1, 2, 3, 4, 5, 6, 7],
                               [3, 4, 5, 6, 7, 8, 9]] * 7] * batch_size, dtype=torch.long)
    questions = torch.tensor([[0, 0, 0, 0, 0, 0, 7]] * batch_size, dtype=torch.long)
    sentence_line_types = torch.tensor([[0, 1] * 7] * batch_size, dtype=torch.long)
    question_line_types = torch.tensor([0] * batch_size, dtype=torch.long)
    labels = torch.tensor([0] * batch_size, dtype=torch.long)

    data_loader = DataLoader(list(zip(sentences, questions, sentence_line_types, question_line_types, labels)),
                             batch_size=batch_size, shuffle=True)

    data_loader_metadata = edict({
        'sentence_len': 7,
        'vocab_size': 10,
        'n_sentence_line_types': 2,
        'n_question_line_types': 1
    })
    data_loaders = edict({
        'train': data_loader,
        'val': data_loader,
        'test': data_loader
    })
    return data_loaders, data_loader_metadata


def get_babi_dataloaders(task_ids, batch_size=32, shuffle=True):
    # task_ids: integer or list of integers corresponding to the task ids of bAbI we want to include in the dataset
    tensoriser = Tensoriser()
    story_collections = {
        'train': StoryCollection(),
        'val': StoryCollection(),
        'test': StoryCollection()
    }
    for task_id in task_ids:
        for split in SPLITS:
            if split == 'val':
                file_split = 'valid'
            else:
                file_split = split
            story_collections[split].extend(parse_file(f'data/en-valid/qa{task_id}_{file_split}.txt'))

    data_loaders = edict({
        key: tensoriser.to_dataloader(story_collections[key], batch_size=batch_size, shuffle=shuffle)
        for key in story_collections})
    max_sentence_len = max([story_collections[key].max_sentence_length for key in story_collections])
    metadata = edict({
        'sentence_len': max_sentence_len,
        'vocab_size': tensoriser._wordmap._max_id + 1,  # Accounting for the padding "word"
        'n_sentence_line_types': max(tensoriser.seen_sentence_line_types) + 1,
        'n_question_line_types': max(tensoriser.seen_question_line_types) + 1
    })
    return data_loaders, tensoriser, metadata

