import torch
from torch.utils.data import DataLoader
from ml.parsing.parser import parse_file


class WordMap:
    def __init__(self):
        self._words = {}
        self._ix_to_word = {0: '<NULL>'}
        self._max_id = 0

    def add_word(self, word):
        if word in self._words:
            return self._words[word]
        else:
            self._max_id += 1
            self._words[word] = self._max_id
            self._ix_to_word[self._max_id] = word
            return self._max_id

    def get_id(self, word, default=None):
        return self._words.get(word, default)

    def get_word(self, ix, default=None):
        return self._ix_to_word.get(ix, default)


class Tensoriser:
    def __init__(self):
        self._wordmap = WordMap()
        self.seen_sentence_line_types = set()
        self.seen_question_line_types = set()

    def _sentence_line_type(self, sentence, sentence_max_length):
        line_type = (sentence_max_length * sentence.line_class) + len(sentence.text.strip().split(' '))
        self.seen_sentence_line_types.add(line_type)
        return line_type

    def _question_line_type(self, question, sentence_max_length):
        line_type = (sentence_max_length * question.question_class) + len(question.text.strip().split(' '))
        self.seen_question_line_types.add(line_type)
        return line_type

    def _story_line_types(self, story, sentence_max_length, story_max_length):
        line_types = [self._sentence_line_type(sentence, sentence_max_length) for sentence in story]
        line_types.extend([0] * (story_max_length - len(story)))
        return line_types

    def _tensorise_sentence(self, sentence, sentence_max_length):
        sentence = sentence.strip().strip('.').lower()
        words = sentence.split(' ')
        transformed_words = [self._wordmap.add_word(word) for word in words]
        transformed_words.extend([0] * (sentence_max_length - len(words)))
        return transformed_words

    def _tensorise_story(self, story, sentence_max_length, story_max_length):
        transformed_sentences = [self._tensorise_sentence(sentence.text, sentence_max_length) for sentence in story]
        pad_sentence = [0] * sentence_max_length
        transformed_sentences.extend([pad_sentence] * (story_max_length - len(transformed_sentences)))
        return transformed_sentences

    def tensorise(self, story_collection):
        transformed_stories = torch.tensor(
            [self._tensorise_story(story, story_collection.max_sentence_length, story_collection.max_story_length)
             for story in story_collection.stories])
        sentence_line_types = torch.tensor(
            [self._story_line_types(story, story_collection.max_sentence_length, story_collection.max_story_length)
             for story in story_collection.stories])
        transformed_questions = torch.tensor(
            [self._tensorise_sentence(question.text, story_collection.max_sentence_length)
             for question in story_collection.questions])
        question_line_types = torch.tensor(
            [self._question_line_type(question, story_collection.max_sentence_length)
             for question in story_collection.questions])
        transformed_answers = torch.tensor([self._wordmap.add_word(answer) for answer in story_collection.answers])
        story_lengths = torch.tensor([len(story) for story in story_collection.stories], dtype=torch.long)
        return (transformed_stories, transformed_questions, sentence_line_types, question_line_types,
                story_lengths, transformed_answers)

    def to_dataloader(self, story_collection, batch_size, shuffle):
        return DataLoader(list(zip(*self.tensorise(story_collection))), batch_size=batch_size, shuffle=shuffle)


def tensorised_to_full_sequence(tensorized_story, tensorized_question, word_map):
    result = [[word_map.get_word(int(ix)) for ix in sent] for sent in tensorized_story]
    result.append([word_map.get_word(int(ix)) for ix in tensorized_question])
    return result


if __name__ == '__main__':
    test_data = parse_file('data/en-valid/qa10_train.txt')
    print(Tensoriser().tensorise(test_data))
