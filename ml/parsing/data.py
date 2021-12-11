from enum import IntEnum

with open('ml/parsing/movement.txt') as f:
    movement_verbs = f.read().split('\n')
with open('ml/parsing/get_item.txt') as f:
    get_item_verbs = f.read().split('\n')
with open('ml/parsing/lose_item.txt') as f:
    lose_item_verbs = f.read().split('\n')


class LineClass(IntEnum):
    SINGLE_LOCATION = 0
    DOUBLE_LOCATION = 1
    GET_ITEM = 3
    LOSE_ITEM = 4
    EITHER = 5
    IS_IN = 6


class QuestionClass(IntEnum):
    CURRENT_LOCATION = 0
    PREVIOUS_LOCATION = 1
    YES_NO_MAYBE = 2


class Sentence:
    def __init__(self, text):
        self.text = text
        self.line_class = get_line_class(text)


class Question:
    def __init__(self, text):
        self.text = text
        self.question_class = get_question_class(text)


class StoryCollection:
    def __init__(self):
        self.stories = []
        self.questions = []
        self.answers = []
        self.max_sentence_length = 0
        self.max_story_length = 0

    def add_story(self, story, question, answer):
        story_max_sentence_length = max([len(sentence.text.split(' ')) for sentence in story])
        self.max_sentence_length = max(self.max_sentence_length, story_max_sentence_length,
                                       len(question.text.split(' ')))
        self.stories.append(story.copy())
        self.max_story_length = max(self.max_story_length, len(story))
        self.questions.append(question)
        self.answers.append(answer)

    def extend(self, story_collection):
        self.max_sentence_length = max(self.max_sentence_length, story_collection.max_sentence_length)
        self.max_story_length = max(self.max_story_length, story_collection.max_story_length)
        self.stories.extend(story_collection.stories)
        self.questions.extend(story_collection.questions)
        self.answers.extend(story_collection.answers)


def get_line_class(line):
    # Inspecting the dataset showed that the word 'and' appears iff two people change location
    if ' and ' in line:
        return LineClass.DOUBLE_LOCATION
    if ' is either in the ' in line:
        return LineClass.EITHER
    if ' is in the ' in line:
        return LineClass.IS_IN

    line_words = set(line.split(' '))
    if set(movement_verbs) & line_words:
        return LineClass.SINGLE_LOCATION
    if set(get_item_verbs) & line_words:
        return LineClass.GET_ITEM
    if set(lose_item_verbs) & line_words:
        return LineClass.LOSE_ITEM
    raise LookupError(f'The verb has not been categorised: {line}')


def get_question_class(line):
    line_split = line.split(' ')
    if ' before ' in line:
        return QuestionClass.PREVIOUS_LOCATION
    elif line_split[0].lower() == 'is' and line_split[2] == 'in':
        return QuestionClass.YES_NO_MAYBE
    elif line_split[0].lower() == 'where':
        return QuestionClass.CURRENT_LOCATION
    raise LookupError(f'The line has not been categorised: {line}')
