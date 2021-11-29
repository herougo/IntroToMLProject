from enum import IntEnum


class LineClass(IntEnum):
    SINGLE_LOCATION = 0
    DOUBLE_LOCATION = 1
    GET_ITEM = 3
    LOSE_ITEM = 4


class QuestionClass(IntEnum):
    CURRENT_LOCATION = 0
    PREVIOUS_LOCATION = 1


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


def get_line_class(line):
    # Inspecting the dataset showed that the word 'and' appears iff two people change location
    if ' and ' in line:
        return LineClass.DOUBLE_LOCATION
    movement = []
    get_item = []
    lose_item = []
    with open('../parsing/movement.txt') as f:
        movement = f.readlines()
    with open('../parsing/get_item.txt') as f:
        get_item = f.readlines()
    with open('../parsing/lose_item.txt') as f:
        lose_item = f.readlines()
    for verb in movement:
        # Remove trailing newline
        verb = verb[:-1]
        if verb in line:
            return LineClass.SINGLE_LOCATION
    for verb in get_item:
        # Remove trailing newline
        verb = verb[:-1]
        if verb in line:
            return LineClass.GET_ITEM
    for verb in lose_item:
        # Remove trailing newline
        verb = verb[:-1]
        if verb in line:
            return LineClass.LOSE_ITEM
    return None


def get_question_class(line):
    if ' before ' in line:
        return QuestionClass.PREVIOUS_LOCATION
    else:
        return QuestionClass.CURRENT_LOCATION
