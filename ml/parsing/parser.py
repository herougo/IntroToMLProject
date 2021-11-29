from ml.parsing.data import StoryCollection, Sentence, Question
import re

line_pattern = re.compile(r"(?P<id>\d+) (?P<text>.*?)(\?\s*\t(?P<answer>.*?)\t(?P<supporting>.*))?$")


def parse_file(path):
    data = StoryCollection()
    curr_story = []
    curr_id = 0
    with open(path) as f:
        for line in f:
            match = line_pattern.match(line)
            line_id = int(match.group("id"))
            line_text = match.group('text')
            answer = match.group('answer')
            if line_id < curr_id:
                curr_story = []
            curr_id = line_id
            if answer is not None:
                # If there is an answer, the text must be a question. It doesn't get added to the story.
                question = Question(line_text)
                data.add_story(curr_story, question, answer)
            else:
                sentence = Sentence(line_text)
                curr_story.append(sentence)
    return data


if __name__ == '__main__':
    test_data = parse_file('data/en-valid/qa6_valid.txt')
    for story, question, answer in zip(test_data.stories, test_data.questions, test_data.answers):
        print(story)
        print(question)
        print(answer)

