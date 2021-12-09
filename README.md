# IntroToMLProject

Learning simple bAbI tasks with a memory-based network.

To start an experiment, create a folder in `ml/experiments`. See `ml/experiments/babione` for an example and organize the code similar to it. Make use of the code in the common code if you can (though the common folder should rarely be modified (so that past experiments still work).

Implementation To Do

- [x] (Henri) basic code structure
- [x] (Henri) basic memory network model
- [x] (Henri) integrate the memory network with the rest of the code
- [x] (Henri) base interpretability code
- [x] (Kirill) writing a data parser
  - converting bAbI file into raw (line type sequence, sentence sequence, question, answer) tuples
  - creating a dataloader (includes tokenizing and padding)


Experiments To Do

- [x] bAbI task 1 - single task
  - [x] with the memory retrieval being the output
  - [x] with a linear layer at the end
  - [x] with a's, b's, and c's being learned from the input
- [x] bAbI task 12 - conjunction
  - [x] with the memory retrieval being the output
  - [x] with a linear layer at the end
  - [x] with a's, b's, and c's being learned from the input
- [ ] bAbI task 6 - yes/no
  - [ ] with the memory retrieval being the output
  - [ ] with a linear layer at the end
  - [ ] with a's, b's, and c's being learned from the input
- [ ] bAbI task 6 - indefinite knowledge (e.g. either in the bathroom or)
  - [ ] with the memory retrieval being the output
  - [ ] with a linear layer at the end
  - [ ] with a's, b's, and c's being learned from the input
- [ ] interpretability and code extraction of bAbI tasks 1-3
- [ ] 1, 6, 10, 12 learned jointly
  
Future Work

- [ ] bAbI task 2
  - ...
- [ ] bAbI task 11 and 13
  - ...
- [ ] merge 2, 11, and 13 results
- [ ] learn all tasks separately with the same architecture
- [ ] learn all tasks jointly with the same architecture
- [ ] interpretability and code extraction of bAbI tasks 4-5
