# IntroToMLProject

Learning simple bAbI tasks with a memory-based network.

To start an experiment, create a folder in `ml/experiments`. See `ml/experiments/babione` for an example and organize the code similar to it. Make use of the code in the common code if you can (though the common folder should rarely be modified (so that past experiments still work).

Implementation To Do

- [x] (Henri) basic code structure
- [x] (Henri) basic memory network model
- [ ] (Henri) integrate the memory network with the rest of the code
- [ ] interpretability code
- [ ] writing a data parser
  - converting bAbI file into raw (line type sequence, sentence sequence, question, answer) tuples
  - creating a dataloader (includes tokenizing and padding)


Experiments To Do

- [ ] 1st bAbI task
  - [ ] with the memory retrieval being the output
  - [ ] with a linear layer at the end
  - [ ] with a's, b's, and c's being learned from the input
- [ ] 2nd bAbI task (try 3 versions like the 1st bAbI task)
- [ ] 3rd bAbI task (try 3 versions like the 1st bAbI task)
- [ ] interpretability and code extraction of bAbI tasks 1-3
- [ ] 1, 2, and 3 learned jointly
- [ ] 4th bAbI task
  - ...
- [ ] 5th bAbI task
  - ...
- [ ] merge 4th and 5th results
- [ ] learn all tasks separately with the same architecture
- [ ] learn all tasks jointly with the same architecture
- [ ] interpretability and code extraction of bAbI tasks 4-5
- [ ] (Now trying the above without access to line types)
