# Basic Chatbot Trained on Reddit Data

This code takes [suriyadeepan/practical_seq2seq](https://github.com/suriyadeepan/practical_seq2seq) 
and applies it to a subset of the may reddit dataset.

### Execution

First, clone the repo
```
git clone https://github.com/Data-Bot-Box/SnarkyChatBot2.git
```

Next, download reddit shower thoughts dataset from [here](https://www.kaggle.com/reddit/reddit-comments-may-2015)
Then, run the following code (from ~/SnarkyChatBot2/datasets/may_reddit):

```
pip3 install nltk
python3 data.py
```

After the data is extracted, run the following code to go back to the source
directory and make a checkpoint directory and train the model.

```
cd ../../
mkdir ckpt
mkdir ckpt/reddit
python3 train_chatbot.py
```

The model should train from there. The code will also run inference on the test set.
This inference code can be used to integrate with the chatbot.
