# Basic Chatbot Trained on Reddit Data

To execute download reddit shower thoughts dataset from [here](https://www.kaggle.com/reddit/reddit-comments-may-2015)
place in the datasets/may_reddit diretory and unzip and run (Make sure you're in the datasets/may_reddit directory).

```
python data.py
```

After the data is extracted, run the following code to go back to the source
directory and make a checkpoint directory and train the model.

```
cd ../../
mkdir ckpt
mkdir ckpt/reddit
python train_chatbot.py
```

The model should train from there. The code will also run inference on the test set.
This inference code can be used to integrate with the chatbot.
