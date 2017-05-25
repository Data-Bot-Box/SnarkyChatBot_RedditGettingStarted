import nltk
import itertools
import numpy as np
import pickle
import sqlite3

EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz '
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''

FILENAME = 'data/chat.txt'

limit = {
    'maxq': 20,
    'minq': 0,
    'maxa': 20,
    'mina': 3}

UNK = 'unk'
VOCAB_SIZE = 6000


def ddefault():
    return 1


def read_lines(filename):
  '''
   read lines from file
       return [list of lines]

  '''
  query = "SELECT A1.name, A1.body AS question, A1.parent_id, " +\
          "A2.name, A2.body AS response, A2.parent_id " +\
          "FROM (SELECT * FROM May2015 " +\
          "WHERE subreddit='Showerthoughts') as A1 " +\
          "INNER JOIN (SELECT * FROM May2015 " +\
          "WHERE subreddit='Showerthoughts') as A2 " +\
          "ON A1.name = A2.parent_id "

  print(query)

  sql_conn = sqlite3.connect('database.sqlite')
  print(sql_conn)
  query_results = sql_conn.execute(query)
  print(query_results)

  questions, responses = [], []
  idx = 0
  for result in query_results:
    questions.append(result[1])
    responses.append(result[4])
    print(idx)
    idx += 1

  print(len(questions), len(responses))
  print(questions[0:10])

  return questions, responses


def split_line(line):
  '''
   split sentences in one line
    into multiple lines
      return [list of lines]

  '''
  return line.split('.')


def filter_line(line, whitelist):
  '''
   remove anything that isn't in the vocabulary
      return str(pure ta/en)

  '''
  return ''.join([ch for ch in line if ch in whitelist])


def index_(tokenized_sentences, vocab_size):
  '''
   read list of words, create index to word,
    word to index dictionaries
      return tuple( vocab->(word, count), idx2w, w2idx )

  '''
  # get frequency distribution
  freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
  # get vocabulary of 'vocab_size' most used words
  vocab = freq_dist.most_common(vocab_size)
  # index2word
  index2word = ['_'] + [UNK] + [x[0] for x in vocab]
  # word2index
  word2index = dict([(w, i) for i, w in enumerate(index2word)])
  return index2word, word2index, freq_dist


def filter_data(questions, responses):
  '''
   filter too long and too short sequences
      return tuple( filtered_ta, filtered_en )

  '''
  filtered_q, filtered_a = [], []
  raw_data_len = len(questions)

  for i in range(0, len(questions)):
    qlen = len(questions[i].split(' '))
    alen = len(responses[i].split(' '))
    if qlen >= limit['minq'] and qlen <= limit['maxq']:
      if alen >= limit['mina'] and alen <= limit['maxa']:
        filtered_q.append(questions[i])
        filtered_a.append(responses[i])

  # print the fraction of the original data, filtered
  filt_data_len = len(filtered_q)
  filtered = int((raw_data_len - filt_data_len) * 100 / raw_data_len)
  print(str(filtered) + '% filtered from original data')

  return filtered_q, filtered_a


def zero_pad(qtokenized, atokenized, w2idx):
  '''
   create the final dataset :
    - convert list of items to arrays of indices
    - add zero padding
        return ( [array_en([indices]), array_ta([indices]) )
  '''
  # num of rows
  data_len = len(qtokenized)

  # numpy arrays to store indices
  idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
  idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)

  for i in range(data_len):
    q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'])
    a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'])

    # print(len(idx_q[i]), len(q_indices))
    # print(len(idx_a[i]), len(a_indices))
    idx_q[i] = np.array(q_indices)
    idx_a[i] = np.array(a_indices)

  return idx_q, idx_a


def pad_seq(seq, lookup, maxlen):
  '''
   replace words with indices in a sequence
    replace with unknown if word not in lookup
      return [list of indices]

  '''
  indices = []
  for word in seq:
    if word in lookup:
      indices.append(lookup[word])
    else:
      indices.append(lookup[UNK])
  return indices + [0] * (maxlen - len(seq))


def process_data():

  print('\n>> Read lines from file')
  questions, responses = read_lines(filename=FILENAME)

  # change to lower case (just for en)
  questions = [line.lower() for line in questions]
  responses = [line.lower() for line in responses]

  print('\n:: Sample from read(p) questions')
  print(questions[121:125])

  print('\n:: Sample from read(p) responses')
  print(responses[121:125])

  # filter out unnecessary characters
  print('\n>> Filter questions')
  questions = [filter_line(line, EN_WHITELIST) for line in questions]
  print(questions[121:125])

  print('\n>> Filter responses')
  responses = [filter_line(line, EN_WHITELIST) for line in responses]
  print(responses[121:125])

  # filter out too long or too short sequences
  print('\n>> 2nd layer of filtering')
  qlines, alines = filter_data(questions, responses)
  print('\nq : {0} ; a : {1}'.format(qlines[60], alines[60]))
  print('\nq : {0} ; a : {1}'.format(qlines[61], alines[61]))

  # convert list of [lines of text] into list of [list of words]
  print('\n>> Segment lines into words')
  qtokenized = [wordlist.split(' ') for wordlist in qlines]
  atokenized = [wordlist.split(' ') for wordlist in alines]
  print('\n:: Sample from segmented list of words')
  print('\nq : {0} ; a : {1}'.format(qtokenized[60], atokenized[60]))
  print('\nq : {0} ; a : {1}'.format(qtokenized[61], atokenized[61]))

  # indexing -> idx2w, w2idx : en/ta
  print('\n >> Index words')
  idx2w, w2idx, freq_dist = index_(qtokenized + atokenized,
                                   vocab_size=VOCAB_SIZE)

  print('\n >> Zero Padding')
  idx_q, idx_a = zero_pad(qtokenized, atokenized, w2idx)

  print('\n >> Save numpy arrays to disk')
  # save them
  np.save('idx_q.npy', idx_q)
  np.save('idx_a.npy', idx_a)

  # let us now save the necessary dictionaries
  metadata = {
      'w2idx': w2idx,
      'idx2w': idx2w,
      'limit': limit,
      'freq_dist': freq_dist}

  # write to disk : data control dictionaries
  with open('metadata.pkl', 'wb') as f:
      pickle.dump(metadata, f)


def load_data(PATH=''):
    # read data control dictionaries
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_q = np.load(PATH + 'idx_q.npy')
    idx_a = np.load(PATH + 'idx_a.npy')
    return metadata, idx_q, idx_a


if __name__ == '__main__':
    process_data()
