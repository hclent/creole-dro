"""
Main driver code
"""


import argparse # option parsing
from src.dataset import Dataset
from src.model import SVM
import random
import numpy as np
from operator import itemgetter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process_command_line():
  """
  Return a 1-tuple: (args list).
  `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
  """

  parser = argparse.ArgumentParser(description='usage') # add description
  # positional arguments
  parser.add_argument('d1s', metavar='domain1-source', type=str, help='domain 1 source')

  parser.add_argument('d2s', metavar='domain2-source', type=str, help='domain 2 source')

  parser.add_argument('v', metavar='vocab', type=str, help='shared bpe vocab')

  # optional arguments
  parser.add_argument('-b', '--batch-size', dest='b', type=int, default=32, help='batch_size')

  args = parser.parse_args()
  return args


def f_importances(coef, names, wordLUT):
    coef = coef.toarray()
    top_pos_coefficients = np.argsort(coef)[-1:]
    for blah in top_pos_coefficients:
        print blah , " is type  " , type(blah)
        for number in blah:
            print wordLUT[number]
 
    
    top_neg_coefficients = np.argsort(coef)[:10]
    top_coefficients = np.hstack([top_neg_coefficients, top_pos_coefficients])
    plt.figure(figsize=(15,5))
    print "type coef" , coef , " is type ", type(coef)
    print "top pos coefficients " , top_pos_coefficients
    print "top neg coefficients ", top_neg_coefficients
    print "top coeffs " , top_coefficients , " is type " , type(top_coefficients)
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arrange(20), coef[top_coefficients], color=colors)
    feature_names = np.array(names)
    plt.xticks(np.arange(1, 1 + 2 * 10), feature_names[top_coefficients], rotation=60, ha='right')
    plt.savefic('wtf1.png')
    print "done"
    """
    imp = coef
    imp, names = zip(*sorted(zip(imp, names)))
    for i, n in zip(imp, names):
        #print "i: ", i , " n: " , n
        #print "type i ", type(i)
        #print "type n " , type(n)
        #print wordLUT[n]
        print n
    """

def main(domain1_source, domain2_source,  vocab, batch_size):
    data_iterator = Dataset(domain1_source,
                            domain2_source,
                            vocab, 
                            batch_size=batch_size)
    model = SVM(batch_size, data_iterator.get_vocab_size())
    model.fit(data_iterator) 
    print 'INFO: testing...'
    test_mae = model.test(data_iterator, mae=True)
    print 'INFO: test MAE: ', test_mae
    print 'INFO: PAD value: ', 2. * (1. - 2. * test_mae)
   
    """ 
    feature_names = []
    id_to_word = {}
    for word, i in data_iterator.vocab_map.items():
        feature_names.append(i)
        id_to_word[i] = word
   
    #features = [t for t, i in sorted(data_iterator.vocab_map.items(), key=itemgetter(1))]
    #print(features)	 
    f_importances(model.model.coef_, feature_names, id_to_word)
    """

if __name__ == '__main__':

    args = process_command_line()
    main(args.d1s, args.d2s, args.v, args.b)
