#!/usr/bin/env python3.10
# Copyright 2022 Educational Testing Services

import asyncio
import enum
import json
import os
import re
import sys
import websockets

from nltk.tokenize import sent_tokenize
from difflib import ndiff
from pygtrie import Trie
from symspellpy import SymSpell, Verbosity
import autocorrect
import difflib
import nltk
import numpy as np
import operator
import pkg_resources
import pygtrie
import scipy
import stat
import time
import torch
import unicodedata

import neuspell
from neuspell import BertChecker
checker = BertChecker()
checker.from_pretrained()

if sys.version_info[0] == 3:
    xrange = range

transformer_types = enum.Enum('Transformer Type', "NONE BERT NEUSPELL")


class SpellCorrect:
    # Global variables
    VALID: dict  # aspell dictionary ends up here
    SORTED_ATTESTED: dict  # Word type dictionary from the responses
    TRIE_ATTESTED: dict  # Same thing, in a trie
    lemmatizer: nltk.stem.WordNetLemmatizer  # Puts words into canonical form
    localDict: dict
    # Temporary dictionary from SORTED_ATTESTED (written into a file)
    symspell_responses: SymSpell
    # SymSpell with dictionary from student responses
    symspell_native: SymSpell  # SymSpell with native dictionary
    embeddings_dict: dict  # A place to store deep learning embeddings?
    stopwords: dict

    # This is used for naive BERT spell checker to detect missing
    # copulas. English has "be" verbs (e.g. "I am hungry") where many
    # languages don't ("Jestem głodny"), so this is a common fix.
    infl = ['am', 'are', 'is', 'was', 'were']

    # These should be tokenized as two words, whether because they are a
    # contraction or a typo (missing space)
    splittables = {
        'nt': ['n\'t'],
        'couldnt': ['could', 'n\'t'],
        'shouldnt': ['should', 'n\'t'],
        'wouldnt': ['would', 'n\'t'],
        'mightnt': ['might', 'n\'t'],
        'mustnt': ['must', 'n\'t'],
        'wont': ['wo', 'n\'t'],
        'dont': ['do', 'n\'t'],
        'doesnt': ['does', 'n\'t'],
        'didnt': ['did', 'n\'t'],
        'isnt': ['is', 'n\'t'],
        'arent': ['are', 'n\'t'],
        'wasnt': ['was', 'n\'t'],
        'werent': ['were', 'n\'t'],
        'havent': ['have', 'n\'t'],
        'hasnt': ['has', 'n\'t'],
        'hadnt': ['had', 'n\'t'],
        'alot': ['a', 'lot'],
        'ofthe': ['of', 'the'],
    }

    # We use these to identify missing spaces. These tend to be glummed onto
    # another word.
    fwords = [
        'a', 'all', 'and', 'as', 'at', 'by', 'for', 'from', 'if', 'in', 'more',
        'most', 'of', 'off', 'on', 'or', 'over', 'so', 'that', 'the', 'to',
        'under', 'with']

    def __init__(self, aspell, symspell):
        '''
        Load dictionaries and lemmatizer. The parameters are the locations
        of the data files.
        '''
        self.VALID = self.load_dict(aspell)
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.symspell_responses = SymSpell(max_dictionary_edit_distance=2,
                                           prefix_length=7)
        self.symspell_native = SymSpell(max_dictionary_edit_distance=2,
                                        prefix_length=7)
        self.symspell_native.load_dictionary(symspell, 0, 1, " ")
        self.stopwords = nltk.corpus.stopwords.words('english')

    # TODO: Change `preprocess` to also use an `Enum` object
    def spellcheck_corpus(self,
                          documents,
                          transformer=transformer_types.NONE,
                          verbose=False,
                          preprocess='native'):

        '''
        Run your choice of spell checkers. There are:
        * Surface spell checkers (`autocorrect` library, or our own corpus
          library). These can use soundex, edit distance, and
          similar constraints.
        * Deep learning spell checkers. Either `Neuspell` or our own naive
          BERT library, which runs slooowly, but has high accuracy.
        '''
        if transformer == transformer_types.NONE \
           and preprocess != 'autocorrect':
            preprocess = 'native'
        # Create a frequency dictionary from a list of documents
        self.SORTED_ATTESTED = \
            self.document_list_to_frequency_histogram(documents)
        self.TRIE_ATTESTED = self.make_trie(self.SORTED_ATTESTED)

        self.localDict = self.createLocalDict(self.SORTED_ATTESTED)
        # Should be streaming:
        #   import io
        #   dictionary = "Hello 5"
        #   sp.load_dictionary_stream(io.StringIO(dictionary), 0, 1)
        # More properly, DDG "file-like object"
        self.symspell_responses.create_dictionary(self.localDict)

        # We are setting up our deep learning models.
        # xfspell was nice, but too much work!
        tokenizer = None
        model = None

        # Load our deep learning libraries.
        # ---------------------------------
        # Deep learning libraries are large, so we load these on demand. We
        # don't always need these.

        # Naive model that combines BERT with a rule-based controller
        if transformer == transformer_types.BERT:
            from transformers import BertTokenizer, BertForMaskedLM
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            # This masks a word, and gives top N words which belong there.
            model = BertForMaskedLM.from_pretrained('bert-base-uncased',
                                                    return_dict=True)
        # One of the better deep learning spell check libraries

        # We have a set of student responses.
        # A response is a submission by one
        # student on one prompt.
        #
        # We extract chunks from this. Chunks are clause-length pieces
        # of text which we match on.
        #
        # For example, "John is happy with my performance, but not
        # willing to recommend me." becomes two chunks:
        # "John is happy with my performance" and "but not willing to
        # recommend me."

        # Return variables
        response_ids = []
        # Sequential number of the text in the response corpus
        responses = []
        # List of responss in the response corpus
        chunkids = []
        # Response ID identifying the response each chunk is extracted from
        responsechunks = []
        # The actual chunks (corresponding to chunkids, by position)
        response_index = 0
        # We're keeping track of which response we're on.
        # This increments sequentially

        # for doc, response_index in zip(documents, range(len(documents))):
        for i, doc in enumerate(documents):
            print('doc ', i, 'of', len(documents))
            doc = self.adjust_punctuation(doc)
            if verbose:
                print('\n--------\n')
                print(doc)
            if preprocess == 'native':
                if transformer == transformer_types.NEUSPELL:
                    new = self.surface_spellcheck(doc, False)
                else:
                    new = self.surface_spellcheck(doc, True)
                if verbose:
                    print('\n')
                    print(new)
                    print('\n')
            elif preprocess == 'autocorrect':
                new = self.autocorrect(doc)
                if verbose:
                    print('\n')
                    print(new)
                    print('\n')
            else:
                new = nltk.word_tokenize(doc)

            if transformer == transformer_types.BERT:
                new = self.deep_spellcheck(new,
                                           tokenizer,
                                           model,
                                           verbose)
                if verbose:
                    print('\n')
                    self.uprint(new)
                    print('\n\n\n')

                response_ids.append(str(response_index))
                responsechunks = self.chunk_sentences(new,
                                                      chunkids,
                                                      responsechunks,
                                                      response_index,
                                                      False,
                                                      verbose)
                responses.append(self.untokenize(new))

            elif transformer == transformer_types.NEUSPELL:
                outstring = ''
                lastToken = ''
                for token in new:
                    outstring += ' ' + token

                outstring = self.normalize(outstring)

                sentences = sent_tokenize(outstring)
                new = ''

                # Neuspell is trained to spell correct sentence
                # by sentence, so we have to sentence tokenize
                # before we send data to Neuspell.
                for sent in sentences:

                    # Work-around to deal with the fact that
                    # Neuspell strips out newlines
                    newS = checker.correct(sent.replace('\n', ' ~~~~ '))
                    newS = self.normalize(newS.replace(' ~ ~ ~ ~', '<>'))
                    if verbose:
                        print('\n')
                        print(newS)
                        print('\n\n\n')

                    new += ' ' + newS

                response_ids.append(str(response_index))

                responsechunks = self.chunk_sentences(new,
                                                      chunkids,
                                                      responsechunks,
                                                      response_index,
                                                      True,
                                                      verbose)
                responses.append(self.normalize(new))

            else:
                lastToken = ''
                outstring = ''
                for token in new:
                    token = token.replace('.\n', '\n')
                    token = token.replace('\n\n', '\n')
                    token = token.replace('. ', ' ')
                    token = token.replace('.  ', '  ')
                    if lastToken == '' and token != token.upper():
                        outstring += ' ' + token.capitalize()
                    elif ((lastToken == '.'
                           or lastToken == '?'
                           or lastToken == '!')
                          and token != token.upper()):
                        outstring += ' ' + token.capitalize()
                    else:
                        outstring += ' ' + token
                    lastToken = token

                outstring = self.normalize(outstring)
                response_ids.append(str(response_index))
                responsechunks = self.chunk_sentences(new,
                                                      chunkids,
                                                      responsechunks,
                                                      response_index,
                                                      False,
                                                      verbose)
                responses.append(outstring)
            response_index += 1
        os.remove(self.localDict)
        return (responses, response_ids, responsechunks, chunkids)

    def getWordFrequencies(self):
        return self.SORTED_ATTESTED

    def chunk_sentences(self,
                        new,
                        chunkids,
                        responsechunks,
                        cnt,
                        string_output,
                        verbose):
        sentences = self.split_by_sentence(new, string_output)
        backOne = None
        backTwo = None
        for sentence in sentences:
            chunks = self.split_sentence_into_chunks(sentence)
            for chunk in chunks:
                chunkids.append(str(cnt))
                if chunk == '.':
                    responsechunks.append(' ')
                elif (chunk == ' \n' and
                      backOne == '.'
                      and backTwo == ' \n'):
                    responsechunks.append(' ')
                else:
                    responsechunks.append(self.normalize(chunk))
                if verbose:
                    print('chunk: ', chunk)
            backTwo = backOne
            backOne = chunk
        return responsechunks

    def autocorrect(self, doc):
        spell = autocorrect.Speller()
        corr = spell(doc)
        return nltk.word_tokenize(corr)

    def surface_spellcheck(self, doc, normalize=True):
        tokenList = []
        paras = doc.split('\n')
        document = []
        for i, para in enumerate(paras):
            try:
                tokens = nltk.word_tokenize(para)
            except Exception as e:
                tokens = nltk.word_tokenize(para,
                                            preserve_line=True)
            if i == len(paras) - 1:
                document += tokens
            elif (len(tokens) > 0
                  and tokens[len(tokens) - 1] in '.?!'):
                document += tokens + ['<>']
                # special coding for paragraph breaks.
                # We are assuming newline = paragraph break.
            elif len(tokens) == 0:
                document += tokens + ['<>']
            else:
                document += tokens + ['.', '<>']
        for i in range(0, len(document)):
            word = document[i]
            if word in self.splittables:
                tokenList += self.splittables[word]
                continue
            if re.match('^[^A-Za-z0-9]+$', word):
                tokenList += [word]
                continue
            if len(word.strip()) > 0:
                if re.match('[0-9,.]+', word):
                    tokenList += [word]
                    continue
                kword = self.known_word(word)
                if len(kword) > 0:
                    if word == word.capitalize() and normalize:
                        tokenList += [kword[0].capitalize()]
                    else:
                        tokenList += [kword[0]]
                    continue
                c2 = self.candidates(word)
                if c2 is not None:
                    if word == c2 + ['s']:
                        tokenList += c2 + ["''s"]
                    else:
                        tokenList += c2
                else:
                    if word == word.capitalize() and normalize:
                        tokenList += [word.capitalize()]
                    else:
                        tokenList += [word]
        return tokenList

    def deep_spellcheck(self, tokenList, tokenizer, model, verbose):
        for i in range(0, len(tokenList)):
            if i < 4:
                left = 'Someone said that'
            else:
                left = ''
            leftbeam = i - 5
            if leftbeam < 0:
                leftbeam = 0
            for j in range(0, i):
                left += ' ' + tokenList[j]
            left = self.normalize(left)
            right = ''
            for k in range(i+1,
                           len(tokenList)):
                right += ' ' + tokenList[k]
            right = self.normalize(right)
            topn = self.topn_words(left, right, 15, tokenizer, model)
            contractionFlag = False
            # if verbose:
            #     self.uprint(tokenList[i],topn)
            if (tokenList[i].lower() == 'there'
                or tokenList[i].lower() == 'their') \
               and 'they' in topn \
               and 'are' in topn:
                contractionFlag = True
                if tokenList[i] == tokenList[i].capitalize():
                    tokenList[i] = 'They\'re'
                else:
                    tokenList[i] = 'they\'re'
            elif (tokenList[i].lower() == 'hes'
                  and 'he' in topn
                  and 'is' in topn):
                contractionFlag = True
                if tokenList[i] == tokenList[i].capitalize():
                    tokenList[i] = 'He\'s'
                else:
                    tokenList[i] = 'he\'s'
            elif tokenList[i].lower() == 'its' and 'it' in topn:
                contractionFlag = True
                if tokenList[i] == tokenList[i].capitalize():
                    tokenList[i] = 'It\'s'
                else:
                    tokenList[i] = 'it\'s'
            elif (topn[0] in self.infl
                  and tokenList[i] not in self.infl
                  and i > 0):
                if tokenList[i] == tokenList[i].capitalize():
                    tokenList = tokenList[0: i] \
                                + [topn[0].capitalize()] \
                                + tokenList[i:]
                else:
                    tokenList = tokenList[0: i] + [topn[0]] + tokenList[i:]
                left = left + ' ' + topn[0]
                topn = self.topn_words(left, right, 15, tokenizer, model)
            elif (topn[1] in self.infl
                  and tokenList[i] not in self.infl
                  and i > 0):
                if tokenList[i] == tokenList[i].capitalize():
                    tokenList = tokenList[0: i] \
                                + [topn[1].capitalize()] \
                                + tokenList[i:]
                else:
                    tokenList = tokenList[0: i]\
                                + [topn[1]] \
                                + tokenList[i:]
                left = left + ' ' + topn[1]
                topn = self.topn_words(left, right, 15, tokenizer, model)
            if not contractionFlag \
               and not re.match('[0-9]+', tokenList[i]) \
               and not tokenList[i] == tokenList[i].capitalize():
                # Don't even TRY doing spellcheck replacements from
                # the BERT list on numerals and proper names. It won't
                # work. We're looking for words that are good general
                # syntactic fits to the context that look like
                # they are a spelling variant of an ordinary word ...
                for altword in topn:
                    if abs(len(altword) - len(tokenList[i])) > 4:
                        continue
                    ld = self.LevenshteinDistance(tokenList[i], altword)
                    if (self.soundex(tokenList[i]) == self.soundex(altword)
                        and (len(tokenList[i].lower()) < 5
                             or self.consonants(tokenList[i].lower())
                             == self.consonants(altword))) \
                            or (ld <= 3
                                and len(tokenList[i]) > 2
                                and tokenList[i].lower()[0: 2]
                                == altword[0: 2]
                                and len(altword) > 2):
                        if tokenList[i] == tokenList[i].capitalize():
                            tokenList[i] = altword.capitalize()
                        else:
                            tokenList[i] = altword
                        break
            if verbose:
                self.uprint(tokenList[i])
        return tokenList

    def candidates(self, word):

        if len(word) < 4:
            return None

        # We'll start by matching simple transpositions, as long as
        # they are unambiguously valid from the larger spelling list.
        if word not in self.VALID.keys() \
           and word.lower() not in self.VALID.keys()\
           and word.capitalize() not in self.VALID.keys():
            filtered = [x for x in self.transposes(word)
                        if (x in self.VALID.keys()
                            or x.lower() in self.VALID.keys())]
            if len(filtered) == 1:
                if filtered[0].lower() in self.VALID.keys():
                    return [filtered[0].lower()]
                else:
                    return [filtered[0]]

        # Next we check an edit definition, again against correct
        # words that occur in our text sample.
        candList = {}
        if word != word.capitalize():
            candList = self.make_candlist(word.capitalize(),
                                          candList)
        if word != word.lower():
            candList = self.make_candlist(word.lower(),
                                          candList)
        if word != word.capitalize():
            candList = self.make_candlist(word, candList)
        matchedAt = 0
        matched = ''
        matchedCnt = 0
        if len(candList) > 0:
            # Basically, we pick the first item from the candidate
            # list in descending order of corpus frequency
            # that has an acceptable edit distance. This works,
            # basically, because we know that correctly spelled
            # words on the same topic as other writers to the same
            # prompt are far more probable than apparently
            # less plausible items.
            sorted_candList = dict(sorted(candList.items(),
                                          key=operator.itemgetter(1),
                                          reverse=False))
            for item in sorted_candList.keys():
                if item in self.SORTED_ATTESTED \
                   and self.SORTED_ATTESTED[item] > matchedCnt \
                   and (matchedAt == 0
                        or sorted_candList[item] <= matchedAt):
                    matched = item
                    matchedCnt = self.SORTED_ATTESTED[item]
                    matchedAt = sorted_candList[item]
            if matched.lower() != word.lower() \
               and matchedAt > 0 \
               and len(matched) >= len(word) - 2:
                if word == word.capitalize():
                    return [matched.capitalize()]
                else:
                    return [matched]

        # Next try a standard spellcheck match on the frequency
        # list of words attested in the document sample
        # using SymSpell for speed of matching
        suggestions = \
            self.symspell_responses.lookup(word,
                                           Verbosity.ALL,
                                           max_edit_distance=2,
                                           include_unknown=True)
        if len(suggestions) == 0:
            suggestions = \
                self.symspell_responses.lookup(word.lower(),
                                               Verbosity.ALL,
                                               max_edit_distance=2,
                                               include_unknown=True)

        if len(suggestions) == 1:
            corr = suggestions[0].term
        else:
            filtered_suggestions = \
                [x for x in suggestions
                 if (self.soundex(x.term) == self.soundex(word.lower())
                     and len(self.consonants(x.term))
                     <= len(self.consonants(word)) + 1)
                    or self.consonants(x.term) == self.consonants(word.lower())
                    or len(x.term) > 3
                    and len(self.consonants(x.term))
                    == len(self.consonants(word))
                    and x.term.lower()[0: 3] == word.lower()[0: 3]]
            if len(filtered_suggestions) > 0:
                corr = filtered_suggestions[0].term
            else:
                corr = suggestions[0].term

        if word.lower() != corr.lower():
            if self.soundex(word) == self.soundex(corr) \
               or len(corr) > 2 \
               and word[0: 2].lower() == corr[0: 2].lower():
                if corr.capitalize() in self.VALID \
                   and corr.lower() not in self.VALID:
                    return [corr.capitalize()]
                elif corr in self.VALID:
                    return [corr]

        # Next we'll match vowel or vowel digraph replacements, as long
        # as they don't change soundex category or word prefix and are
        # unambiguiously valid from the larger spelling list.
        if word not in self.VALID.keys() \
           and word.lower() not in self.VALID.keys() \
           and word.capitalize() not in self.VALID.keys():
            filtered = [x for x in self.replacevowel(word)
                        if x in self.VALID.keys()
                        and self.soundex(x) == self.soundex(word.lower())
                        and len(word) > 2
                        and len(x) > 2
                        and word.lower()[0: 2] == x.lower()[0: 2]]
            if len(filtered) == 1:
                if word == word.capitalize():
                    return [filtered[0].capitalize()]
                else:
                    return [filtered[0]]

        # Next we'll match simple vowel insertions, as long as
        # they don't change soundex category or word prefix and
        # are unambiguously valid from the larger spelling list.
        if word not in self.VALID.keys() \
           and word.lower() not in self.VALID.keys() \
           and word.capitalize() not in self.VALID.keys():
            filtered = [x for x in self.insertvowel(word)
                        if x in self.VALID.keys()
                        and self.soundex(x) == self.soundex(word.lower())
                        and len(word) > 2
                        and len(x) > 2
                        and word.lower()[0: 2] == x.lower()[0: 2]]
            if len(filtered) == 1:
                if word == word.capitalize():
                    return [filtered[0].capitalize()]
                else:
                    return [filtered[0]]

        # Next we'll match simple vowel deletions, as long as
        # they don't change soundex category or word prefix
        # and are unambiguously valid from the larger spelling list.
        if word not in self.VALID.keys() \
           and word.lower() not in self.VALID.keys() \
           and word.capitalize() not in self.VALID.keys():
            filtered = [x for x in self.deletevowel(word)
                        if x in self.VALID.keys()
                        and self.soundex(x) == self.soundex(word.lower())
                        and len(word) > 2
                        and len(x) > 2
                        and word.lower()[0: 2] == x.lower()[0: 2]]
            if len(filtered) == 1:
                if word == word.capitalize():
                    return [filtered[0].capitalize()]
                else:
                    return [filtered[0]]

        # We'll also match simple replacements,
        # as long as they are unambiguously valid
        # from the larger spelling list.
        if len(word) > 5 \
           and word not in self.VALID.keys() \
           and word.lower() not in self.VALID.keys() \
           and word.capitalize() not in self.VALID.keys():
            filtered = [x for x in self.replaces(word)
                        if x in self.VALID.keys()
                        and (x[0: 2] == word[0: 2]
                             or self.consonants(x.lower())
                             == self.consonants(word.lower()))]
            if len(filtered) == 1 \
               and filtered[0].lower() != word.lower():
                if word == word.capitalize():
                    return [filtered[0].capitalize()]
                else:

                    return [filtered[0]]

        cnt = 0
        # At this point we'll consider soundex-type matches.
        for item in self.SORTED_ATTESTED:
            cnt += 1
            if self.soundex(item.lower()) == self.soundex(word.lower()) \
               and self.consonants(item.lower()) \
               == self.consonants(word.lower()):
                return [item]
            if abs(len(item)-len(word)) < 3 \
               and len(word) > 2 \
               and len(item) > 2 \
               and self.LevenshteinDistance(item.lower(),
                                            word.lower()) < 3 \
               and self.consonants(item.lower()) \
               == self.consonants(word.lower()):
                return [item]
            if cnt > 500:
                break

        # The general symspell algorithm generally gives too many possible
        # matches. Only fall back on it when it gives one, unique
        # valid possibilities
        suggestions = self.symspell_native.lookup(word.lower(),
                                                  Verbosity.ALL,
                                                  max_edit_distance=2,
                                                  include_unknown=True)
        filtered_suggestions = [x for x in suggestions
                                if (x.distance == 1
                                    or self.soundex(x.term) ==
                                    self.soundex(word.lower())
                                    and self.consonants(x.term) ==
                                    self.consonants(word.lower()))]
        if len(filtered_suggestions) == 1:
            corr = suggestions[0].term
            if word.lower() != corr.lower():
                if self.soundex(word) == self.soundex(corr) \
                   or (len(corr) > 2
                       and word[0:2].lower() == corr[0:2].lower()):
                    if corr.capitalize() in self.VALID \
                       and corr.lower() not in self.VALID:
                        return [corr.capitalize()]
                    else:
                        return [corr]

        # At this point, if the word and the longest prefix
        # in the word are relatively long, the most likely
        # explanation is a missing space. So let's check that
        # out using TRIE_ATTESTED
        suffix = ''
        if len(word) > 6:
            (prefix, freq) = \
                self.TRIE_ATTESTED.longest_prefix(word)
            if prefix is not None \
               and len(prefix) < len(word) \
               and (len(prefix) > 4
                    or prefix in self.fwords
                    or word[len(prefix):len(word)]
                    in self.fwords):
                suffix = word[len(prefix): len(word)]
                if word[0: len(prefix)] in self.VALID \
                   and suffix in self.VALID:
                    return [word[0: len(prefix)], suffix]

            (prefix, freq) = \
                self.TRIE_ATTESTED.longest_prefix(word.lower())
            if prefix is not None \
               and len(prefix) < len(word) \
               and (len(prefix) > 4
                    or prefix in self.fwords
                    or word.lower()[len(prefix): len(word)]
                    in self.fwords):
                suffix = word.lower()[len(prefix): len(word)]
                if word.lower()[0: len(prefix)] in self.VALID \
                   and suffix in self.VALID:
                    return [word.lower()[0: len(prefix)], suffix]

            (prefix, freq) = \
                self.TRIE_ATTESTED.longest_prefix(word.capitalize())
            if prefix is not None \
               and len(prefix) < len(word) \
               and (len(prefix) > 4
                    or prefix in self.fwords
                    or word.capitalize()[len(prefix): len(word)]
                    in self.fwords):
                suffix = word.capitalize()[len(prefix): len(word)]
                if word.capitalize()[0: len(prefix)] in self.VALID \
                   and suffix in self.VALID:
                    return [word.capitalize()[0: len(prefix)],
                            suffix]

        return None

    def load_dict(self, dictpath):
        theDict = {}
        with open(dictpath) as file:
            for line in file:
                theDict[line.rstrip()] = 1
        file.close()
        return theDict

    def document_list_to_frequency_histogram(self, documents):
        '''
        Create a frequency dictionary from a list of documents.

        Use the modified `aspell` dictionary to determine whether to
        treat the word as all-lowercase, caps, or all-caps.
        '''
        ATTESTED = {}  # <-- change to collections.defaultdict
        for document in documents:
            doc = nltk.word_tokenize(document)
            for word in doc:
                if word == word.lower():
                    if word.lower() in self.VALID:
                        if word.lower() not in ATTESTED:
                            ATTESTED[word.lower()] = 1
                        else:
                            ATTESTED[word.lower()] += 1
                elif word == word.capitalize():
                    if word.lower() in self.VALID:
                        if word.lower() not in ATTESTED:
                            ATTESTED[word.lower()] = 1
                        else:
                            ATTESTED[word.lower()] += 1
                    elif word.capitalize() in self.VALID:
                        if word.capitalize() not in ATTESTED:
                            ATTESTED[word] = 1
                        else:
                            ATTESTED[word] += 1
                elif word == word.upper():
                    if word.lower() in self.VALID:
                        if word.lower() not in ATTESTED:
                            ATTESTED[word.lower()] = 1
                        else:
                            ATTESTED[word.lower()] += 1
                    elif word.capitalize() in self.VALID:
                        if word.capitalize() not in ATTESTED:
                            ATTESTED[word.capitalize()] = 1
                        else:
                            ATTESTED[word.capitalize()] += 1
                    elif word in self.VALID:
                        if word not in ATTESTED:
                            ATTESTED[word] = 1
                        else:
                            ATTESTED[word] += 1
        return dict(sorted(ATTESTED.items(),
                    key=operator.itemgetter(1),
                    reverse=True))

    def make_trie(self, dictionary):
        t = pygtrie.Trie(separator='')
        for entry in dictionary:
            t[entry] = dictionary[entry]
        return Trie(dictionary)

    def createLocalDict(self, dictionary):
        '''
        This should be loaded from a stream.

        Needs revision.
        '''
        ts = time.time()
        theDict = 'localdict_' + str(ts) + 'txt'
        # <- should use some form of mktemp
        f = open(theDict, 'w')
        for word in dictionary:
            f.write(word + ' ' + str(dictionary[word]) + '\n')
        f.close()
        return theDict

    def known_word(self, word):
        if word == word.lower():
            if word.lower() in self.VALID:
                return [word.lower()]
            elif word.capitalize() in self.VALID:
                return [word.capitalize()]
        elif word == word.capitalize():
            if word.capitalize() in self.VALID:
                return [word.capitalize()]
            elif word.lower() in self.VALID:
                return [word.lower()]
        elif word == word.upper():
            if word.lower() in self.VALID:
                return [word.lower()]
            elif word.capitalize() in self.VALID:
                return [word.capitalize()]
            elif word in self.VALID:
                return [word]
        return []

    def make_candlist(self, word, candList):
        for x in self.SORTED_ATTESTED:

            # We won't match aggressively unless the prefix of
            # the word implies it's a safe thing to try
            # words stating with two vowels need to match
            # the first three characters
            if len(x) > 2 \
               and len(word) > 2 \
               and x.lower()[0] in 'aeiouy' \
               and x.lower()[1] in 'aeiouy' \
               and not x.lower()[0:2] == word.lower()[0:2]:
                continue
            # otherwise words need to match either the first two letters,
            # or the first and third if those are consonants
            if not (x.lower().startswith(word[0:2].lower())
               and (word[1] not in 'aeiouy'
               or word[2] in 'aeiouy')) \
               and not (len(word) > 3
                        and len(x) > 3
                        and x.lower()[0] == word.lower()[0]
                        and x.lower()[0] not in 'aeiouy'
                        and x.lower()[2] == word.lower()[2]
                        and x.lower()[2] not in 'aeiouy'
                        and x.lower()[1] in 'aeiouy'
                        and word.lower()[1] in 'aeiouy'):
                continue

            # Short words are really dangerous so we don't want to be #
            # too aggressive with them. So we check that short candidates
            # are in the same soundex as the original
            if len(word) < 5 \
               and self.soundex(word) != self.soundex(x):
                continue

            num = 0
            for line in difflib.Differ().compare(word, x):
                if '+' in line or '-' in line:
                    num += 1
            if num < 6:
                if x not in candList \
                   or num < candList[x]:
                    candList[x] = num
            num = 0
            for line in difflib.Differ().compare(word,
                                                 x.lower()):
                if '+' in line or '-' in line:
                    num += 1
            if num < 6:
                if x.lower() not in candList \
                   or num < candList[x.lower()]:
                    candList[x.lower()] = num
            num = 0
            for line in difflib.Differ().compare(word,
                                                 x.capitalize()):
                if '+' in line or '-' in line:
                    num += 1
            if num < 6:
                if x.lower not in candList \
                   or num < candList[x.capitalize()]:
                    candList[x.lower()] = num
        return candList

    def transposes(self, word):
        "All edits that are one transpose away from `word`."
        letters = 'abcdefghijklmnopqrstuvwxyzABCDEF' \
                  + 'GHIJKLMNOPQRSTUVWXYZ0123456789''-'
        splits = [(word[:i], word[i:])
                  for i in range(len(word) + 1)]

        transposes = [L + R[1] + R[0] + R[2:]
                      for L, R in splits
                      if len(R) > 1]

        transposes2 = [L + R[2] + R[1] + R[0] + R[3:]
                       for L, R in splits
                       if len(R) > 2
                       and R[0] in 'aeiouy'
                       and R[2] in 'aeiouy']

        transposes3 = [L + R[3] + R[1] + R[2] + R[0] + R[4:]
                       for L, R in splits
                       if len(R) > 3 and R[0] in 'aeiouy'
                       and R[3] in 'aeiouy']

        transposes4 = [L + R[3] + R[0] + R[1] + R[2] + R[4:]
                       for L, R in splits
                       if len(R) > 3 and R[3] in 'aeiouy']
        replacey = [L + c + R[1:]
                    for L, R in splits
                    if R for c in 'iy']
        return set(transposes + transposes2 + transposes3 + transposes4)

    def replacevowel(self, word):
        "All edits that involve replacements of vowel letters or digraphs"
        letters = 'abcdefghijklmnopqrstuvwxyzABCDEF' \
                  + 'GHIJKLMNOPQRSTUVWXYZ0123456789''-'
        splits = [(word[:i], word[i:])
                  for i in range(len(word) + 1)]
        replaces = [L + c + R[1:]
                    for L, R in splits
                    if R for c in ['a',
                                   'e',
                                   'i',
                                   'o',
                                   'u',
                                   'y',
                                   'ae',
                                   'ai',
                                   'au',
                                   'ay',
                                   'ea',
                                   'ei',
                                   'eu',
                                   'ey',
                                   'ie',
                                   'oa',
                                   'oe',
                                   'oi',
                                   'ou',
                                   'oy',
                                   'ue',
                                   'ui',
                                   'aa',
                                   'ee',
                                   'ii',
                                   'oo',
                                   'uu',
                                   'yy']]
        return set(replaces)

    def deletevowel(self, word):
        "All edits that involve vowel deletions"
        letters = 'abcdefghijklmnopqrstuvwxyzABCDEF' \
                  + 'GHIJKLMNOPQRSTUVWXYZ0123456789''-'
        splits = [(word[:i], word[i:])
                  for i in range(len(word) + 1)]
        deletes = [L + R[1:]
                   for L, R in splits
                   if R and R[0] in 'aeiouy']
        return set(deletes)

    def insertvowel(self, word):
        "All edits that involve vowel insertions"
        letters = 'abcdefghijklmnopqrstuvwxyzABCDEF' \
                  + 'GHIJKLMNOPQRSTUVWXYZ0123456789''-'
        splits = [(word[:i], word[i:])
                  for i in range(len(word) + 1)]
        inserts = [L + c + R[1:]
                   for L, R in splits
                   if R for c in 'aeiouy']
        return set(inserts)

    def replaces(self, word):
        "All edits that are one replacement or involve \
         simple letter doublings"
        letters = \
            'abcdefghijklmnopqrstuvwxyzABCDEF' \
            + 'GHIJKLMNOPQRSTUVWXYZ0123456789''-'
        splits = [(word[:i], word[i:])
                  for i in range(len(word) + 1)]
        replaces = [L + c + R[1:]
                    for L, R in splits
                    if R for c in letters]
        extra = [L + c + R
                 for L, R in splits
                 if R for c in letters
                 if len(L) > 0
                 and (c == L[len(L) - 1]
                      or (c == 't'
                          and L[len(L) - 1] == 's'))]
        return set(replaces + extra)

    def soundex(self, s):
        # Snippet taken from https://jellyfish.readthedocs.io/en/latest/
        # with modifications

        if not s:
            return ""

        # phonetic pronunciation of single letters
        alphabet = {'a': 'ey',
                    'b': 'be',
                    'c': 'sea',
                    'd': 'dee',
                    'f': 'ef',
                    'g': 'gee',
                    'd': 'dee',
                    'f': 'ef',
                    'h': 'aitch',
                    'i': 'aye',
                    'j': 'jay',
                    'k': 'kay',
                    'l': 'el',
                    'm': 'em',
                    'n': 'en',
                    'o': 'oh',
                    'p': 'pee',
                    'q': 'queue',
                    'r': 'are',
                    's': 'es',
                    't': 'tee',
                    'u': 'you',
                    'v': 'vee',
                    'w': 'doubleyou',
                    'x': 'ex',
                    'y': 'why',
                    'z': 'zee'}
        if s.lower() in alphabet:
            s = alphabet[s.lower()]

        s = unicodedata.normalize("NFKD", s)
        s = s.upper()

        replacements = (
            ("BFPV", "1"),
            ("CGJKQSXZ", "2"),
            ("DT", "3"),
            ("L", "4"),
            ("MN", "5"),
            ("R", "6"),
        )
        result = [s[0]]
        count = 1

        # find would-be replacment for first character
        for lset, sub in replacements:
            if s[0] in lset:
                last = sub
                break
        else:
            last = None

        for letter in s[1:]:
            for lset, sub in replacements:
                if letter in lset:
                    if sub != last:
                        result.append(sub)
                        count += 1
                    last = sub
                    break
            else:
                if letter != "H" and letter != "W":
                    # leave last alone if middle letter is H or W
                    last = None
            if count == 4:
                break

        result += "0" * (4 - count)
        return "".join(result)

    def consonants(self, word):
        returnString = ''
        for i in range(0, len(word)):
            if word[i] not in 'aeiouy':
                returnString += word[i]
        if returnString.endswith('s'):
            returnString = returnString[0:len(returnString)-1]
        return returnString

    def LevenshteinDistance(self, str1, str2):
        counter = {"+": 0, "-": 0}
        distance = 0
        for edit_code, *_ in ndiff(str1, str2):
            if edit_code == " ":
                distance += max(counter.values())
                counter = {"+": 0, "-": 0}
            else:
                counter[edit_code] += 1
        distance += max(counter.values())
        return distance

    # from https://stackoverflow.com/questions/
    #         46759492/syllable-count-in-python
    def sylco(self, word):
        word = word.lower()

        # exception_add are words that need extra syllables
        # exception_del are words that need less syllables

        exception_add = ['serious', 'crucial']
        exception_del = ['fortunately', 'unfortunately']

        co_one = ['cool',
                  'coach',
                  'coat',
                  'coal',
                  'count',
                  'coin',
                  'coarse',
                  'coup',
                  'coif',
                  'cook',
                  'coign',
                  'coiffe',
                  'coof',
                  'court']
        co_two = ['coapt', 'coed', 'coinci']

        pre_one = ['preach']

        syls = 0  # added syllable number
        disc = 0  # discarded syllable number

        # 1) if letters < 3: return 1
        if len(word) <= 3:
            syls = 1
            return syls

        # 2) if doesn't end with "ted" or "tes" or "ses" or "ied" or "ies",
        # discard "es" and "ed" at the end. If it has only 1 vowel or 1 set
        # of consecutive vowels, discard. (like "speed", "fled" etc.)

        if word[-2:] == "es" or word[-2:] == "ed":
            doubleAndtripple_1 = \
                len(re.findall(r'[eaoui][eaoui]', word))
            if doubleAndtripple_1 > 1 \
               or len(re.findall(r'[eaoui][^eaoui]', word)) > 1:
                if word[-3:] == "ted" \
                   or word[-3:] == "tes" \
                   or word[-3:] == "ses" \
                   or word[-3:] == "ied" \
                   or word[-3:] == "ies":
                    pass
                else:
                    disc += 1

        # 3) discard trailing "e", except where ending is "le"

        le_except = ['whole',
                     'mobile',
                     'pole',
                     'male',
                     'female',
                     'hale',
                     'pale',
                     'tale',
                     'sale',
                     'aisle',
                     'whale',
                     'while']

        if word[-1:] == "e":
            if word[-2:] == "le" \
               and word not in le_except:
                pass

            else:
                disc += 1

        # 4) check if consecutive vowels exists, triplets or pairs,
        # count them as one.

        doubleAndtripple = len(re.findall(r'[eaoui][eaoui]', word))
        tripple = len(re.findall(r'[eaoui][eaoui][eaoui]', word))
        disc += doubleAndtripple + tripple

        # 5) count remaining vowels in word.
        numVowels = len(re.findall(r'[eaoui]', word))

        # 6) add one if starts with "mc"
        if word[:2] == "mc":
            syls += 1

        # 7) add one if ends with "y" but is not surrouned by vowel
        if word[-1:] == "y" and word[-2] not in "aeoui":
            syls += 1

        # 8) add one if "y" is surrounded by non-vowels and is not in
        # the last word.

        for i, j in enumerate(word):
            if j == "y":
                if (i != 0) and (i != len(word) - 1):
                    if word[i - 1] not in "aeoui" \
                       and word[i + 1] not in "aeoui":
                        syls += 1

        # 9) if starts with "tri-" or "bi-" and is followed by a vowel,
        # add one.

        if word[:3] == "tri" and word[3] in "aeoui":
            syls += 1

        if word[:2] == "bi" and word[2] in "aeoui":
            syls += 1

        # 10) if ends with "-ian", should be counted as two syllables,
        # except for "-tian" and "-cian"

            if word[-4:] == "cian" or word[-4:] == "tian":
                pass
            else:
                syls += 1

        # 11) if starts with "co-" and is followed by a vowel, check
        # if exists in the double syllable dictionary, if not, check
        # if in single dictionary and act accordingly.

        if word[:2] == "co" and word[2] in 'eaoui':

            if word[:4] in co_two \
               or word[:5] in co_two \
               or word[:6] in co_two:
                syls += 1
            elif (word[:4] in co_one
                  or word[:5] in co_one
                  or word[:6] in co_one):
                pass
            else:
                syls += 1

        # 12) if starts with "pre-" and is followed by a vowel, check
        # if exists in the double syllable dictionary, if not, check
        # if in single dictionary and act accordingly.

        if word[:3] == "pre" and word[3] in 'eaoui':
            if word[:6] in pre_one:
                pass
            else:
                syls += 1

        # 13) check for "-n't" and cross match with dictionary to add syllable.

        negative = ["doesn't", "isn't", "shouldn't", "couldn't", "wouldn't"]

        if word[-3:] == "n't":
            if word in negative:
                syls += 1
            else:
                pass

        # 14) Handling the exceptional words.

        if word in exception_del:
            disc += 1

        if word in exception_add:
            syls += 1

        # calculate the output
        return numVowels - disc + syls

    def adjust_punctuation(self, text):
        '''
        This cleans up occasional messiness around punctuation from
        tokenizers.

        To do: Make data file
        '''
        text = text.replace('."', '.\'\'')
        text = text.replace('. "', '. \'\'')
        text = text.replace('.', '. ')
        text = text.replace(' - ', ' -- ')
        text = text.replace('-', ' - ')
        text = text.replace(' \'', ' \' ')
        text = text.replace('\'', ' \' ')
        text = text.replace('/', ' / ')
        text = text.replace('  ', ' ')
        text = text.replace('\' s ', '\'s ')
        text = text.replace('\' d ', '\'d ')
        text = text.replace('\' ll ', '\'ll ')
        text = text.replace('\' ve ', '\'ve ')
        text = text.replace('isn \' t', 'isn\'t ')
        text = text.replace('aren \' t', 'aren\'t ')
        text = text.replace('wasn \' t', 'wasn\'t ')
        text = text.replace('weren \' t', 'weren\'t ')
        text = text.replace('don \' t', 'don\'t ')
        text = text.replace('doesn \' t', 'doesn\'t')
        text = text.replace('didn \' t', 'didn\'t')
        text = text.replace('won \' t', 'won\'t')
        text = text.replace('can \' t', 'can\'t')
        text = text.replace('mustn \' t', 'mustn\'t')
        text = text.replace('wouldn \' t', 'wouldn\'t')
        text = text.replace('couldn \' t', 'couldn\'t')
        text = text.replace('mayn \' t', 'mayn\'t')
        text = text.replace('shouldn \' t', 'shouldn\'t')
        text = text.replace('mightn \' t', 'mightn\'t')
        text = text.replace('daren \' t', 'daren\'t')
        text = text.replace('oughtn \' t', 'oughtn\'t')
        text = text.replace('is n\'t', 'isn\'t ')
        text = text.replace('are n\'t', 'aren\'t ')
        text = text.replace('was n\'t', 'wasn\'t ')
        text = text.replace('were n\'t', 'weren\'t ')
        text = text.replace('do n\'t', 'don\'t ')
        text = text.replace('does n\'t', 'doesn\'t.')
        text = text.replace('did n\'t', 'didn\'t.')
        text = text.replace('wo n\'t', 'won\'t')
        text = text.replace('ca n\'t', 'can\'t')
        text = text.replace('must n\'t', 'mustn\'t')
        text = text.replace('would n\'t', 'wouldn\'t')
        text = text.replace('could n\'t', 'couldn\'t')
        text = text.replace('may n\'t', 'mayn\'t')
        text = text.replace('should n\'t', 'shouldn\'t')
        text = text.replace('might n\'t', 'mightn\'t')
        text = text.replace('dare n\'t', 'daren\'t')
        text = text.replace('ought n\'t', 'oughtn\'t')

        return self.fix_abbrevs(text)

    def fix_abbrevs(self, text):
        pattern = \
            '( [A-Za-z]|Mr|Mrs|Ms|Dr|Sr|Jr|Ph|Ms)\.( [A-Za-z](\.)?)+[ ]'
        results = re.finditer(pattern, text)
        for item in results:
            change = item.group().replace(' ', '')
            text = text.replace(item.group(),
                                ' ' + change + ' ')
        return text

    def untokenize(self, doc):
        text = ''
        for token in doc:
            text += ' ' + token
        return self.normalize(text)

    def normalize(self, text):
        text = text.replace(' . . . ', ' ... ')
        text = text.replace(' . .', '. ')
        text = text.replace(': .', ': ')
        text = text.replace('? .', '? ')
        text = text.replace('! .', '! ')
        text = text.replace('. \' .', '.\'')
        text = text.replace('. " .', '."')
        text = text.replace('. ” .', '.”')
        text = text.replace(' . ', '. ')
        text = text.replace(' ? ', '? ')
        text = text.replace(' ! ', '! ')
        text = text.replace(' : ', ': ')
        text = text.replace(' - ', '-')
        text = text.replace(' , ', ', ')
        text = text.replace('`', '\'')
        text = text.replace(' ' ' ', ' \'\' ')
        text = text.replace(' ; ', '; ')
        text = text.replace(' % ', ' percent ')
        text = text.replace(' \' s ', '\'s ')
        text = text.replace(' \' s.', '\'s.')
        text = text.replace(' \' s,', '\'s,')
        text = text.replace(' \' s?', '\'s?')
        text = text.replace(' \' s!', '\'s!')
        text = text.replace(' \' s;', '\'s;')
        text = text.replace(' \' \' ', ' \'\' ')
        text = text.replace("ai n\'t", "ai\'nt")
        text = text.replace(" \' d ", "\'d ")
        text = text.replace(" \' d.", "\'d.")
        text = text.replace(" \' d,", "\'d,")
        text = text.replace(" \' d!", "\'d!")
        text = text.replace(" \' d?", "\'d?")
        text = text.replace(" \' d;", "\'d;")
        text = text.replace(" \' m ", "\'m ")
        text = text.replace(" \' m.", "\'m.")
        text = text.replace(" \' m,", "\'m,")
        text = text.replace(" \' m!", "\'m!")
        text = text.replace(" \' m?", "\'m?")
        text = text.replace(" \' m;", "\'m;")
        text = text.replace(" \' ve", "\'ve")
        text = text.replace(" \' re", "\'re")
        text = text.replace(" \' ll", "\'ll")
        text = text.replace(" don \' t ", " don\'t ")
        text = text.replace(" doesn \' t ", " doesn\'t ")
        text = text.replace(" didn \' t ", " didn\'t ")
        text = text.replace(" won \' t ", " won\'t ")
        text = text.replace(" it \' s ", " it\'s ")
        text = text.replace(" wo n\'t ", " won\'t ")
        text = text.replace(" can \' t ", " can\'t ")
        text = text.replace(" ca n\'t ", " can\'t ")
        text = text.replace(" shouldn \' t ", " shouldn\'t ")
        text = text.replace(" wouldn \' t ", " wouldn\'t ")
        text = text.replace(" mightn \' t ", " mightn\'t ")
        text = text.replace(" mustn \' t ", " mustn\'t ")
        text = text.replace(" oughtn \' t ", " oughtn\'t ")
        text = text.replace(" daren \' t ", " daren\'t ")
        text = text.replace(" isn \' t ", " isn\'t ")
        text = text.replace(" doesn \' t ", " doesn\'t ")
        text = text.replace(" wasn \' t ", " wasn\'t ")
        text = text.replace(" aren \' t ", " aren\'t ")
        text = text.replace(" weren \' t ", " weren\'t ")
        text = text.replace(" n \' t ", "n\'t ")
        text = text.replace(" n\'t ", "n\'t ")
        text = text.replace(" nt ", "n\'t ")
        text = text.replace(" n \' t.", "n\'t.")
        text = text.replace(" n\'t.", "n\'t.")
        text = text.replace(" nt.", "n\'t.")
        text = text.replace(" n \' t?", "n\'t?")
        text = text.replace(" n\'t?", "n\'t?")
        text = text.replace(" nt?.", "n\'t?")
        text = text.replace(" n \' t!", "n\'t?")
        text = text.replace(" n\'t!", "n\'t?")
        text = text.replace(" nt!.", "n\'t!")
        text = text.replace(" n \' t,", "n\'t,")
        text = text.replace(" n\'t,", "n\'t,")
        text = text.replace(" nt,", "n\'t,")
        text = text.replace(" n \' t;", "n\'t;")
        text = text.replace(" n\'t;", "n\'t;")
        text = text.replace(" nt;", "n\'t;")
        text = text.replace(' \' t ', '\'t ')
        text = text.replace(' \' re ', '\'re ')
        text = text.replace(' \' ve ', '\'ve ')
        text = text.replace(' \' ll ', '\'ll ')
        text = text.replace(' \' d ', '\'d ')
        text = text.replace(' \' m ', '\'m ')
        text = text.replace(' ( ', ' (')
        text = text.replace(' ) ', ') ')
        text = text.replace(' .', '.')
        text = text.replace(' ?', '?')
        text = text.replace(' !', '!')
        text = text.replace(' %', '%')
        text = text.replace('$ ', '$')
        text = text.replace('# ', '#')
        text = text.replace('\' \'', '\'\'')
        text = text.replace(' . . . ', ' ... ')
        text = text.replace(' .', '.')
        text = text.replace(' ?', '?')
        text = text.replace(' !', '!')
        text = text.replace(' :', ':')
        text = text.replace(' -', '-')
        text = text.replace(' ,', ',')

        text = text.replace('..<>', '.\n')
        # special coding for paragraph break we need to get rid of

        text = text.replace('<>', '\n')
        # special coding for paragraph break we need to get rid of

        text = text.replace(':.', '!')
        text = text.replace('!.', '!')
        text = text.replace('?.', '?')
        text = text.replace('t. v.', 'tv')
        text = text.replace('T. V.', 'tv')
        text = text.replace('. 0', ' .0')
        text = text.replace('. 1', ' .1')
        text = text.replace('. 2', ' .2')
        text = text.replace('. 3', ' .3')
        text = text.replace('. 4', ' .4')
        text = text.replace('. 5', ' .5')
        text = text.replace('. 6', ' .6')
        text = text.replace('. 7', ' .7')
        text = text.replace('. 8', ' .8')
        text = text.replace('. 9', ' .9')
        text = text.replace(' un ', ' UN ')
        text = text.replace(' etc .', ' etc.')
        text = text.replace('n ’ t', 'n\'t')
        text = text.replace(' ’ s ', '\'s ')
        text = text.replace(' ’ d ', '\'d ')
        text = text.replace(' ’ m ', '\'m ')
        text = text.replace(' ’ ll ', '\'ll ')
        text = text.replace(' ’ ve ', '\'ve ')
        text = text.replace(' ’ re ', '\'re ')
        text = text.replace('“ ', '“')
        text = text.replace(' ”', '”')
        text = text.replace('etc.', 'etc')
        text = text.replace('etc .', 'etc')
        text = text.replace('’', '\'')
        text = text.replace(': \'\' \n', ':\n')
        text = text.replace('”. \n', '”\n')

        return text

    def topn_words(self, left, right, n, tokenizer, model):
        from torch.nn import functional as F
        text = left + tokenizer.mask_token + right
        input = tokenizer.encode_plus(text, return_tensors="pt")
        mask_index = torch.where(input["input_ids"][0]
                                 == tokenizer.mask_token_id)
        output = model(**input)
        logits = output.logits
        softmax = F.softmax(logits, dim=-1)
        mask_word = softmax[0, mask_index, :]
        returnSet = []
        for token in torch.topk(mask_word, n, dim=1)[1][0]:
            try:
                word = tokenizer.decode([token])
            except Exception as e:
                print(e)
                continue
            returnSet.append(word)
        return returnSet

    def f(self, obj, enc):
        return str(obj).encode(enc,
                               errors='backslashreplace').decode(enc)

    def uprint(self, *objects, sep=' ', end='\n', file=sys.stdout):
        enc = file.encoding
        if enc == 'UTF-8':
            print(*objects, sep=sep, end=end, file=file)
        else:
            try:
                print(*map(f, objects), sep=sep, end=end, file=file)
            except Exception as e:
                print('error printing decoded word\n', e)

    def split_by_sentence(self, tokens, string_output):
        sentences = []
        current_sentence = []
        ct = 0
        last_token = ''
        if string_output:
            paras = ''.join(tokens).split('\n')
            tokens = []
            for para in paras:
                tk = nltk.WordPunctTokenizer()
                tokens += tk.tokenize(para) + ['\n']
                tokens[0] = tokens[0].capitalize()

        for i in range(0, len(tokens)):
            token = tokens[i]
            if last_token == '.' \
               or last_token == '?' \
               or last_token == '!' \
               or last_token == '\n':
                token = token.capitalize()
            if token == '.' or token == '?' \
               or token == '!' \
               or token.endswith('.') \
               or token == '<>' \
               or token == '--':
                if token == '.' \
                   and re.match('(^[A-Za-z]|[0-9,]+|Dr|Mr|Mrs|Ms|Ph|Ms)$',
                                last_token):
                    current_sentence.append(token)
                else:
                    current_sentence.append(token)
                    if len(current_sentence) > 0:
                        sentences.append(current_sentence)
                    current_sentence = []
            else:
                current_sentence.append(token)
            ct += 1
            last_token = token
        if len(current_sentence) > 0 \
           and self.numContentTokens(current_sentence) > 0:
            sentences.append(current_sentence)
        return sentences

    def numContentTokens(self, tokens):
        ct = 0

        for token in tokens:
            if re.match('[A-Za-z0-9]+', token) \
               and len(token) > 3 \
               and token not in self.stopwords:
                ct += 1
        return ct

    def split_sentence_into_chunks(self, tokens):
        chunks = []
        current_chunk = ''
        punct = [',', ';', '(', ')']
        post_delimiters = [',', ';', ')']
        pre_delimiters = ['(', 'but', 'because', 'so', 'if']
        soft_delimiters = ['and', 'or', 'that']
        for i in range(0, len(tokens)):
            token = tokens[i]
            if token in pre_delimiters \
               and self.numContentTokens(current_chunk.split(' ')) > 3 \
               and self.numContentTokens(tokens[i:]) > 3 \
               and not tokens[i + 1] in punct \
               and not tokens[i + 2] in punct \
               and not tokens[i + 3] in punct:
                chunks.append(current_chunk)
                current_chunk = token
            elif (token in post_delimiters
                  and self.numContentTokens(current_chunk.split(' ')) > 3
                  and self.numContentTokens(tokens[i:]) > 3
                  and not tokens[i + 1] in punct
                  and not tokens[i + 2] in punct
                  and not tokens[i + 3] in punct):
                current_chunk += ' ' + token
                chunks.append(''.join(self.normalize(current_chunk)))
                current_chunk = ''
            elif (token in soft_delimiters
                  and self.numContentTokens(current_chunk.split(' ')) > 5
                  and self.numContentTokens(tokens[i:]) > 4
                  and not tokens[i + 1] in punct
                  and not tokens[i + 2] in punct
                  and not tokens[i + 3] in punct
                  and not tokens[i + 4] in punct
                  and not tokens[i + 2] in soft_delimiters
                  and not tokens[i + 3] in soft_delimiters):
                chunks.append(''.join(
                              self.normalize(current_chunk)))
                current_chunk = token
            else:
                current_chunk += ' ' + token
        if len(current_chunk) > 0:
            chunks.append(''.join(
                          self.normalize(current_chunk)))
        return chunks
