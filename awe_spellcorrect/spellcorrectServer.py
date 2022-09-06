#!/usr/bin/env python3.10
# Copyright 2022 Educational Testing Services

import asyncio
import enum
import json
import os
import re
import sys
import websockets
import neuspell
import awe_spellcorrect.spellCorrect
from importlib import resources
from neuspell import BertChecker

checker = BertChecker()
checker.from_pretrained()


if sys.version_info[0] == 3:
    xrange = range

transformer_types = enum.Enum('Transformer Type', "NONE BERT NEUSPELL")


class spellcorrectServer:

    # Initialize the spellchecker
    cs = None
    ASPELL_PATH = None
    PYSPELL_PATH = None

    def __init__(self):

        self.ASPELL_PATH = \
           resources.path('awe_spellcorrect', 'aspell.txt')

        self.PYSPELL_PATH = \
            resources.path('symspellpy',
                           'frequency_dictionary_en_82_765.txt')

        if not os.path.exists(self.ASPELL_PATH) \
           or not os.path.exists(self.PYSPELL_PATH):
            raise Error(
                "Trying to load AWE Workbench Lexicon Module \
                 without supporting datafiles")

        self.cs = awe_spellcorrect.spellCorrect.SpellCorrect(
            self.ASPELL_PATH, self.PYSPELL_PATH)
        asyncio.get_event_loop().run_until_complete(
            websockets.serve(self.run_spellchecker, 'localhost', 8765))
        print('running spell corrector')
        asyncio.get_event_loop().run_forever()
        print('spell corrector died')

    async def kill(self, websocket):
        await websocket.close()
        exit()

    async def run_spellchecker(self, websocket, path):
        async for message in websocket:

            messagelist = json.loads(message)
            if messagelist == ['kill()']:
                await self.kill(websocket)
            else:
                (texts, textids, textchunks, chunkids) = \
                    self.cs.spellcheck_corpus(messagelist,
                                              transformer_types.NEUSPELL,
                                              False,
                                              'native')
                await websocket.send(json.dumps(texts))


if __name__ == '__main__':
    spc = spellcorrectServer()