#!/usr/bin/env python3.10
# Copyright 2022, Educational Testing Service

import argparse

from distutils.command.build import build
from distutils.command.build import build as _build
from distutils.command.install import install
from distutils.command.install import install as _install

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.develop import develop as _develop

import os
import os.path
from pathlib import Path

from distutils.sysconfig import get_python_lib
import sysconfig


class data:
    def grab_subwordbert(self, vocabpath, modelpath):
        '''
        This is a datafile that neuspell needs that somehow got corrupted
        in its original location. We have a copy we're using to fix this
        problem at the location specified here. To fix once the issue
        is properly addressed.
        '''
        os.system('wget \
        https://s3.amazonaws.com/mitros.org/p/ets/subwordbert.tar.gz')
        os.system('tar -xzf subwordbert.tar.gz')

        # TO-DO: This definition of siteloc may not always work. We need
        # to find a better way to locate the /data/checkpoints subdirectory
        # that the BERT application is expecting to find, no matter what kind
        # of virtual or real environment we have. Hopefully this bug gets
        # fixed upstream and this bit of hacky code can go away.
        os.makedirs(os.path.basename(modelpath), exist_ok=True)
        Path("subwordbert-probwordnoise/model.pth.tar").rename(modelpath)
        Path("subwordbert-probwordnoise/vocab.pkl").rename(vocabpath)
        os.remove('subwordbert.tar.gz')
        os.rmdir('subwordbert-probwordnoise')

    def extra_install_commands(self):
        '''
        Grab extra data files. For now, `subwordbert`
        '''
        # Typically, something like:
        # ~/.virtualenvs/venv/lib/python3.10/site-packages
        siteloc = sysconfig.get_paths()["purelib"]
        datapath = siteloc + '/data/checkpoints/subwordbert-probwordnoise/'
        vocabpath = datapath + 'vocab.pkl'
        modelpath = datapath + 'model.pth.tar'

        if not os.path.exists(vocabpath) or not os.path.exists(vocabpath):
            print("Missing data files. Grabbing:")
            print(vocabpath)
            print(modelpath)
            self.grab_subwordbert(vocabpath, modelpath)

    def __init__(self, args):
        if args.install or args.develop:
            self.extra_install_commands()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run AWE \
                                     Workbench data download')
    parser.add_argument("-d",
                        "--develop",
                        help="Runs the data downloads to the development \
                              rather than the build location.",
                        action="store_true")
    parser.add_argument("-i",
                        "--install",
                        help="Runs the data downloads to the development \
                              rather than the build location.",
                        action="store_true")

    args = parser.parse_args()

    data(args)
