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
import shutil
from pathlib import Path

from distutils.sysconfig import get_python_lib
import sysconfig


class data:
    def __init__(self, args):

        if args.install or args.develop:
            self.extra_install_commands()

    
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
            self.grab_subwordbert(datapath)


            
    def grab_subwordbert(self, datapath, cleanup_files=True):
        '''
        This is a datafile that neuspell needs that somehow got corrupted
        in its original location. We have a copy we're using to fix this
        problem at the location specified here. To fix once the issue
        is properly addressed.
        '''

        # Added to assist in debugging with repeated uses of the
        # package.
        if (not os.path.exists('subwordbert.tar.gz')):
            os.system('wget \
              https://s3.amazonaws.com/mitros.org/p/ets/subwordbert.tar.gz')
            os.system('tar -xzf subwordbert.tar.gz')

        # TO-DO: This definition of siteloc may not always work. We need
        # to find a better way to locate the /data/checkpoints subdirectory
        # that the BERT application is expecting to find, no matter what kind
        # of virtual or real environment we have. Hopefully this bug gets
        # fixed upstream and this bit of hacky code can go away.

        # Make the data/checkpoint/subwordbert directory it is
        # not already present.
        
        #os.makedirs(os.path.basename(modelpath), exist_ok=True)
        # os.makedirs(os.path.dirname(modelpath), exist_ok=True)
        if (not os.path.exists(datapath)):
            print("Making path: {}".format(datapath))
            os.makedirs(datapath, exist_ok=True)

        
        # Path rename does not work if you are moving across filesystems
        # or across partitions/drives.  Thus we use the shtuil doing a
        # copy followed by a remove.  In future if we can test for this
        # case that would help to reduce the time but given how rarely
        # we do this it is no big deal.

        # Path("subwordbert-probwordnoise/model.pth.tar").rename(modelpath)
        # Path("subwordbert-probwordnoise/vocab.pkl").rename(vocabpath)

        print("Moving files.")
        shutil.copy("subwordbert-probwordnoise/model.pth.tar", datapath)
        shutil.copy("subwordbert-probwordnoise/vocab.pkl", datapath)

        
        # Handle file cleaning if required.
        if (cleanup_files == True):
            print("Cleaning up downloads.")
            os.remove("subwordbert-probwordnoise/model.pth.tar")
            os.remove("subwordbert-probwordnoise/vocab.pkl")
            os.remove('subwordbert.tar.gz')
            os.rmdir('subwordbert-probwordnoise')

            


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
