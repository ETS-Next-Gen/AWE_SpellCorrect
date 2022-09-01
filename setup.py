#!/usr/bin/env python3.9

from distutils.command.build import build
from distutils.command.build import build as _build
from distutils.command.install import install
from distutils.command.install import install as _install
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.develop import develop as _develop
           
class develop(develop):
    def run(self):        
        _develop.run(self)
        print('develop')

class build_(build):
    def run(self):
        _build.run(self)
        print('build')
        
class install_(install):
    def run(self):
        _install.run(self)               
        print('install')

if __name__ == '__main__':
 
    setup(
        # see 'setup.cfg'
        cmdclass={
            'build': build_,
            'install': install_,
            'develop': develop,
        }
)


