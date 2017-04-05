from setuptools import find_packages,setup

setup(name='Movienator',
      version='0.1dev',
      author="Nicolo' Brandizzi",
      packages=find_packages(exclude=['Debug']),
      url='https://github.com/nicofirst1/progetto.git',
      description='Sentiment analisys from movie reviews',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Machine learning :: Sentiment analisys',
        'Programming Language :: Python :: 3+',
        ],
      install_requires=['pandas','sklearn','nltk','numpy','gensim','matplotlib'],

      )

