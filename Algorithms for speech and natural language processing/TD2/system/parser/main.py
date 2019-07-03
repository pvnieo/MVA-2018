# stdlib
import argparse
# project
from pcfg import PCFG


if __name__ == "__main__":
    print("Welcome to my parser!")
    print("Please wait while loading the model ...")
    pcfg = PCFG()
    pcfg.from_path('sequoia-corpus+fct.mrg_strict.txt')
    pcfg.fit()
    print("Model loaded!")
    while True:
        print("Please enter phrase to parse!")
        phrase = str(input('>>> '))
        tokenized = phrase.split()
        parsed = pcfg.pcky(tokenized)
        if not parsed:
            print("Sorry, we couldn't parse your line :(")
        else:
            print(parsed)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
