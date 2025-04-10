#!/usr/bin/env python3
import argparse
from task1 import encode_sentence

def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('sentence', type=str, help='sentence for encoder (req)')
    parser.add_argument('-f', '--file', action='store_true', 
                        help='bool - the sentence input was a filename (opt)')
    parser.add_argument('-s', '--sentence-classification', action='store_true', 
                        help='perform sentence classification (opt)')
    parser.add_argument('-n', '--named-entity-recognition', action='store_true',
                        help='perform sentiment analysis')
    args = parser.parse_args()
    
    # Task 1
    print("    ----    TASK 1    ----    ")
    encode_sentence(args.sentence, args.file)

if __name__ == "__main__":
    main()