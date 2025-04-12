#!/usr/bin/env python3
import argparse
from task1 import embed_sentences
from task2 import classify_recognize

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
    sentence_embeddings, attention_weights = embed_sentences(args.sentence, args.file)

    # Task 2
    print("\n\n    ----    TASK 2    ----    ")
    results = classify_recognize(args.sentence, args.file, args.sentence_classification, args.named_entity_recognition)

if __name__ == "__main__":
    main()