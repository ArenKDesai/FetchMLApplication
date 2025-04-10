# Fetch ML Apprentice Take Home Exercise
## Solution
The dockerfile can be run like so:

`docker run -it imagename "your sentence here"`

with these optional arguments:
* -f --file
    * the sentence you passed was a filename. Iterates through the files separated by newline (\n)
* -sc --sentence-classification 
    * performs sentence classification
* -sa --sentiment-analysis
    * performs sentiment analysis

For example:

```
docker build -t fetchML .
docker run --gpus all --ipc=host -it fetchML examples.txt -f -sc -sa
```
`-it` is optional, but it allows you to see the `stdout` before the program terminates. 

## Explanations
### Task 1: Sentence Transformer Implementation
HuggingFace's `AutoTokenizer` streamlines this step well. 
### Final Notes
I speant over 2 hours 