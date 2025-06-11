# GenTextDetect
Repository for AI-generated vs human-authored texts: comparative analysis of datasets and NLP methods MSC thesis.

## Repository Structure

1. `./logs/` - Training history for all experiments
2. `./notebooks` - Jupyter Notebooks used for development
3. `./plots/` - Plots used in the thesis
4. `./predictions/` - Prediction of all evaludated models on the test datasets
5. `./src/` - Source code

All Jupyter Notebooks were used for development of the solutions. They may contain errors, unused plots, or experimental solutions.

## SLURM

All `runner.sh` scripts were used to submit jobs to the SLURM Queuing System.

## Data sources

https://www.kaggle.com/datasets/kazanova/sentiment140 \
https://www.kaggle.com/datasets/smagnan/1-million-reddit-comments-from-40-subreddits \
https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus \
https://www.kaggle.com/datasets/benjaminawd/new-york-times-articles-comments-2020 \
https://www.kaggle.com/datasets/thedrcat/daigt-external-train-dataset \
https://huggingface.co/datasets/liamdugan/raid \
https://huggingface.co/datasets/EdinburghNLP/xsum \
https://huggingface.co/datasets/euclaise/writingprompts \
https://huggingface.co/datasets/google-research-datasets/natural_questions