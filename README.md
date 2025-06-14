# Göttingen University | Winter Semester 2024/25

This repository contains the implementation and reports for projects of the Language as Data course at Göttingen University. The projects focuses on working with pre-trained language models , including monolingual (English) and multilingual models , to perform tasks such as sentiment classification, fine-tuning strategies, and cross-lingual transfer.

# languageAsData
This course introduces  the characteristics of language data and the associated challenges for representation learning. Natural language is a powerful and creative means of communication: it follows rules only to make exceptions, and it evolves over time and from domain to domain. Language signals are highly ambiguous and the form—meaning mapping can only be interpreted in context. In this course, we acquired the conceptual knowledge to analyze structure in language data and understand the methodological assumptions underlying representation learning in large language models.

The course consists of three parts: 
1) Analyzing Language:  introducing the characteristics of language as a signal 
2) Modeling Language: introducing the foundations of language modeling
3) Using Language Models: introducing techniques for applying LLMs for transfer (e.g. finetuning, prompting, adapting, etc).

# What I Did in Each Phase

## Phase 1: Task Analysis (GLUE Benchmark - SST-2)

I analyzed the Stanford Sentiment Treebank (SST-2) task from the GLUE benchmark.
Reviewed dataset composition, annotation process, and inter-annotator agreement.
Manually categorized 20 instances into "easy" and "difficult" based on linguistic complexity and ambiguity.

## Phase 2: Fine-Tuning GPT-2

I implemented and compared multiple fine-tuning strategies:
1) Full fine-tuning
2) Head-only tuning
3) Partial fine-tuning
4) LoRA (Low-Rank Adaptation)
5) Training from scratch (no pre-training)
In the end, I evaluated each method using accuracy, training time, and test performance; and conducted error analysis on manually selected examples.


## Phase 3: Multilingual Modeling & Translation

I Prompted a multilingual model (e.g., mBART or mT5) to translate Persian sentences into English.
Then I generated translations and evaluated them using BLEU score and human judgment for fluency and contextual adequacy.
And in the end, I analyzed consistency, coherence, and challenges in translation of complex syntactic structures.

 
## Tools Used

- HuggingFace Transformers
- PyTorch
- GPT-2 and multilingual models
- BLEU scoring
- Krippendorff’s Alpha for inter-annotator agreement
- Python scripts for preprocessing, evaluation, and visualization
