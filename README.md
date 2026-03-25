# CSL 7640 - Assignment 2

This assignment consists of two problems.

Problem 1 is about learning word embeddings using Word2Vec. Data was collected from IIT Jodhpur related sources like PDFs and text files. The raw text was cleaned by removing noise, converting to lowercase, removing stopwords, and splitting into sentences.

After preprocessing, Word2Vec models were trained using both CBOW and Skip-gram with different parameters such as embedding dimension and window size. The models were evaluated using nearest neighbors and analogy tasks. PCA and t-SNE were also used to visualize the embeddings and observe clustering of similar words.

Problem 2 is about character-level name generation. In this part, different sequence models were used to generate Indian names.

The models implemented are:
- Vanilla RNN (implemented from scratch)
- BiLSTM
- Attention + RNN

Each name is treated as a sequence of characters. The model learns to predict the next character step by step using a start token '<' and an end token '>'.

The Vanilla RNN is simple but sometimes produces noisy results because it cannot capture long dependencies well. BiLSTM performs better since it processes the sequence in both directions. The Attention-based model gives the best results as it focuses on important parts of the sequence.

How to run the code:

For Problem 1:
python collect_data.py
python preprocess.py
python train_word2vec.py
python semantic_analysis.py
python visualization.py

For Problem 2:
python vanilla_rnn.py
python bilstm.py
python attention_rnn.py

Notes:
- Outputs are saved in respective folders
- Results may vary slightly due to randomness
- Small dataset can lead to some noisy outputs
