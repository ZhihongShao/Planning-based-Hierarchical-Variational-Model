import pickle
import Vocabulary
import subprocess
import Config
from subprocess import PIPE

if __name__ == "__main__":
    config = Config.config
    subprocess.run("python word2vec.py --corpus ./data/processed/corpus.txt --result\_dir ./data/processed/ --min\_count 1".split(), stdout=PIPE, stderr=PIPE)
    pickle.dump(Vocabulary.Vocabulary(), open(config.vocab_file, "wb"))
