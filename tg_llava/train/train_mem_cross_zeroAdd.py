import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tg_llava.train.train_cross_zeroAdd_learnable_query import train

if __name__ == "__main__":
    train()
