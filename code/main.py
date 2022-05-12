import os
import argparse

from train import Trainer
from inference import Inferencer


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('command',help="'train' or 'generate'")
    parser.add_argument('--image', help='Name of input image', required=True)
    args = parser.parse_args()
    
    if args.command == 'train':
        assert os.path.exists(args.image), 'Training image not found !'
        trainer = Trainer()
        trainer.train(training_image=args.image)

    elif args.command == 'generate':
        assert os.path.exists(args.image), 'Reference image not found !'
        inferencer = Inferencer()
        inferencer.inference(task="random_sample",rimg=args.image)

    else:
        print('Your arguments are not correct. Try somethign like "python main.py train --image ../seki.jpg"')
        
if __name__ == '__main__':
    main()