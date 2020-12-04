import os
import argparse
import numpy as np

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Monodepthv2 options")

    # PATHS
    parser.add_argument("--data_path",
                                type=str,
                                help="path to the training data")
    parser.add_argument("--out_path",
                                type=str,
                                help="path to the training data")

    args = parser.parse_args()

    for split in ['train','test','val']:
        split_path = os.path.join(args.data_path, split)

        data = []

        for env in os.listdir(split_path):
            env_path = os.path.join(split_path, env, 'Easy')
            for seq in os.listdir(env_path):
                frames = [f.replace(".png", "") for f in os.listdir(os.path.join(env_path, seq, 'cam0','data'))]

                for f in frames:

                    data.append([os.path.join(split, env, 'Easy',seq,'cam0','data'), f, 'l'])
                    data.append([os.path.join(split, env, 'Easy',seq,'cam1','data'), f, 'r'])

        if not os.path.isdir('splits/tartanair/'): os.makedirs('splits/tartanair/')
        data = np.asarray(data)
        print(data)
        np.savetxt('splits/tartanair/' + split + '_files.txt', data, fmt="%s %s %s")