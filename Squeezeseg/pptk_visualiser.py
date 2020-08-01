if __name__ == "__main__":
    import pickle
    import argparse
    import os
    import pptk
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()
    
    name = os.path.splitext(os.path.basename(args.filename))[0]
    np_pcd = np.load(args.filename)

    np_single = np_pcd[:, :3] * np.tile([1,-1,1], (np_pcd.shape[0], 1))
    print(np_single)
    v = pptk.viewer(np_single, np_pcd[:,3])