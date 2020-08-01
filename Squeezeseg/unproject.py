import argparse
import os
import numpy as np
import pptk
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def append_labels(points, prediction, W=512, H=64, C=5, phi_range=60, theta_range=40):
    """
    Outputs labels (0: non-obstacle or 1: obstacle) for each point in the point cloud as an array,
    in the same order as input points.
    """
    # Copied from lidar_projection: To unify into separate module

    dphi = np.radians(phi_range)/W           #Phi = Horizontal angle (Azimuth): x-z plane
    dtheta = np.radians(theta_range)/H       #Theta = Vertical angle (Elevation): y-z plane

    x_lidar = points[:, 0]
    y_lidar = points[:, 1]
    z_lidar = points[:, 2]

    # Distance relative to origin
    d = np.sqrt(x_lidar ** 2 + y_lidar ** 2 + z_lidar ** 2)
    r = np.sqrt(x_lidar ** 2 + z_lidar ** 2)

    theta = np.arcsin(y_lidar/d)
    phi = np.arcsin(-x_lidar/r)

    max_phi = np.degrees(max(phi))
    min_phi = np.degrees(min(phi))
    max_theta = np.degrees(max(theta))
    min_theta = np.degrees(min(theta))

    #Discretisation
    theta = (theta/dtheta + H//2).astype(int)
    phi = ((phi/dphi) + W//2).astype(int)

    #Within range
    theta[theta<0] = 0
    theta[theta>=H] = H - 1

    phi[phi<0] = 0
    phi[phi>=W] = W -1

    print(theta.shape)
    print(phi.shape)

    true_labels = []

    for t, p in zip(theta, phi):
        true_labels.append(prediction[t, p])

    return np.asarray(true_labels)

def occupancy_grid():
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('point_cloud')
    parser.add_argument('prediction')
    parser.add_argument('--vis', dest='vis', default=False, action='store_true')
    parser.add_argument('--metrics', dest='metrics', default=False, action='store_true')

    args = parser.parse_args()

    pc = np.load(args.point_cloud)
    prediction = np.load(args.prediction)

    # Check if number of columns is greater than 3
    assert pc.shape[1] >= 3

    # Extract pure x, y, z coordinates
    pc_coord = pc[:,:3]

    labels = append_labels(pc_coord, prediction)

    if args.vis:
        v = pptk.viewer(pc_coord, labels)

    # Add saving functionality

    if args.metrics:
        # Calculate accuracy
        cm = confusion_matrix(pc[:, 3].T, labels)
        print(cm)

        acc = accuracy_score(pc[:, 3].T, labels)
        print(acc)

        cr = classification_report(pc[:, 3].T, labels)
        print(cr)


    # Add occupancy_grid functionality    
    #augmented_pc = np.concatenate((pc, labels), axis=0)
    #oc = occupancy_grid()


    

    

