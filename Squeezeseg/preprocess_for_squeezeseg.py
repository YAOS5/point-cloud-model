#Adapted from: https://github.com/BichenWuUCB/SqueezeSeg/issues/37

def lidar_to_2d_front_view(points, W=512, H=64, C=5, phi_range=60, theta_range=40):
    import numpy as np
    import os

    #Temporary fix for wall classification

    
    dphi = np.radians(phi_range)/W           #Phi = Horizontal angle (Azimuth): x-z plane
    dtheta = np.radians(theta_range)/H       #Theta = Vertical angle (Elevation): y-z plane

    x_lidar = points[:, 0]
    y_lidar = points[:, 1]
    z_lidar = points[:, 2]
    labels = points[:, 3]

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

    #Let intensity be inversely proportional to distance squared for visualisation
    range = max(z_lidar) - min(z_lidar)
    intensity = (1- (z_lidar/range)) * 255

    depth_map = np.zeros((H, W, C+1))#+255
    depth_map[theta, phi, 0] = x_lidar
    depth_map[theta, phi, 1] = y_lidar
    depth_map[theta, phi, 2] = z_lidar
    depth_map[theta, phi, 3] = 1
    depth_map[theta, phi, 4] = d
    depth_map[theta, phi, 5] = 1-labels  #Temporary fix to swap labels: True = Obstacle

    return depth_map
    

if __name__ == "__main__":
    import pickle
    import argparse
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('out_folder')
    parser.add_argument('--vis', dest='vis', default=False, action='store_true')
    args = parser.parse_args()
    
    name, ext = os.path.splitext(os.path.basename(args.filename))

    if ext == '.pickle':
        pickle_in = open(args.filename, "rb")
        np_pcd = pickle.load(pickle_in, encoding='latin1')
    else:
        np_pcd = np.load(args.filename)
    
    #Preprocessing specific to data source
    #Flip y value so it is positive
    np_single = np_pcd * np.tile([1,1,1, 1], (np_pcd.shape[0], 1))

    #Remove 'too far' as obstacle - set 'too far' as traversible. True = Traversible
    for i in range(np_single.shape[0]):
        if np_single[i][2] == 10: np_single[i][3] = 1
    
    depth_map = lidar_to_2d_front_view(np_single)

    np.save(os.path.join(args.out_folder, name) + '-projection.npy', depth_map)

    #Visualise
    if args.vis == True:
        plt.scatter(depth_map[:,:,5], cmap='gray')
        plt.show()
    
