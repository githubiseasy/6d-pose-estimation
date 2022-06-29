import os
import time
import torch
import argparse
import scipy.io
import warnings
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np

import datetime
import json
import dataset
from darknet import Darknet
from utils import *
from MeshPly import MeshPly


def valid(datacfg, modelcfg, weightfile, logging_command):
    def truths_length(truths, max_num_gt=50):
        for i in range(max_num_gt):
            if truths[i][1] == 0:
                return i
    
    # set result path
    result_path = '../experimental_results'
    c_id = datacfg.split('/')[1]
    #c_id = eng2id(datacfg.split('/')[1])
    c_kor = id2kor(c_id)
    c_eng = id2eng(c_id)
    # Parse configuration files
    datacfg = datacfg.replace(c_id, c_eng)
    data_options = read_data_cfg(datacfg)
    valid_images = data_options['valid']
    meshname = data_options['mesh']
    ###
    weightfile = weightfile.replace(c_id, c_eng)
    #meshname = '../test_datasets' + '/' + c_id + '.' + c_kor + '/'+ c_id + '.'+ c_kor + '.원천데이터/' + c_id + '.' + c_kor + '.3D_Shape/' + meshname.split('/')[-1]
    meshname = 'new_dataset' + '/' + c_id + '.' + c_kor + '/'+ c_id + '.'+ c_kor + '.원천데이터/' + c_id + '.' + c_kor + '.3D_Shape/' + meshname.split('/')[-1]
    backupdir = data_options['backup']
    name = data_options['name']
    gpus = data_options['gpus']
    im_width = int(data_options['width'])
    im_height = int(data_options['height'])
    if not os.path.exists(backupdir):
        makedirs(backupdir)

    # Parameters
    seed = int(time.time())
    #os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)
    proj            = True
    num_classes = 1
    testing_samples = 0.0

    # To save
    testing_error_pixel = 0.0
    iou_acc = []
    iou_convex_acc = []
    errs_2d = []
    errs_3d = []

    # Read object model information, get 3D bounding box corners
    mesh = MeshPly(meshname)
    vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    indices = np.c_[np.array(mesh.indices), np.ones((len(mesh.indices), 2))].transpose()
    corners3D = get_3D_corners(vertices)
    try:
        diam = float(options['diam'])
    except:
        diam = calc_pts_diameter(np.array(mesh.vertices))

    # Get validation file names
    with open(valid_images) as fp:
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]

    # Specicy model, load pretrained weights, pass to GPU and set the module in evaluation mode
    model = Darknet(modelcfg)
    model.print_network()
    model.load_weights(weightfile)
    model.cuda()
    model.eval()
    test_width = model.test_width
    test_height = model.test_height
    num_keypoints = model.num_keypoints
    num_labels = num_keypoints * 2 + 3  # +2 for width, height,  +1 for class label

    # Get the parser for the test dataset
    valid_dataset = dataset.listDataset(valid_images,
                                        shape=(test_width, test_height),
                                        shuffle=False,
                                        transform=transforms.Compose([transforms.ToTensor(), ]))

    # Specify the number of workers for multiple processing, get the dataloader for the test dataset
    kwargs = {'num_workers': 4, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, **kwargs)
    logging("   Testing {}...".format(name))
    logging("   Number of test samples: %d" % len(test_loader.dataset))
    # Iterate through test batches (Batch size for test data is 1)
    count = 0
    # save experiment results as CSV format
    class_name = test_loader.dataset.lines[0].split('/')[-1].split('_')[0]
    makedirs(result_path)
    c = open(os.path.join(result_path, class_name+'.csv'), 'w', encoding="UTF-8")
    c.write('{}\n'.format(logging_command))
    c.write('Running Start, {} \n'.format(datetime.datetime.now()))
    c.write('Data ID, x0-GT, y0-GT, x1-GT, y1-GT, x2-GT, y2-GT, x3-GT, y3-GT, x4-GT, y4-GT, x5-GT, y5-GT, x6-GT, y6-GT, x7-GT, y7-GT, x8-GT, y8-GT, x0-predict, y0-predict, x1-predict, y1-predict, x2-predict, y2-predict, x3-predict, y3-predict, x4-predict, y4-predict, x5-predict, y5-predict, x6-predict, y6-predict, x7-predict, y7-predict, x8-predict, y8-predict, pixel error, 2D projection, IoU, IoU score, \n')
    for batch_idx, (data, target, camera_info) in enumerate(test_loader):
        data_id = test_loader.dataset.lines[batch_idx].split('/')[-1][:-8]
        # Read intrinsic camera parameters
        fx = float(camera_info['fx'])
        fy = float(camera_info['fy'])
        u0 = float(camera_info['u0'])
        v0 = float(camera_info['v0'])
        internal_calibration = get_camera_intrinsic(fx, fy, u0, v0)

        # Pass data to GPU
        data = data.cuda()
        target = target.cuda()
        # Wrap tensors in Variable class, set volatile=True for inference mode and to use minimal memory during inference
        with torch.no_grad():
            data = data

        # Forward pass
        output = model(data).data
        # eliminate low-confidence predictions
        all_boxes = get_region_boxes(output, num_classes, num_keypoints)
        # Evaluation
        # Iterate through all batch elements
        for box_pr, target in zip([all_boxes], [target[0]]):
            # For each image, get all the targets (for multiple object pose estimation, there might be more than 1 target per image)
            truths = target.view(-1, num_labels)
            # Get how many objects are present in the scene
            num_gts = truths_length(truths)
            # Iterate through each ground-truth object
            for k in range(num_gts):
                box_gt = list()
                for j in range(1, 2 * num_keypoints + 1):
                    box_gt.append(truths[k][j])
                box_gt.extend([1.0, 1.0])
                box_gt.append(truths[k][0])

                # Denormalize the corner predictions
                corners2D_gt = np.array(np.reshape(box_gt[:18], [9, 2]), dtype='float32')
                corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')
                corners2D_gt[:, 0] = corners2D_gt[:, 0] * im_width
                corners2D_gt[:, 1] = corners2D_gt[:, 1] * im_height
                corners2D_pr[:, 0] = corners2D_pr[:, 0] * im_width
                corners2D_pr[:, 1] = corners2D_pr[:, 1] * im_height

                # write gt, predict
                gt = ''
                for i in range(9):
                    gt += '{:.1f},'.format(corners2D_gt[i][0])
                    gt += '{:.1f},'.format(corners2D_gt[i][1])
                predict = ''
                for i in range(9):
                    predict += '{:.1f},'.format(corners2D_pr[i][0])
                    predict += '{:.1f},'.format(corners2D_pr[i][1])

                # Compute [R|t] by pnp
                R_gt, t_gt = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)),
                                          dtype='float32'), corners2D_gt,
                                 np.array(internal_calibration, dtype='float32'))
                R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)),
                                          dtype='float32'), corners2D_pr,
                                 np.array(internal_calibration, dtype='float32'))

                # Compute pixel error
                Rt_gt = np.concatenate((R_gt, t_gt), axis=1)
                Rt_pr = np.concatenate((R_pr, t_pr), axis=1)
                proj_2d_gt = compute_projection(vertices, Rt_gt, internal_calibration)
                proj_2d_pred = compute_projection(vertices, Rt_pr, internal_calibration)
                proj_corners_gt = np.transpose(compute_projection(corners3D, Rt_gt, internal_calibration))
                proj_corners_pr = np.transpose(compute_projection(corners3D, Rt_pr, internal_calibration))

                norm = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=0)
                pixel_dist = np.mean(norm)
                errs_2d.append(pixel_dist)

                # Compute 3D distances
                transform_3d_gt = compute_transformation(vertices, Rt_gt)
                transform_3d_pred = compute_transformation(vertices, Rt_pr)
                norm3d = np.linalg.norm(transform_3d_gt - transform_3d_pred, axis=0)
                vertex_dist = np.mean(norm3d)
                errs_3d.append(vertex_dist)

                # Compute IOU
                frame = valid_files[count][-11:-7]
                if proj==True:
                    makedirs(backupdir + '/vis')
                    f_prj = open(backupdir + "/vis/prj_{}.txt".format(frame), 'w')
                    for i in  range(proj_2d_pred.shape[1]):
                        p_x = int(proj_2d_pred[0][i])
                        p_y = int(proj_2d_pred[1][i])
                        f_prj.write("{} {}\n".format(p_x, p_y))
                    f_prj.close()

                    f_ind = open(backupdir + "/vis/ind_{}.txt".format(frame), 'w')
                    for j in range(indices.shape[1]):
                        f_ind.write("{} {} {}\n".format(int(indices[0,j]), int(indices[1,j]), int(indices[2,j])))
                    f_ind.close()
                iou_convex = compute_convexhull_iou(corners2D_gt, corners2D_pr)
                iou_convex_acc.append(iou_convex)

                iou = compute_iou(name, frame, True)
                iou_acc.append(iou)
                # Sum errors
                testing_error_pixel += pixel_dist
                testing_samples += 1
                
                # csv write
                context = data_id + ',' + gt + predict + '{:.2f}, {}, {:.2f}, {}\n'.format(pixel_dist, pixel_dist <= 20, iou_convex, iou_convex>=0.5)
                c.write(context)
                
                count = count + 1
    c.write('Running END, {}'.format(datetime.datetime.now()))
    c.close()
    # Compute 2D projection error, 6D pose error, 5cm5degree error
    px_threshold = 20  # 5 pixel threshold for 2D reprojection error is standard in recent sota 6D object pose estimation works
    eps = 1e-5
    iou_test_c   = len(np.where(np.array(iou_convex_acc) >= 0.5)[0]) * 100 / (len(iou_convex_acc) + eps)
    iou_test25   = len(np.where(np.array(iou_acc) >= 0.25)[0]) * 100 / (len(iou_acc) + eps)
    iou_test     = len(np.where(np.array(iou_acc) >= 0.5)[0]) * 100 / (len(iou_acc) + eps)
    iou_test75   = len(np.where(np.array(iou_acc) >= 0.75)[0]) * 100 / (len(iou_acc) + eps)
    proj_test05  = len(np.where(np.array(errs_2d) <= 5)[0]) * 100. / (len(errs_2d)+eps)
    proj_test    = len(np.where(np.array(errs_2d) <= px_threshold)[0]) * 100. / (len(errs_2d)+eps)
    proj_test15  = len(np.where(np.array(errs_2d) <= 15)[0]) * 100. / (len(errs_2d)+eps)
    proj_test20  = len(np.where(np.array(errs_2d) <= 20)[0]) * 100. / (len(errs_2d)+eps)
    nts = float(testing_samples)

    # Print test statistics
    logging('Results of {} ({})'.format(name, datetime.datetime.now()))
    logging('   Mean 2D Err. (Pixel Dist.) = {:.2f} pix.'.format(testing_error_pixel/nts))
    logging('   Mean 3D Err. (Vertex Dist.) = {:.2f} mm'.format(np.mean(errs_3d)))
    logging('   Acc. using  5 px. 2D Projection = {:.2f}%'.format(proj_test05))
    logging('   Acc. using {} px. 2D Projection = {:.2f}%'.format(px_threshold, proj_test))
    logging('   Acc. using 15 px. 2D Projection = {:.2f}%'.format(proj_test15))
    logging('   Acc. using 20 px. 2D Projection = {:.2f}%'.format(proj_test20))
    logging('   Acc. using Intersection Of Union (IoU, convex) = {:.2f}%'.format(iou_test_c))

    logging('   Acc. using Intersection Of Union (IoU > 0.25) = {:.2f}%'.format(iou_test25))
    logging('   Acc. using Intersection Of Union (IoU > 0.50) = {:.2f}%'.format(iou_test))
    logging('   Acc. using Intersection Of Union (IoU > 0.75) = {:.2f}%'.format(iou_test75))
    fid = open("{}/{}.txt".format(result_path, c_id), "w")
    fid.write("{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format( testing_error_pixel/nts, np.mean(errs_3d), proj_test05, proj_test, proj_test15, proj_test20, iou_test_c, iou_test25, iou_test, iou_test75))
    fid.close()
if __name__ == '__main__':
    # Parse configuration files
    parser = argparse.ArgumentParser(description='SingleShotPose')
    parser.add_argument('--datacfg', type=str, default='data/beaker1/beaker1.data') # data config
    parser.add_argument('--modelcfg', type=str, default='data/beaker1/model/yolo-pose.cfg') # network config
    parser.add_argument('--weightfile', type=str, default='data/beaker1/model.weights') # imagenet initialized weights
    parser.add_argument('--command', type=str, default='python valid.py')
    args       = parser.parse_args()
    datacfg    = args.datacfg    
    modelcfg   = args.modelcfg
    weightfile = args.weightfile
    logging_command = args.command
    valid(datacfg, modelcfg, weightfile, logging_command)
