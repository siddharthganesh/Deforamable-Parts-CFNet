import tensorflow as tf
print('Using Tensorflow '+tf.__version__)
import matplotlib.pyplot as plt
import sys
# sys.path.append('../')
import os
import csv
import numpy as np
from PIL import Image
import time
from scipy import *
import src.siamese as siam
from src.visualization import show_frame, show_crops, show_scores
import skimage.measure
from skimage.transform import resize
from scipy import ndimage
import math
#from distance_transform import dt2d
# gpu_device = 2
# os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_device)

# read default parameters and override with custom ones

def tracker(hp, run, design, frame_name_list, pos_x, pos_y, target_w, target_h, final_score_sz, filename, image, templates_z, scores, start_frame):
    
    num_frames = np.size(frame_name_list)
    # stores tracker's output for evaluation
    bboxes = np.zeros((num_frames,4))
    bboxesupper = np.zeros((num_frames,4))
    bboxeslower = np.zeros((num_frames,4))

    scale_factors = hp.scale_step**np.linspace(-np.ceil(hp.scale_num/2), np.ceil(hp.scale_num/2), hp.scale_num)
    # cosine window to penalize large displacements    
    hann_1d = np.expand_dims(np.hanning(final_score_sz), axis=0)
    penalty = np.transpose(hann_1d) * hann_1d
    penalty = penalty / np.sum(penalty)

    context = design.context*(target_w+target_h)
    z_sz = np.sqrt(np.prod((target_w+context)*(target_h+context)))
    x_sz = float(design.search_sz) / design.exemplar_sz * z_sz

    # thresholds to saturate patches shrinking/growing
    min_z = hp.scale_min * z_sz
    max_z = hp.scale_max * z_sz
    min_x = hp.scale_min * x_sz
    max_x = hp.scale_max * x_sz

    # run_metadata = tf.RunMetadata()
    # run_opts = {
    #     'options': tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
    #     'run_metadata': run_metadata,
    # }

    run_opts = {}

    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        # save first frame position (from ground-truth)
        bboxes[0,:] = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h                
        bboxesupper[0,:] = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h/2
        bboxeslower[0,:]= pos_x-target_w/2, pos_y, target_w, target_h/2
        
        image_, templates_z_ = sess.run([image, templates_z], feed_dict={
                                                                        siam.pos_x_ph: pos_x,
                                                                        siam.pos_y_ph: pos_y,
                                                                        siam.z_sz_ph: z_sz,
                                                                        filename: frame_name_list[0]})
        image_, templates_z_upper = sess.run([image, templates_z], feed_dict={
                                                                        siam.pos_x_ph: pos_x,
                                                                        siam.pos_y_ph: pos_y-target_h/2,
                                                                        siam.z_sz_ph: z_sz,
                                                                        filename: frame_name_list[0]})
        image_, templates_z_lower = sess.run([image, templates_z], feed_dict={
                                                                        siam.pos_x_ph: pos_x,
                                                                        siam.pos_y_ph: pos_y+target_h/2,
                                                                        siam.z_sz_ph: z_sz,
                                                                        filename: frame_name_list[0]})
        new_templates_z_ = templates_z_    
        new_templates_z_upper = templates_z_upper    
        new_templates_z_lower = templates_z_lower
        
        t_start = time.time()
        sco_final = np.zeros((3,257,257))
        # Get an image from the queue
        for i in range(1, num_frames):        
            for j in range(1,4):
                scaled_exemplar = z_sz * scale_factors
                scaled_search_area = x_sz * scale_factors
                scaled_target_w = target_w * scale_factors
                scaled_target_h = target_h * scale_factors
                image_, scores_ = sess.run(
                    [image, scores],
                    feed_dict={
                        siam.pos_x_ph: pos_x,
                        #siam.pos_y_ph: pos_y,
                        siam.pos_y_ph: pos_y-target_h/2 if j == 1 else (pos_y+target_h/2 if j == 2 else pos_y),
                        siam.x_sz0_ph: scaled_search_area[0],
                        siam.x_sz1_ph: scaled_search_area[1],
                        siam.x_sz2_ph: scaled_search_area[2],
                        templates_z: np.squeeze(templates_z_upper) if j==1 else (np.squeeze(templates_z_lower) if j==2 else np.squeeze(templates_z_)),
                        filename: frame_name_list[i],
                    }, **run_opts)
                
                if j==1:
                    templates_zupper=np.squeeze(templates_z_upper)
                    templates_zupper=tf.convert_to_tensor(templates_zupper,np.float32)
                elif j==2:
                    templates_zlower=np.squeeze(templates_z_lower)
                    templates_zlower=tf.convert_to_tensor(templates_zlower,np.float32)
                else:
                    templates_zmain=np.squeeze(templates_z_)
                    templates_zmain=tf.convert_to_tensor(templates_zmain,np.float32)
                
                scores_ = np.squeeze(scores_)
                # penalize change of scale
                scores_[0,:,:] = hp.scale_penalty*scores_[0,:,:]
                scores_[2,:,:] = hp.scale_penalty*scores_[2,:,:]
                # find scale with highest peak (after penalty)
                new_scale_id = np.argmax(np.amax(scores_, axis=(1,2)))
                # update scaled sizes
                x_sz = (1-hp.scale_lr)*x_sz + hp.scale_lr*scaled_search_area[new_scale_id]        
                target_w = (1-hp.scale_lr)*target_w + hp.scale_lr*scaled_target_w[new_scale_id]
                target_h = (1-hp.scale_lr)*target_h + hp.scale_lr*scaled_target_h[new_scale_id]
                # select response with new_scale_id
                score_ = scores_[new_scale_id,:,:]
                score_ = score_ - np.min(score_)
                score_ = score_/np.sum(score_)
                # apply displacement penalty
                #score_ = (1-hp.window_influence)*score_ + hp.window_influence*penalty
                min1=score_.min()
                max1=score_.max()
                #score_max = skimage.measure.block_reduce(score_, (5,5), np.max)
                
                #score_max=ndimage.distance_transform_edt(score_)
                w=[0.1,0,0.1,0]
                score_ = (score_/score_.max())*255
                score_max=dt2d(score_,w,4)
                score_max = (((score_max - min1) * (score_max.max() - score_max.min())) / (max1 - min1)) + score_max.min()
                
                
                #score_max_norm = Image.fromarray(score_max)
                
                
                new_width = 257
                new_height = 257
                #sco = score_max_norm.resize((new_width,new_height),Image.ANTIALIAS)
                #sco = resize(int(score_max), (257, 257))
                sco = score_max    
                sco_final[j-1,:,:] = sco
            
            #####################################################################################
            sco_f = sco_final[0,:,:] + sco_final[1,:,:] + sco_final[2,:,:]
            #sco_f = sco_final[0,:,:]
            pos_x, pos_y = _update_target_position(pos_x, pos_y, sco_f, final_score_sz, design.tot_stride, design.search_sz, hp.response_up, x_sz)
            pos_x_upper, pos_y_upper = _update_target_position(pos_x, pos_y, sco_final[0,:,:], final_score_sz, design.tot_stride, design.search_sz, hp.response_up, x_sz)
            pos_x_lower, pos_y_lower = _update_target_position(pos_x, pos_y, sco_final[1,:,:], final_score_sz, design.tot_stride, design.search_sz, hp.response_up, x_sz)
            
            # convert <cx,cy,w,h> to <x,y,w,h> and save output
            bboxes[i,:] = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h
            bboxesupper[i,:] = pos_x_upper-target_w/2, pos_y_upper-target_h/2, target_w, target_h/2
            bboxeslower[i,:] = pos_x_lower-target_w/2, pos_y_lower, target_w, target_h/2
            
            
            # update the target representation with a rolling average
            if hp.z_lr>0:
                new_templates_z_ = sess.run([templates_zmain], feed_dict={
                                                                siam.pos_x_ph: pos_x,
                                                                siam.pos_y_ph: pos_y,
                                                                siam.z_sz_ph: z_sz,
                                                                image: image_
                                                                })
                new_templates_z_upper = sess.run([templates_zupper], feed_dict={
                                                                siam.pos_x_ph: pos_x_upper,
                                                                siam.pos_y_ph: pos_y_upper,
                                                                siam.z_sz_ph: z_sz,
                                                                image: image_
                                                                })
                new_templates_z_lower = sess.run([templates_zlower], feed_dict={
                                                                siam.pos_x_ph: pos_x_lower,
                                                                siam.pos_y_ph: pos_y_lower,
                                                                siam.z_sz_ph: z_sz,
                                                                image: image_
                                                                })

                templates_z_=(1-hp.z_lr)*np.asarray(templates_z_) + hp.z_lr*np.asarray(new_templates_z_)
                templates_z_upper=(1-hp.z_lr)*np.asarray(templates_z_upper) + hp.z_lr*np.asarray(new_templates_z_upper)
                templates_z_lower=(1-hp.z_lr)*np.asarray(templates_z_lower) + hp.z_lr*np.asarray(new_templates_z_lower)
            
            # update template patch size
            z_sz = (1-hp.scale_lr)*z_sz + hp.scale_lr*scaled_exemplar[new_scale_id]
            
            if run.visualization:
                show_frame(image_, bboxes[i,:],bboxesupper[i,:],bboxeslower[i,:], 1)        

        t_elapsed = time.time() - t_start
        speed = num_frames/t_elapsed

        # Finish off the filename queue coordinator.
        coord.request_stop()
        coord.join(threads) 

        # from tensorflow.python.client import timeline
        # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        # trace_file = open('timeline-search.ctf.json', 'w')
        # trace_file.write(trace.generate_chrome_trace_format())

    plt.close('all')

    return bboxes, speed


def _update_target_position(pos_x, pos_y, score, final_score_sz, tot_stride, search_sz, response_up, x_sz):
    # find location of score maximizer
    p = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
    # displacement from the center in search area final representation ...
    center = float(final_score_sz - 1) / 2
    disp_in_area = p - center
    # displacement from the center in instance crop
    disp_in_xcrop = disp_in_area * float(tot_stride) / response_up
    # displacement from the center in instance crop (in frame coordinates)
    disp_in_frame = disp_in_xcrop *  x_sz / search_sz
    # *position* within frame in frame coordinates
    pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
    return pos_x, pos_y

@profile
def dt1d(vals,out_vals,I,step,shift,n,a,b):
    for i in range(0,n):
        max_val=-float('inf')
        argmax=0
        first=max(0,i-shift)
        last=min(n-1,i+shift)
        for j in range(first,last+1):
            #val = vals[j*step] - a*(i-j)*(i-j) - b*(i-j)
            val = vals[j*step] - a*((i-j)**2)
            if val>max_val:
                max_val=val
                argmax=j
        out_vals[i*step]=max_val
        I[i*step]=argmax
    return I,out_vals

@profile
def dt2d(scoreMap,w,shift):
    ax = w[0]
    bx = w[1]
    ay = w[2]
    by = w[3]
    [n1,n2]=np.shape(scoreMap)
    tmpOut = np.zeros((n1, n2))
    tmpIy = np.zeros((n1, n2))
    Ix = np.zeros((n1, n2))
    Iy = np.zeros((n1, n2))
    for x in range(0,n2):
        tmpIy[:,x],tmpOut[:,x] = dt1d(scoreMap[:,x],tmpOut[:,x],tmpIy[:,x],1, shift, n1, ay, by)
    for y in range(0,n1):
        Ix[y,:],scoreMap[y,:] = dt1d(tmpOut[y,:],scoreMap[y,:],Ix[y,:],1, shift, n2, ax, bx)
    
    
    for x in range(0,n2): 
        for y in range(0,n1):
            t = Ix[y,x]
            Iy[y, x] = tmpIy[y, int(Ix[y,x])]
    return scoreMap