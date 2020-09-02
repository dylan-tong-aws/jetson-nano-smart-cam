import sys
import numpy as np
from collections import deque

class CVObjectTracker:

    EUCLIDIAN_LAST_FRAME = 0
    EUCLIDIAN_LAST_N_FRAMES = 1
    SIAMESE_NETWORK = 2

    def __init__(self, scale_factor, min_dist, threshold=0.7, max_objects=10, track_n= 10, algo=EUCLIDIAN_LAST_FRAME) :
        
        self.__prev_trackings = deque()
        self.__id_counter = 0
        self.__selection = self.__id_counter
        self.__max_objects = max_objects
        self.__algo = algo
        
        self.track_n = track_n
        self.scale_factor = scale_factor
        self.min_dist = min_dist
        self.threshold = threshold

        if self.__algo == CVObjectTracker.EUCLIDIAN_LAST_FRAME :
            self.track_n = 1
#            self.get_ids = CVObjectTracker.__euclidian_get_ids
#        elif self.__algo == CVObjectTracker.SIAMESE_NETWORK :
#            sys.exit("Siamese network based object tracking hasn't been implemented yet.")
#        elif self.__algo == CVObjectTracker.EUCLIDIAN_LAST_N_FRAMES :
#           self.get_ids = CVObjectTracker.__euclidian_get_ids
#        else :
#            sys.exit("Unknown algorithm specified.")

    def __euclidian_id_score(self, frame_id, dist, prev_score, prev_score_ratio=0.3) :

        score = ((frame_id/self.track_n)**1.5) * ((self.min_dist - dist)/self.min_dist)
        return ((1.0-prev_score_ratio)*score) + (prev_score_ratio*prev_score)

    def __create_new_id(self, det, active_ids, cur_obj_map) :
        active_ids[self.__id_counter] = (0, det, 1.0)
        cur_obj_map[(det[2],det[3],det[4],det[5])] = (self.__id_counter,1.0)
        self.__id_counter += 1
   
    def get_next_selection(self, active_ids) :

        if len(active_ids) > 1 :
            next_id = self.__selection + 1
            if next_id >= self.__id_counter :
                next_id = self.__id_counter
                for i in active_ids :
                    if i < next_id :
                        next_id = i
            else :
                next_id = self.__id_counter
                for i in active_ids :                    
                    if i > self.__selection and i <= next_id :
                        next_id = i
            self.__selection = next_id
            return next_id
        elif len(active_ids) == 1 :
            self.__selection = list(active_ids)[0]
            return self.__selection 
        else :
            return None 

    def __update_tracker(self, detections, cur_obj_map) :

        tracking = {"d": detections, "m": cur_obj_map}
        self.__prev_trackings.appendleft(tracking)

        if len(self.__prev_trackings) > self.track_n :
            self.__prev_trackings.pop()

    @staticmethod
    def __find_nearest_detection(det, prev_dets) :

        dist0 = np.linalg.norm(prev_dets[:,2:4]-det[2:4], axis=1)
        dist1 = np.linalg.norm(prev_dets[:,4:6]-det[4:6], axis=1)

        # if we assume the QR codes use a standard size, we can use the bounding boxes to estimate depth
        # and formulate a more accurate estimate of distance of detections between frames. This is practical
        # for a solution that requires speed, few computations while aiming to maximize accuracy.
        diag = ((det[4] - det[2])**2 + (det[5] - det[3])**2)**0.5
        z_est = 3*np.absolute((np.linalg.norm(prev_dets[:,4:6] - prev_dets[:,2:4], axis=1)-diag))

        #print("diag {}".format(diag))
        #print("z: {}".format(z_est))
        #print("d0: {}".format(dist0))
        #print("d1: {}".format(dist1))

        dist = (dist0 + dist1 + z_est)/3
        min_idx = np.argmin(dist)
        min_dist = dist[min_idx]

        return min_dist, min_idx   


    def __euclidean_compare_n_frames(self, det, active_ids, cur_obj_map, first_frame=0, dist_per_frame=0) :
        
        track_count = len(self.__prev_trackings)
        conflicts = None
        obj_found = False

        for i in range(first_frame,track_count):

            prev_map = self.__prev_trackings[i]["m"]
            prev_dets = self.__prev_trackings[i]["d"]
            n_detections = prev_dets.shape[0]
            
            adj_min_dist = (self.min_dist + (dist_per_frame*i))
            min_dist = adj_min_dist

            if n_detections > 0 :
                min_dist, min_idx = CVObjectTracker.__find_nearest_detection(det, prev_dets)

            if min_dist < adj_min_dist :
                #try :    
                oid, prev_confidence = prev_map[(prev_dets[min_idx,2],prev_dets[min_idx,3],prev_dets[min_idx,4],prev_dets[min_idx,5])]
                #except KeyError as e:
                #    print("Attempted key: {}".format((prev_dets[min_idx,2],prev_dets[min_idx,3],prev_dets[min_idx,4],prev_dets[min_idx,5])))
                #    print(prev_map.items())
                
                assign_map = active_ids.get(oid)
                score = self.__euclidian_id_score(self.track_n-i, min_dist, prev_confidence)
                    
                if assign_map :

                    if assign_map[2] < score :
                        
                        conflict_det = assign_map[1]
                        conflict = (assign_map[0], conflict_det)
                        
                        active_ids[oid] = (i, det, score)
                        cur_obj_map[(det[2],det[3],det[4],det[5])] = (oid,score)
                        #del cur_obj_map[(conflict_det[2],conflict_det[3],conflict_det[4],conflict_det[5])]
                        #print("CONFLICT FOUND! Attempted del {}".format((conflict_det[2],conflict_det[3],conflict_det[4],conflict_det[5])))
                        #print(cur_obj_map.items())
                        obj_found = True
                        break

                else :
                    active_ids[oid] = (i, det, score)
                    cur_obj_map[(det[2],det[3],det[4],det[5])] = (oid,score)
                    obj_found = True
                    break
                    
        return (obj_found, conflicts)               

    def euclidian_get_ids(self, od_results, target_cls_id=1) :
        
        ## only consider the top detections for qr codes and are over a threshold
        #od_results = od_results[(od_results[:,1] > self.threshold) & (od_results[:,0] == target_cls_id)][:self.__max_objects,:]
        od_results = od_results[(od_results[:,1] > self.threshold)][:self.__max_objects,:]
        od_results[:,2:] *= self.scale_factor
        detections = od_results.tolist()

        cur_obj_map = {}
        active_ids = {}

        for det in detections:

            obj_found, conflicts = self.__euclidean_compare_n_frames(det, active_ids, cur_obj_map)

            if not obj_found :
                self.__create_new_id(det, active_ids, cur_obj_map)
            else :
                while conflicts :
                    restart = conflicts[0] + 1
                    if restart < self.track_n :
                        obj_found, conflicts = self.__euclidean_compare_n_frames(conflicts[1], active_ids, cur_obj_map, first_frame=restart)
                        if not obj_found :
                            self.__create_new_id(det, active_ids, cur_obj_map)
                            break
                    else :
                        self.__create_new_id(det, active_ids, cur_obj_map)
                        break

        if not active_ids.get(self.__selection) :
            self.get_next_selection(active_ids)

        self.__update_tracker(od_results,cur_obj_map)

        return cur_obj_map, self.__selection, active_ids