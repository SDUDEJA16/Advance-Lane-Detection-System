class LaneDetection:
    
    def __init__(self, objectpoints, imagepoints,src_pts,dest_pts, window_no, 
                 margin, minpix, 
                 small_img_size=(256, 144), small_img_x_offset=20, small_img_y_offset=10, lane_width_px=800,img_dimensions=(720, 1280), 
                 lane_center_px_psp=600, real_world_lane_size_meters=(32, 3.7)):
        
        self.objectpoints = objectpoints
        self.imagepoints = imagepoints
        (self.matrix, self.inverse_matrix) = compute_matrix(src_pts,dest_pts)

        self.window_no = window_no
        self.half_width = margin
        self.minpix = minpix
        
        #self.small_img_size = small_img_size
        #self.small_img_x_offset = small_img_x_offset
        #self.small_img_y_offset = small_img_y_offset
        
        self.img_dimensions = img_dimensions
        self.lane_width_px = lane_width_px
        self.lane_center_px_psp = lane_center_px_psp 
        self.real_world_lane_size_meters = real_world_lane_size_meters

        # We can pre-compute some data here
        self.ym_per_px = self.real_world_lane_size_meters[0] / self.img_dimensions[0]
        self.xm_per_px = self.real_world_lane_size_meters[1] / self.lane_width_px
        self.ploty = np.linspace(0, self.img_dimensions[0] - 1, self.img_dimensions[0])
        
        self.previous_left_lane_line = None
        self.previous_right_lane_line = None
        
        self.previous_left_lane_lines = LaneLineHistory()
        self.previous_right_lane_lines = LaneLineHistory()
        
        self.total_img_count = 0
        
    
    def process_image(self, img):
        
        #undistort the image using image points and object points
        undistorted_img = undistort_image(img, self.objectpoints, self.imagepoints)
        
        # Produce threshold image with color and gradient thresholding
        threshold_img = combine_color_and_gradient_threshold(undistorted_img)
        
        # Create the perspective transform
        img_size = (undistorted_img.shape[1], undistorted_img.shape[0])
        undist_img_perspective = cv2.warpPerspective(undistorted_img, self.matrix, img_size, flags=cv2.INTER_LINEAR)
        thresh_img_perspective = cv2.warpPerspective(threshold_img, self.matrix, img_size, flags=cv2.INTER_LINEAR)
        
        #do sliding window approach and fit the polynomial on warped image
        leftline, rightline = self.lane_fit(thresh_img_perspective)
        
        #compute curvature of road 
        lcr, rcr, lco = self.calculate_curvature(leftline, rightline)
        
        
        final_img=self.draw(thresh_img_perspective,leftline,rightline,undistorted_img,undist_img_perspective,lcr,rcr,lco)
        
        #drawn_lines = self.draw_lane_lines(thresh_img_perspective, leftline, rightline)        
        #plt.imshow(drawn_lines)
        
        #drawn_lines_regions = self.draw_lane_lines_regions(thresh_img_perspective, leftline, rightline)
        #plt.imshow(drawn_lines_regions)
        
        #drawn_lane_area = self.draw_lane_area(thresh_img_perspective, undistorted_img, leftline, rightline)        
        #plt.imshow(drawn_lane_area)
        
        #drawn_hotspots = self.draw_lines_hotspots(thresh_img_perspective, leftline, rightline)
        #plt.imshow(drawn_hotspots)
        
        #combined_lane_img = self.combine_images(drawn_lane_area, drawn_lines, drawn_lines_regions, drawn_hotspots, undist_img_perspective)
        #final_img = self.draw_lane_curvature_text(combined_lane_img, lcr, rcr, lco)
        
        self.total_img_count += 1
        self.previous_left_lane_line = leftline
        self.previous_right_lane_line = rightline
        
        return final_img
    
    def draw(self,image,leftline,rightline,undistorted_img,undist_img_perspective,lcr,rcr,lco):
        drawn_lines=draw_lines(image, leftline, rightline)
        drawn_lines_regions = draw_lines_regions(image,leftline, rightline)
        drawn_lane_area = draw_area(image, undistorted_img, leftline, rightline,self.inverse_matrix)
        drawn_hotspots = draw_hotspots(image, leftline, rightline)
        combined=combine(drawn_lane_area, drawn_lines, drawn_lines_regions, drawn_hotspots, undist_img_perspective)
        final_img=draw_text(combined, lcr, rcr, lco)
        return final_img
        
    def draw_lane_curvature_text(self, img, left_curvature_meters, right_curvature_meters, center_offset_meters):
        """
        Returns an image with curvature information inscribed
        """
        
        offset_y = self.small_img_size[1] * 1 + self.small_img_y_offset * 5
        offset_x = self.small_img_x_offset
        
        template = "{0:17}{1:17}{2:17}"
        txt_header = template.format("Left Curvature", "Right Curvature", "Center Alignment") 
        print(txt_header)
        txt_values = template.format("{:.4f}m".format(left_curvature_meters), 
                                     "{:.4f}m".format(right_curvature_meters),
                                     "{:.4f}m Right".format(center_offset_meters))
        if center_offset_meters < 0.0:
            txt_values = template.format("{:.4f}m".format(left_curvature_meters), 
                                     "{:.4f}m".format(right_curvature_meters),
                                     "{:.4f}m Left".format(math.fabs(center_offset_meters)))
            
        
        print(txt_values)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, txt_header, (offset_x, offset_y), font, 1, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(img, txt_values, (offset_x, offset_y + self.small_img_y_offset * 5), font, 1, (255,255,255), 2, cv2.LINE_AA)
        
        return img
    
    def combine_images(self, lane_area_img, lines_img, lines_regions_img, lane_hotspots_img, psp_color_img):        
        """
        Returns a new image made up of the lane area image, and the remaining lane images are overlaid as
        small images in a row at the top of the the new image
        """
        small_lines = cv2.resize(lines_img, self.small_img_size)
        small_region = cv2.resize(lines_regions_img, self.small_img_size)
        small_hotspots = cv2.resize(lane_hotspots_img, self.small_img_size)
        small_color_psp = cv2.resize(psp_color_img, self.small_img_size)
                
        lane_area_img[self.small_img_y_offset: self.small_img_y_offset + self.small_img_size[1], self.small_img_x_offset: self.small_img_x_offset + self.small_img_size[0]] = small_lines
        
        start_offset_y = self.small_img_y_offset 
        start_offset_x = 2 * self.small_img_x_offset + self.small_img_size[0]
        lane_area_img[start_offset_y: start_offset_y + self.small_img_size[1], start_offset_x: start_offset_x + self.small_img_size[0]] = small_region
        
        start_offset_y = self.small_img_y_offset 
        start_offset_x = 3 * self.small_img_x_offset + 2 * self.small_img_size[0]
        lane_area_img[start_offset_y: start_offset_y + self.small_img_size[1], start_offset_x: start_offset_x + self.small_img_size[0]] = small_hotspots

        start_offset_y = self.small_img_y_offset 
        start_offset_x = 4 * self.small_img_x_offset + 3 * self.small_img_size[0]
        lane_area_img[start_offset_y: start_offset_y + self.small_img_size[1], start_offset_x: start_offset_x + self.small_img_size[0]] = small_color_psp
        
        
        return lane_area_img
    
        
    def draw_lane_area(self, warped_img, undist_img, left_line, right_line):
        """
        Returns an image where the inside of the lane has been colored in bright green
        """
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped_img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        ploty = np.linspace(0, warped_img.shape[0] - 1, warped_img.shape[0])
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_line.line_fit_x, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line.line_fit_x, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.inverse_matrix, (undist_img.shape[1], undist_img.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)
        
        return result
        
        
    def draw_lane_lines(self, warped_img, left_line, right_line):
        """
        Returns an image where the computed lane lines have been drawn on top of the original warped binary image
        """
        # Create an output image with 3 colors (RGB) from the binary warped image to draw on and  visualize the result
        out_img = np.dstack((warped_img, warped_img, warped_img))*255
        
        # Now draw the lines
        ploty = np.linspace(0, warped_img.shape[0] - 1, warped_img.shape[0])
        pts_left = np.dstack((left_line.line_fit_x, ploty)).astype(np.int32)
        pts_right = np.dstack((right_line.line_fit_x, ploty)).astype(np.int32)

        cv2.polylines(out_img, pts_left, False,  (255, 140,0), 5)
        cv2.polylines(out_img, pts_right, False, (255, 140,0), 5)
        
        for low_pt, high_pt in left_line.windows:
            cv2.rectangle(out_img, low_pt, high_pt, (0, 255, 0), 3)

        for low_pt, high_pt in right_line.windows:            
            cv2.rectangle(out_img, low_pt, high_pt, (0, 255, 0), 3)           
        
        return out_img    
    
    def draw_lane_lines_regions(self, warped_img, left_line, right_line):
        """
        Returns an image where the computed left and right lane areas have been drawn on top of the original warped binary image
        """
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        margin = self.half_width
        ploty = np.linspace(0, warped_img.shape[0] - 1, warped_img.shape[0])
        
        left_line_window1 = np.array([np.transpose(np.vstack([left_line.line_fit_x - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_line.line_fit_x + margin, 
                                      ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        
        right_line_window1 = np.array([np.transpose(np.vstack([right_line.line_fit_x - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_line.line_fit_x + margin, 
                                      ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Create RGB image from binary warped image
        region_img = np.dstack((warped_img, warped_img, warped_img)) * 255

        # Draw the lane onto the warped blank image
        cv2.fillPoly(region_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(region_img, np.int_([right_line_pts]), (0, 255, 0))
        
        return region_img


    def draw_lines_hotspots(self, warped_img, left_line, right_line):
        """
        Returns a RGB image where the portions of the lane lines that were
        identified by our pipeline are colored in yellow (left) and blue (right)
        """
        out_img = np.dstack((warped_img, warped_img, warped_img))*255
        
        out_img[left_line.non_zero_y, left_line.non_zero_x] = [255, 255, 0]
        out_img[right_line.non_zero_y, right_line.non_zero_x] = [0, 0, 255]
        
        return out_img

           
    def calculate_curvature(self, left_line, right_line):
        
        
        ploty = self.ploty
        y_eval = np.max(ploty)
        
        # Define conversions in x and y from pixels space to meters
        leftx = left_line.line_fit_x
        rightx = right_line.line_fit_x
        
        # Fit new polynomials: find x for y in real-world space
        left_fit_cr = np.polyfit(ploty * self.ym_per_px, leftx * self.xm_per_px, 2)
        right_fit_cr = np.polyfit(ploty * self.ym_per_px, rightx * self.xm_per_px, 2)
        
        # Now calculate the radii of the curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * self.ym_per_px + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 *right_fit_cr[0] * y_eval * self.ym_per_px + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])
        
        # Use our computed polynomial to determine the car's center position in image space, then
        left_fit = left_line.polynomial_coeff
        right_fit = right_line.polynomial_coeff
        
        center_offset_img_space = (((left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]) + 
                   (right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2])) / 2) - self.lane_center_px_psp
        center_offset_real_world_m = center_offset_img_space * self.xm_per_px
        
        # Now our radius of curvature is in meters        
        return left_curverad, right_curverad, center_offset_real_world_m     
        
        
        
    def lane_fit(self, img):
        

        #draw histogram
        hist=draw_hist(img)
        
        #find peak of histogram
        left,right=find_peak(hist)
        
        #parameters defining sliding window
        height = np.int(img.shape[0]//self.window_no)
        margin = self.half_width
        min_pix_to_recenter = self.minpix
        
        #Number of non zero pixels in an image
        nonzero = img.nonzero()
        
        #get it x and y coordinates in different arrays
        non_zero_y = np.array(nonzero[0])
        non_zero_x = np.array(nonzero[1])
        
        # Initial high peak position for a histogram 
        left_lane_start=left
        right_lane_start=right 
        
        # List for obtaining pixels inside a sliding window approach
        left_lane_inds = []
        right_lane_inds = []
        
        #storing result of this computation for future sliding window calculation
        left_line = LaneLine()
        right_line = LaneLine()
        
        #Total Non zero pixel 
        total_non_zeros = len(non_zero_x)
        non_zero_found_per = 0.0
           
        #Not the first frame in video    
        if self.previous_left_lane_line is not None and self.previous_right_lane_line is not None:
            
            A=self.previous_left_lane_line.polynomial_coeff[0]
            B=self.previous_left_lane_line.polynomial_coeff[1]
            C=self.previous_left_lane_line.polynomial_coeff[2]
            left_fit_previous=A*(non_zero_y**2)+B*non_zero_y+C
            left_lane_inds = ((non_zero_x > (left_fit_previous - margin))  & (non_zero_x < (left_fit_previous + margin))) 
            A1=self.previous_right_lane_line.polynomial_coeff[0]
            B1=self.previous_right_lane_line.polynomial_coeff[1]
            C1=self.previous_right_lane_line.polynomial_coeff[2]
            right_fit_previous=A1*(non_zero_y**2)+B1*non_zero_y+C1
            right_lane_inds = ((non_zero_x > (right_fit_previous - margin))  & (non_zero_x < (right_fit_previous + margin)))

            
            found_left = np.sum(left_lane_inds)
            found_right = np.sum(right_lane_inds)
            non_zero_found_per = (found_left + found_right) / total_non_zeros
        
        #if not sufficient pixels are not found under sliding window discard this and start from starting
        if non_zero_found_per < 0.85:
            left_lane_inds = []
            right_lane_inds = []

            
            for w in range(self.window_no):
                
                #create boundaries dimension of sliding window
                win_y_bottom=img.shape[0]-w*height
                win_y_top=img.shape[0]-(w+1)*height
                
                #x coordinates
                win_left_low_x=left_lane_start-margin
                win_left_high_x=left_lane_start+margin
                win_right_low_x=right_lane_start-margin
                win_right_high_x=right_lane_start+margin
                
                '''this'''
                left_line.windows.append([(win_left_low_x,win_y_top),(win_left_high_x,win_y_bottom)])
                right_line.windows.append([(win_right_low_x,win_y_top),(win_right_high_x,win_y_bottom)])
                
                
                #index of points in array coordinates non_zero_x and non_zero_y that comes under the sliding window
                sliding_left = ((non_zero_y >= win_y_top) & (non_zero_y <win_y_bottom ) & 
                (non_zero_x >= win_left_low_x) &  (non_zero_x <win_left_high_x)).nonzero()[0]  
                sliding_right = ((non_zero_y >=  win_y_top) & (non_zero_y <win_y_bottom) & 
                (non_zero_x >= win_right_low_x) &  (non_zero_x < win_right_high_x)).nonzero()[0]
                
                # Append these indices to the lists
                left_lane_inds.append(sliding_left)
                right_lane_inds.append(sliding_right)

                # If you found > minpix pixels, recenter next window on their mean position
                if len(sliding_left) > min_pix_to_recenter:
                    left_lane_start = np.int(np.mean(non_zero_x[sliding_left]))
                if len(sliding_right) > min_pix_to_recenter:        
                    right_lane_start = np.int(np.mean(non_zero_x[sliding_right]))

            #concatenate the list to create one list
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
            
        # Extract left and right line pixel positions
        x_left = non_zero_x[left_lane_inds]
        y_left = non_zero_y[left_lane_inds] 
        x_right = non_zero_x[right_lane_inds]
        y_right = non_zero_y[right_lane_inds] 
        
        left_fit = np.polyfit(y_left, x_left, 2)
        right_fit = np.polyfit(y_right, x_right, 2)
        left_line.polynomial_coeff = left_fit
        right_line.polynomial_coeff = right_fit
        
        '''this'''
        if not self.previous_left_lane_lines.append(left_line):
            left_fit = self.previous_left_lane_lines.get_smoothed_polynomial()
            left_line.polynomial_coeff = left_fit
            self.previous_left_lane_lines.append(left_line, force=True)
            print("**** REVISED Poly left {0}".format(left_fit))            
        


        if not self.previous_right_lane_lines.append(right_line):
            right_fit = self.previous_right_lane_lines.get_smoothed_polynomial()
            right_line.polynomial_coeff = right_fit
            self.previous_right_lane_lines.append(right_line, force=True)
            print("**** REVISED Poly right {0}".format(right_fit))

    
        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0] )
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        
        
        left_line.polynomial_coeff = left_fit
        left_line.line_fit_x = left_fitx
        left_line.non_zero_x = x_left  
        left_line.non_zero_y = y_left

        right_line.polynomial_coeff = right_fit
        right_line.line_fit_x = right_fitx
        right_line.non_zero_x = x_right
        right_line.non_zero_y = y_right

        
        return (left_line, right_line)