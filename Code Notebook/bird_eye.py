import cv2
def compute_matrix(src_pts,dest_pts):
    matrix=cv2.getPerspectiveTransform(src_pts,dest_pts)
    inverse_matrix=cv2.getPerspectiveTransform(dest_pts,src_pts)
    return matrix,inverse_matrix

def perspective_transform(img,src_pts,dest_pts):
    matrix,inverse_matrix=compute_matrix(src_pts,dest_pts)
    warped_image = cv2.warpPerspective(img, matrix,(img.shape[1],img.shape[0]))
    return warped_image