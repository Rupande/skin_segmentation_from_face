# skin_segment_robust
Using Fuzzy c-means clustering, this method of skin segmentation works for all skin tones and a variety of illumination conditions. Shadows and specular reflection are mostly ignored, providing a decent ROI of skin pixels.

The file contains three functions :
1. detect_face(): This function detects the face in an image (currently constrained to one face only, can be easily changed) using the Viola Jones face detection method. Classifiers have been included as .xml files here. These classifiers are used to detect a bounding box containing the face.
2. crop_face(): This function extracts the face region from the image, given the boundary from detect_face().
3. skin_segment(): This function classifies pixels into skin/non-skin by the following process:
              - Converting image from RGB to CIELUV colorspace. All of the components in the RGB colorspace contain brightness information. In order to make the segmentation robust to illumination changes, we need to separate brightness from color. The L component in the LUV colorspace contains brightness information, while U and V contain chrominance information.
              - Feature extraction. We use four features for classification - L,U,V, and the Euclidean distance from the centre of the face. This is based on the assumption that the pixels close to the centre of the face will be skin pixels, and that the farther the pixels are from the centre, the more likely they are to be non-skin pixels. Also, in this case, pixels in proximity to each other are more likely to belong to the same cluster. The distance feature especially helps distinction in cases where the background/clothes are similar to the color of the skin.
              - Assigning weights to features. Since the color differences are more important than differences in brightness and distance from the centre (this was empirically seen, and is also intuitive), we assign more weight to the U and V components than to brightness and distance. These weights are empirically determined.
              - With 
