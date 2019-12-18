# skin_segment_robust
This is an unsupervised learning based method for segmentation of skin from the face.
Using Fuzzy c-means clustering, this method of skin segmentation works for all skin tones and a variety of illumination conditions. Shadows and specular reflection are mostly ignored, providing a decent ROI of skin pixels on a given face.

This is done in two steps:
1. Face detection: Using the standard Viola-Jones method for face detection (haar cascades data files included in the *classifiers* folder), the face is detected and extracted from the image. Currently, this code has been limited to one face only.
2. Skin segmentation: This function classifies pixels into skin/non-skin by the following process:
              
	 - Converting image from RGB to CIELUV colorspace. All of the components in the RGB colorspace contain brightness information. In order to make the segmentation robust to illumination changes, we need to separate brightness from color. The L component in the LUV colorspace contains brightness information, while U and V contain chrominance information.
              
	 - Feature engineering. Four features have been identified - L,U,V, and the Euclidean distance from the centre of the face. This is based on the assumption that the pixels close to the centre of the face will be skin pixels, and that the farther the pixels are from the centre, the more likely they are to be non-skin pixels. The distance feature especially helps distinction in cases where the background/clothes are similar to the color of the skin.
              
	  - Assigning weights to features. Since the color differences are more important than differences in brightness and distance from the centre (this was seen during experimentation, and is also intuitive), the U and V components have been assigned more weight than brightness and distance. These weights are empirically determined.
              
	  - Clustering. By the definition of the problem, we have two clusters - "skin" and "not skin". The fuzzy C means algorithm provides better control over the forming of clusters. For each pixel, it assigns a cluster score which gives the probability of the pixel belonging to that cluster. Thresholding over these scores gives the final binary mask for skin segmentation.
	  
The file SkinSegmentation.py contains all the functions for the above. It can be run directly to see demo results, or can be imported for your own testing and application.
