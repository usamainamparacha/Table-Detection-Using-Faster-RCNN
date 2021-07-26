# Pytorch Implementation of FasterRCNN Table Detection
In this notebook, I am going to finetune FasterRCNN model for detecting tables in documents. 

## Inputs To FasterRCNN:
	During Training:
		1) List of tensors, each of shape [C, H, W], one for each image.
		2) Targets: It is a dictionary for each image containing:
			a) boxes('FloatTensor[N, 4]') like [x1, y1, x2, y2] for each table bounding box in image
			b) labels(Int64Tensor[N]): the class label for each ground-truth box 
	
## Inferencing FasterRCNN:
	1) We give the input tensors to model
	2) It returns the post-processed predictions as a List[Dict[Tensor]]
		a) boxes('FloatTensor[N, 4]') like [x1, y1, x2, y2] for each predicted table bounding box 
		b) labels(Int64Tensor[N]): the predicted label for each image
		c) scores(Tensor[N]): the score for each prediction 
    
