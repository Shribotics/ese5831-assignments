## This is a readme file for Assignmnet-5 of ECE-5831 by Shrikant Pawar

1. rock-paper-scissors.py
-Implement the Mnist class and test with your own images.
2. mnist.py for Mnist
- Create five images per digit with your own handwriting. 
- Image size: 28 x 28 x 1 (grayscale)
3. module5.py with two input arguments (1st: image filename, 2nd: digit of the image)
ex) $ python module5.py 3_2.png 3

The output will be 
When the result is not right,
Fail: Image 2_1.png is for digit 2 but the inference result is 3. 

When the result is right,
Success: Image 2_1.png is for digit 2 is recognized as 2. 

module5.ipynb
import mnist.py and use it to show the class works step by step.

# Conclusion:
The model used in the module5.py can predict the numbers correctly only when the background is dark and nubers printed with white color. This model need more training and more data to improve the prediction accuracy.