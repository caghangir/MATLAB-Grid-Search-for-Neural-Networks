I created this function for my projects to find best hyper-parameters of Neural Networks. 
It takes 5 different hyper-parameters such as; 
- hidden node size of layer 1 
- hidden node size of layer 2 
- optimizer type 
- maximum epoch 
- transfer function 
There is an example code block top of the function. You just add which hyper-parameters you want to try. Function will try 10-fold cross validation of each combination that is created using your hyper-parameters. Finally, find best hyper-parameter combination and return these as cell. You can use the cell directly as parameters of your Neural Network. 
You need to give train with label, function automatically create %90 train %10 validation dataset. 
New version is coming with more features.
