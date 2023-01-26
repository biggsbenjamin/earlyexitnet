 #python cli.py -m b_lenet_fcn -bbe 50 -jte 100

 #python cli.py -m b_lenet_se -bbe 50 -jte 100

 #python cli.py -m brnfirst_se -bbe 100
 #python cli.py -m brnfirst -bbe 100
 #python cli.py -m brnsecond -bbe 100

python cli.py -m b_alexnet_cifar -bbe 130 -jte 350 -t1 0.9 -entr 0.0001 -gpu 3 -wrks 6 -d cifar10 -rn "b alexnet - dropout, pre train, then jnt, LRN layers"

python cli.py -m b_alexnet_cifar -bbe 0 -jte 400 -t1 0.9 -entr 0.0001 -gpu 3 -wrks 6 -d cifar10 -rn "b alexnet - dropout, jnt from scratch training - trying to get higher val accuracy, LRN layers"

