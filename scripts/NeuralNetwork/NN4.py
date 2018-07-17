from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import argparse
import sys

from argparse import ArgumentParser

from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("--learnRate", help="display a square of a given number",
                    type=float)

parser.add_argument("--epochs", help="display a square of a given number",
                    type=int)

parser.add_argument("--batchSize", help="display a square of a given number",
                    type=int)

parser.add_argument("--modelChoice", help="display a square of a given number",
                    type=int)

parser.add_argument("--iterations", help="display a square of a given number",
                    type=int)

parser.add_argument("--regularization", help="display a square of a given number",
                    type=float)

parser.add_argument("--l1", help="display a square of a given number", default = 0,
                    type=int)

parser.add_argument("--l2", help="display a square of a given number", default = 0,
                    type=int)

parser.add_argument("--l3", help="display a square of a given number", default = 0,
                    type=int)

parser.add_argument("--l4", help="display a square of a given number", default = 0,
                    type=int)

parser.add_argument("--l5", help="display a square of a given number", default = 0,
                    type=int)

parser.add_argument("--multiClass", help="display a square of a given number", default = False,
                    type=bool)

parser.add_argument("--dataSet", help="display a square of a given number", default = False,
                    type=str)

parser.add_argument("--TrainTestSplitRatio", help="display a square of a given number", default = False,
                    type=float)

parser.add_argument("--confMat", help="display a square of a given number", default = False,
                    type=str)

parser.add_argument("--splitRandom", help="display a square of a given number", default = True,
                    type=bool)


args = parser.parse_args()

print(args)

# ==============================================================================
# batch-function, TODO: shuffle and get random
# ==============================================================================
batchIndex = 0
def nextBatchX(M,batch_size):
    global batchIndex
    v = M[batchIndex:batchIndex+batch_size,:]
    return v

def nextBatchY(M,batch_size):
    global batchIndex
    v = M[batchIndex:batchIndex+batch_size]
    return v

# ==============================================================================
# Read stuff in and create X and Y Training and Testing
# ==============================================================================
#CSV_PATH = '/home/willy/10.Semester/Robotik/MyRobot/SerialPortListener/Training/f2/NNtraining.csv'

CSV_PATH = '/home/willy/10.Semester/Robotik/MyRobot/SerialPortListener/Training/NNtrainingALL.csv'

CSV_PATH = args.dataSet

#CSV_PATH = '/home/willy/10.Semester/Robotik/MyRobot/SerialPortListener/Training/s4/NNtraining.csv'

df=pd.read_csv(CSV_PATH)

df=df.dropna(axis='index')

cols = len(df.columns)
features = cols-1

Xdf = df.iloc[:,0:(cols-1)]
X = Xdf.values
Ydf = df.iloc[:,cols-1]
Yval = Ydf.values

Y = Yval

nClasses = len(set(Y[:]))

print("found " + str(nClasses) + " classes (" + str(set(Y[:])) + ")")

numOfCollisionsTotal = sum(Y[:])

TP_count = 0
FP_count = 0
TN_count = 0
FN_count = 0


print(numOfCollisionsTotal)

X_orig = X
Y_orig = Y


def non_shuffling_train_test_split(X, y, test_size=0.2):
    i = int((1 - test_size) * X.shape[0]) + 1
    X_train, X_test = np.split(X, [i])
    y_train, y_test = np.split(y, [i])
    return X_train, X_test, y_train, y_test


def do_one_iteration():
    X_train, X_test, y_train, y_test = non_shuffling_train_test_split(X_orig, Y_orig, test_size=args.TrainTestSplitRatio)


    if args.splitRandom == True:
        X_train, X_test, y_train, y_test = train_test_split(X_orig, Y_orig,test_size=args.TrainTestSplitRatio)
        print(sum(y_test[:]))

        while sum(y_test[:]) / sum(y_train[:]) < args.TrainTestSplitRatio:
            X_train, X_test, y_train, y_test = train_test_split(X_orig, Y_orig,test_size=args.TrainTestSplitRatio)


        print("y_train: ", sum(y_train[:]), " y_test: ", sum(y_test[:]))

        quotient = 1-sum(Y_orig[:])/len(Y_orig[:])
        print("#collisions / #noCollisions", quotient)



        ratio = sum(Y_orig[:])/len(Y_orig[:])

        print(ratio)





    learning_rate = args.learnRate
    training_epochs = args.epochs
    batch_size = args.batchSize
    display_step = 1


    # Network Parameters
    #n_hidden_1 = 2 # 1st layer number of neurons
    #n_hidden_2 = 2 # 2nd layer number of neurons
    #n_input = features # MNIST data input (img shape: 28*28)
    #n_classes = 2 # MNIST total classes (0-9 digits)

    n_hidden_1 = args.l1 # 1st layer number of neurons
    n_hidden_2 = args.l2 # 2nd layer number of neurons
    n_hidden_3 = args.l3
    n_hidden_4 = args.l4
    n_hidden_5 = args.l5

    n_input = features # MNIST data input (img shape: 28*28)

    #n_classes = 2 # MNIST total classes (0-9 digits)
    n_classes = nClasses

    model_choice =  1 # "logistic_regression"
    #model_choice =  2 # "two_layer_perceptron"
    #model_choice = 3 # "multilayer_perceptron_relu"


    # tf Graph input
    X = tf.placeholder("float", [None, n_input], name = 'X')
    Y = tf.placeholder("float", [None, n_classes], name = 'Y')


    if args.modelChoice == 1: # "logistic_regression"
        n_hidden_1 = n_classes
        n_hidden_2 = n_classes


    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    if args.modelChoice == 4: # "multilayer"
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
            'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
            'h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),

            'out': tf.Variable(tf.random_normal([n_hidden_5, n_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'b3': tf.Variable(tf.random_normal([n_hidden_3])),
            'b4': tf.Variable(tf.random_normal([n_hidden_4])),
            'b5': tf.Variable(tf.random_normal([n_hidden_5])),

            'out': tf.Variable(tf.random_normal([n_classes]))
        }

    # Create model
    def multilayer_perceptron(x):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
        return out_layer

    # Create model
    def multilayer_perceptron2(x):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        # Output fully connected layer with a neuron for each class
        out_layer = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out']), biases['out']))
        return out_layer

    # Create model
    def multilayer_perceptron3(x):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.nn.sigmoid(tf.matmul(layer_2, weights['out']) + biases['out'])
        return out_layer

    # Create model
    def multilayer_perceptron4(x):
        out_layer = tf.nn.sigmoid(tf.matmul(x, weights['h1']) + biases['out'])
        return out_layer

    # Create model
    def logistic_regression(x):
        out_layer = tf.matmul(x, weights['h1']) + biases['b1']
        return out_layer

    # Create model
    def logistic_regression_relu(x):
        out_layer = tf.nn.relu(tf.matmul(x, weights['h1']) + biases['b1'])
        return out_layer

    # Create model
    #def logistic_regression(x):
    #    out_layer = tf.nn.softmax(tf.matmul(x, weights['h1']) + biases['b1'])
    #    return out_layer

    # Create model
    def multilayer_perceptron6(x):
        layer_1 = tf.nn.softmax(tf.matmul(x, weights['h1']) + biases['b1'])

        layer_2 = tf.nn.softmax(tf.matmul(layer_1, weights['h2']) + biases['b2'])

        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer

    # Create model
    def two_layer_perceptron(x):
        layer_1 = tf.nn.softmax(tf.matmul(x, weights['h1']) + biases['b1'])

        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        return out_layer

    def multilayer_perceptron_relu(x):
        layer_1 = tf.nn.relu(tf.matmul(x, weights['h1']) + biases['b1'])

        layer_2 = tf.nn.relu(tf.matmul(layer_1, weights['h2']) + biases['b2'])

        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer

    def small_relu(x):
        layer_1 = tf.nn.relu(tf.matmul(x, weights['h1']) + biases['b1'])

        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        return out_layer

    def multilayer_perceptron_relu_big(x):
        layer_1 = tf.nn.relu(tf.matmul(x, weights['h1']) + biases['b1'])

        layer_2 = tf.nn.relu(tf.matmul(layer_1, weights['h2']) + biases['b2'])

        layer_3 = tf.nn.relu(tf.matmul(layer_2, weights['h3']) + biases['b3'])

        layer_4 = tf.nn.relu(tf.matmul(layer_3, weights['h4']) + biases['b4'])

        layer_5 = tf.nn.relu(tf.matmul(layer_4, weights['h5']) + biases['b5'])

        out_layer = tf.matmul(layer_5, weights['out']) + biases['out']
        return out_layer

    # Construct model

    logits = logistic_regression(X)

    beta = 0.00001 # 

    if args.modelChoice == 1: # "logistic_regression"
        logits = logistic_regression(X)
        beta = args.regularization # 

    elif args.modelChoice == 2: # "two_layer_perceptron":
        logits = two_layer_perceptron(X)
        beta = args.regularization # 

    elif args.modelChoice == 3: # "multilayer_perceptron_relu":
        logits = multilayer_perceptron_relu(X)
        beta = args.regularization # 


    elif args.modelChoice == 4: # "multilayer_perceptron_relu":
        logits = multilayer_perceptron_relu_big(X)
        beta = args.regularization # 

    #--------------------------------------------------
    #ratio = 31.0 / (500.0 + 31.0)
    #class_weight = tf.constant([[ratio, 1.0 - ratio]])

    c2 = np.count_nonzero(Y_orig == 1)/len(Y_orig)
    c3 = np.count_nonzero(Y_orig == 2)/len(Y_orig)
    c4 = np.count_nonzero(Y_orig == 3)/len(Y_orig)
    c5 = np.count_nonzero(Y_orig == 4)/len(Y_orig)
    c1 = 1 - c2 - c3 - c4 - c5

    print("-----------------------------\n Class_weights: 0: " + str(c1) + " 1: " + str(c2) + " 2:" + str(c3) + " 3: " + str(c4) +"\n-----------------------------")

    class_weights = tf.constant([[c1, c2]])
    if n_classes == 4:
        class_weights = tf.constant([[c1, c2, c3, c4]])

    if n_classes == 5:
        class_weights = tf.constant([[1/c1, 1/c2, 1/c3, 1/c4, 1/c5]])

    #class_weights = tf.convert_to_tensor(np.arange(n_classes), dtype=tf.float32)


    #class_weights = tf.constant([[0.0, 1.0, 2.0, 3.0]])
    
    #class_weights = tf.constant(tf.range(n_classes))
    #class_weights = tf.constant(np.array(np.arange(4.0), ndmin=2))

    #print(sess.run(class_weights))
    #quit()


    c_weights = tf.reduce_sum(class_weights * Y, axis=1)
    # compute your (unweighted) softmax cross entropy loss
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)
    # apply the weights, relying on broadcasting of the multiplication
    weighted_losses = unweighted_losses * c_weights
    # reduce the result to get your final loss
    
    loss_op = tf.reduce_mean(weighted_losses)

    #loss_op = tf.reduce_mean(unweighted_losses)

    
    #Y = (Y_orig[:,None] == np.arange(n_classes)).astype(int)
    #Y = tf.cast(Y, tf.float32)

    print("Y_orig: ", Y_orig, " ", len(Y_orig))
    print("Y: ", Y)

    # Define loss and optimizer
    #loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

    #loss_op = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=Y, pos_weight=class_weight))

    # Loss function using L2 Regularization
    regularizers = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2']) + tf.nn.l2_loss(weights['out'])

    loss_op = tf.reduce_mean(loss_op + beta * regularizers)

    optimizer = tf.train.AdamOptimizer(learning_rate=args.learnRate)
    #optimizer = tf.train.GradientDescentOptimizer(0.5)

    # unweighted
    train_op = optimizer.minimize(loss_op)


    # Initializing the variables
    init = tf.global_variables_initializer()

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    #with tf.Session() as sess:
    #    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(X_train)/batch_size)
        # Loop over all batches
        for batchIndex in range(total_batch):
            batch_x = nextBatchX(X_train, batch_size)
            batch_y = nextBatchY(y_train, batch_size)    
            #batch_y = np.reshape(batch_y, (batch_size, 1))
            #print(batch_y)
        
            batch_y = (batch_y[:,None] == np.arange(n_classes)).astype(int)
            #print("rearanged:")
            #print(batch_y)

            #a = np.array([1,4,2,3,1,2,1,4])
            #b = (a[:,None] == np.arange(a.max())+1)
            #print(b)


            # unweigthed
            # Run optimization op (backprop) and cost op (to get loss value)
            #_, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})

            # weighted
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})

            # Compute average loss
            avg_cost += c / total_batch

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")


    # Test model
    {X: X_test, Y: y_test}

    # original
    pred = tf.nn.softmax(logits)  # Apply softmax to logits

    #pred = logits

    #pred = logits  # Apply softmax to logits

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    y_test = (y_test[:,None] == np.arange(n_classes)).astype(int)
    y_test2 = y_test

    print("Accuracy:", accuracy.eval({X: X_test, Y: y_test}))


    y_hat = sess.run(pred, {X: X_test, Y: y_test})

    write_to_file_path = "y.txt";
    output_file2 = open(write_to_file_path, "w+");
    for item in y_hat:
      output_file2.write("%s,\n" % ' '.join(map(str, item)))

    y_hat = np.where(y_hat<=0.5, 0, y_hat)
    y_hat = np.where(y_hat>0.5, 1, y_hat)

    y_hat = y_hat[:,1]
    y_test = y_test[:,1]


    def perf_measure(y_actual, y_hat):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        perc=0

        for i in range(len(y_hat)): 
            if y_actual[i]==y_hat[i]==1.0:
                TP += 1
            if y_hat[i]==1.0 and y_actual[i]!=y_hat[i]:
                FP += 1
            if y_actual[i]==y_hat[i]==0.0:
                TN += 1
            if y_hat[i]==0.0 and y_actual[i]!=y_hat[i]:
                FN += 1

        if FN != 0:
            perc=TP/FN

        return(TP, FP, TN, FN, perc)


    TP, FP, TN, FN, perc= perf_measure(y_test,y_hat)


	


    #TP_count = TP_count + TP
    #FP_count = FP_count + FP
    #TN_count = TN_count + TN
    #FN_count = FN_count + FN

    print("TP:",TP, "FP:",FP, "TN:",TN,"FN:",FN, "TP/FN:", perc , "#collisions / #noCollisions: ", str(quotient));


    #y_test2 = (y_test[:,None] == np.arange(n_classes)).astype(int)

    y_hat2 = sess.run(pred, {X: X_test, Y: y_test2})

    #(y_hat2 == y_hat2.max(axis=1)[:,None]).astype(int)
    #y_hat2 = np.where(y_hat2 < 1, 0, y_hat2)

    print(y_test2)
    print(y_hat2)

    #ind = np.unravel_index(np.argmax(a, axis=None), a.shape)

    #y_hat2 = np.unravel_index(np.argmax(y_hat2, axis=1), y_hat2.shape)
    y_hat2 = np.argmax(y_hat2, axis=1)
    y_test2 = np.argmax(y_test2, axis=1)
    print(y_test2)
    print(y_hat2)

    confusion = tf.confusion_matrix(labels=y_test2, predictions=y_hat2, num_classes=n_classes)

    df_confusion = pd.crosstab(y_test2, y_hat2)
    df_confusion.to_csv(args.confMat, header = ["Kollision vorhergesagt", "keine Kollision vorhergesagt"], index_label = ["Keine Kollision", "Kollision"])

    print("Confusion: by row: y_test, by column: y_hat")
    mat = sess.run(confusion)
    print(mat)

    #np.savetxt(args.confMat, mat, delimiter=",", header="Keine Kollision, Kollision")

    goods = np.zeros(n_classes)
    
    for i in range(1, n_classes):
        goods[i] = mat[i,i]/(sum(mat[i,:]) + sum(mat[:,i]) - mat[i,i])

    print(goods)

    su_sq_po_pred = sum(map(lambda x:x*x,goods))
    print("sum of squared positive predictionratios: " + str(su_sq_po_pred))


    #---------------------------------------------------------------------------------------
    # make a single test
    #---------------------------------------------------------------------------------------

    pred_small = tf.nn.softmax(logits)  # Apply softmax to logits
    # pred_small = logits
    x_small = X_test[0,:]
    x_small = x_small.reshape(1,features)
    y_small = (y_test[0,None] == np.arange(n_classes)).astype(int)
    y_small = y_small.reshape(1,n_classes)
    y_hat = sess.run(pred_small, {X: x_small, Y: y_small})



    # ==============================================================================
    # Save the model to model.txt, so you can just copy paste it into arduino
    # ==============================================================================
    write_to_file_path = "/home/willy/10.Semester/Robotik/MyRobot/Trex2/model.h";
    output_file = open(write_to_file_path, "w+");
    output_file.write("// --------------------------------------------------\n")
    output_file.write("// Model begins here. <---------- copy-paste here!\n")
    output_file.write("// training_epochs: " + str(training_epochs) + "\n")
    stri = "// TP:" + str(TP) + " FP:" + str(FP) + " TN:" + str(TN) + " FN:" + str(FN) + " TP/FN:" + str(perc) + " #collisions / #noCollisions " + str(quotient) + "\n"
    output_file.write(stri)
    output_file.write("// --------------------------------------------------\n\n")

    output_file.write("const int inputNeurons = " + str(features) + ";\n")
    output_file.write("const int n_hidden_1 = " + str(n_hidden_1) + ";\n")
    output_file.write("const int n_hidden_2 = " + str( n_hidden_2) + ";\n")
    output_file.write("const int n_classes = " +str(n_classes) + ";\n")
    output_file.write("const myFloat NN_Col_quotient = " +str(quotient) + ";\n")
    output_file.write("const int model_choice = " + str(args.modelChoice) + ";\n\n")

    #output_file.write("myFloat X[1][inputNeurons];\n\n")
    output_file.write("myFloat X[1][inputNeurons] = \n{\n")
    x_small = np.reshape(x_small, (1,features))
    for item in x_small:
      output_file.write("%s,\n" % '\n,'.join(map(str, item)))
    output_file.write("};\n\n")


    output_file.write("myFloat y_small[1][n_classes] = \n{\n")
    y_hat = np.reshape(y_hat, (1,n_classes))
    for item in y_hat:
      output_file.write("%s,\n" % '\n,'.join(map(str, item)))
    output_file.write("};\n\n")



    output_file.write("myFloat layer1_raw[1][n_hidden_1];\n")
    output_file.write("myFloat layer2_raw[1][n_hidden_2];\n")
    output_file.write("myFloat layer1[1][n_hidden_1];\n")
    output_file.write("myFloat layer2[1][n_hidden_2];\n")
    output_file.write("myFloat out_raw[1][n_classes];\n\n")


    output_file.write("myFloat h1[inputNeurons][n_hidden_1] = \n{\n")
    ts = sess.run(weights['h1'])
    ts = np.reshape(ts, (n_hidden_1*features, 1))
    for item in ts:
      output_file.write("%s,\n" % ' '.join(map(str, item)))
    output_file.write("};\n\n")

    h1_file = open("/home/willy/10.Semester/Robotik/MyRobot/Trex2/h1.txt", "w+");
    for item in ts:
      h1_file.write("%s,\n" % ' '.join(map(str, item)))

    output_file.write("myFloat h2[n_hidden_1][n_hidden_2] = \n{\n")
    ts = sess.run(weights['h2'])
    ts = np.reshape(ts, (n_hidden_1*n_hidden_2, 1))
    for item in ts:
      output_file.write("%s,\n" % ' '.join(map(str, item)))
    output_file.write("};\n\n")


    output_file.write("myFloat h_out[n_hidden_2][n_classes] = \n{\n")
    ts = sess.run(weights['out'])
    ts = np.reshape(ts, (n_hidden_2*n_classes, 1))
    for item in ts:
      output_file.write("%s,\n" % ' '.join(map(str, item)))
    output_file.write("};\n\n")


    output_file.write("myFloat b1[1][n_hidden_1] = \n{\n")
    ts = sess.run(biases['b1'])
    ts = np.reshape(ts, (n_hidden_1, 1))
    for item in ts:
      output_file.write("%s,\n" % ' '.join(map(str, item)))
    output_file.write("};\n\n")


    output_file.write("myFloat b2[1][n_hidden_2] = \n{\n")
    ts = sess.run(biases['b2'])
    ts = np.reshape(ts, (n_hidden_2, 1))
    for item in ts:
      output_file.write("%s,\n" % ' '.join(map(str, item)))
    output_file.write("};\n\n")


    output_file.write("myFloat out[1][n_classes] = \n{\n")
    ts = sess.run(biases['out'])
    ts = np.reshape(ts, (n_classes, 1))
    for item in ts:
      output_file.write("%s,\n" % ' '.join(map(str, item)))
    output_file.write("};\n\n")

    #-------------------------------------------------------------------------------
    output_file.write("void setVar(int ind, myFloat val, myFloat *arr){\n")
    output_file.write("arr[ind] = val;\n}\n")


    def write_test_case(ind):
      pred_small = tf.nn.softmax(logits)  # Apply softmax to logits
      #pred_small = logits
      x_small = X_test[ind,:]
      x_small = x_small.reshape(1,features)
      y_small = (y_test[ind,None] == np.arange(n_classes)).astype(int)
      y_small = y_small.reshape(1,n_classes)
      y_hat = sess.run(pred_small, {X: x_small, Y: y_small})

      t = "void test_case"+ str(ind) + "() {\n"
      output_file.write(t)
      for i in range(0, features):
        output_file.write(str("setVar(" + str(i) + ", "+ str(X_test[ind,i]) + ", *X);\n"));

      ts = np.reshape(y_hat, (n_classes, 1))
      for i in range(0,n_classes):
            output_file.write("setVar(" + str(i) +", " + str(ts[i])[1:-1] + ",*y_small);\n")
      output_file.write("\n}\n")

    write_test_case(1)
    write_test_case(2)
    write_test_case(3)
    write_test_case(4)


    #-------------------------------------------------------------------------------
    output_file.write("// --------------------------------------------------\n")
    output_file.write("// Model ends here. <---------- copy-paste until here!\n")
    output_file.write("// --------------------------------------------------\n\n")

    return(su_sq_po_pred)


pool = Pool()

su_sq_po_pred = [None] * args.iterations


#result1 = pool.apply_async(do_one_iteration)
#TP[0],FP[0],TN[0],FN[0] = result.get(timeout=10)

#result1 = pool.apply_async(do_one_iteration)
#TP[1],FP[1],TN[1],FN[1] = result.get(timeout=10)


result = [[None]] * args.iterations
for iteration in range(0,args.iterations):
    result[iteration] = pool.apply_async(do_one_iteration)


for iteration in range(0,args.iterations):
    su_sq_po_pred[iteration] = result[iteration].get(timeout=10000)


print(su_sq_po_pred)

ev_F_name = "/home/willy/10.Semester/Robotik/MyRobot/SerialPortListener/NeuralNetwork/Models/comparison.txt" 
eval_file = open(ev_F_name, "a");
#stri2 = str(sum(TP)/(sum(FP)+sum(FN))) + " Features: " + str(features) + " TP:" + str(sum(TP)/args.iterations) + " FP:" + str(sum(FP)/args.iterations) + " TN:" + str(sum(TN)/args.iterations) + " FN:" + str(sum(FN)/args.iterations) + " It: " + str(args.iterations) + " Model: " + str(args.modelChoice) + " Ep: " + str(args.epochs) + " Bs: " + str(args.batchSize) + " Lr: " + str(args.learnRate) + " Reg: " + str(args.regularization) + "\n"



stri2 = str(sum(su_sq_po_pred)/args.iterations) + " It: " + str(args.iterations) + " Model: " + str(args.modelChoice) + " Ep: " + str(args.epochs) + " Bs: " + str(args.batchSize) + " Lr: " + str(args.learnRate) + " Reg: " + str(args.regularization) + "\n"
eval_file.write(stri2)








